# Visualization

MacroEconometricModels.jl includes a zero-dependency visualization system that renders interactive HTML/SVG charts using inline [D3.js](https://d3js.org/) v7. No additional packages are required — D3.js is loaded from CDN at runtime.

The unified `plot_result()` function dispatches on 31 result types, producing self-contained HTML documents with interactive tooltips that work in browsers and Jupyter notebooks.

## Quick Start

```julia
using MacroEconometricModels

# Load FRED-MD and prepare stationary 3-variable monetary system
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Estimate a VAR and plot IRFs
m = estimate_var(Y, 4)
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)

# Save to file
save_plot(p, "irf_plot.html")

# Open in browser
display_plot(p)
```

```@raw html
<iframe src="../assets/plots/quickstart_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Output Type

All `plot_result()` methods return a [`PlotOutput`](@ref) struct:

```julia
struct PlotOutput
    html::String  # Complete self-contained HTML document
end
```

### Displaying Results

| Context | How it works |
|---|---|
| **Jupyter/IJulia** | Automatic inline display via `MIME"text/html"` |
| **REPL** | `display_plot(p)` opens in default browser |
| **File** | `save_plot(p, "path.html")` writes HTML to disk |
| **Programmatic** | Access `p.html` directly |

```julia
# Save to file
save_plot(p, "my_plot.html")

# Open in default browser
display_plot(p)

# Access raw HTML
println(length(p.html), " bytes")
```

---

## Impulse Response Functions

### Frequentist IRF

```julia
# Stationary 3-variable system from FRED-MD
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

m = estimate_var(Y, 4)
r = irf(m, 20; ci_type=:bootstrap, reps=500)

# Full n_vars x n_shocks grid
p = plot_result(r)

# Single response variable and shock
p = plot_result(r; var=1, shock=1)

# Custom title
p = plot_result(r; title="Monetary Policy Shock")
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**kwargs**: `var` (Int/String), `shock` (Int/String), `ncols` (Int), `title` (String), `save_path` (String)

### Bayesian IRF

```julia
post = estimate_bvar(Y, 4; n_draws=1000, varnames=["INDPRO", "UNRATE", "CPI"])
r = irf(post, 20)
p = plot_result(r)                    # Full grid
p = plot_result(r; var=1, shock=1)    # Single panel
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Displays posterior median with credible band from the widest quantile interval.

### Local Projection IRF

```julia
lp_m = estimate_lp(Y, 1, 20; lags=4)
r = lp_irf(lp_m)
p = plot_result(r)          # All response variables
p = plot_result(r; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Structural LP

```julia
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
p = plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Forecast Error Variance Decomposition

### Frequentist FEVD

```julia
f = fevd(m, 20)
p = plot_result(f)          # All variables (stacked area)
p = plot_result(f; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/fevd_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Rendered as stacked area charts with proportions summing to 1.0.

### Bayesian FEVD

```julia
f = fevd(post, 20)
p = plot_result(f)
p = plot_result(f; stat=:mean)  # Default: posterior mean
```

```@raw html
<iframe src="../assets/plots/fevd_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### LP-FEVD

```julia
f = lp_fevd(slp, 20)
p = plot_result(f)
p = plot_result(f; bias_corrected=true)  # Default
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Historical Decomposition

### Frequentist HD

```julia
hd = historical_decomposition(m)
p = plot_result(hd)          # All variables
p = plot_result(hd; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Each variable produces two panels: stacked bar chart of shock contributions and a line chart comparing actual values with the sum of contributions.

### Bayesian HD

```julia
hd = historical_decomposition(post)
p = plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/hd_bayesian.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Time Series Filters

All five filter types produce a two-panel figure: trend vs. original series and the extracted cycle component.

### Hodrick-Prescott

```julia
# Log industrial production from FRED-MD (monthly, I(1))
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

p = plot_result(hp_filter(y))
```

```@raw html
<iframe src="../assets/plots/filter_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Hamilton (2018)

```julia
p = plot_result(hamilton_filter(y); original=y)
```

```@raw html
<iframe src="../assets/plots/filter_hamilton.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Beveridge-Nelson

```julia
p = plot_result(beveridge_nelson(y))
```

```@raw html
<iframe src="../assets/plots/filter_bn.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Baxter-King

```julia
p = plot_result(baxter_king(y); original=y)
```

```@raw html
<iframe src="../assets/plots/filter_bk.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Boosted HP (Phillips & Shi 2021)

```julia
p = plot_result(boosted_hp(y))
```

```@raw html
<iframe src="../assets/plots/filter_boosted_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

!!! note "Original series"
    Hamilton and Baxter-King filters produce shorter output than the input. Pass `original=y` to overlay the full original series in the plot.

---

## Forecasts

### ARIMA Forecast

```julia
# IP growth rate (log first difference) from FRED-MD
fred = load_example(:fred_md)
y1 = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

ar = estimate_ar(y1, 2)
fc = forecast(ar, 20)

# Forecast only
p = plot_result(fc)

# With recent history
p = plot_result(fc; history=y1, n_history=30)
```

```@raw html
<iframe src="../assets/plots/forecast_arima.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Volatility Forecast

```julia
# S&P 500 returns from FRED-MD (falls back to INDPRO growth)
sp_idx = findfirst(v -> occursin("S&P", v) && occursin("500", v), varnames(fred))
y_vol = sp_idx !== nothing ?
    filter(isfinite, apply_tcode(fred[:, varnames(fred)[sp_idx]], 5)) : y1

gm = estimate_garch(y_vol, 1, 1)
fc = forecast(gm, 10)
p = plot_result(fc)
p = plot_result(fc; history=gm.conditional_variance)
```

```@raw html
<iframe src="../assets/plots/forecast_volatility.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### VECM Forecast

```julia
# Cointegrated quarterly I(1) system from FRED-QD
qd = load_example(:fred_qd)
Y_ci = log.(to_matrix(qd[:, ["GDPC1", "PCECC96", "GPDIC1"]]))
Y_ci = Y_ci[all.(isfinite, eachrow(Y_ci)), :]

vecm = estimate_vecm(Y_ci, 2; rank=1)
fc = forecast(vecm, 10)
p = plot_result(fc)          # All variables
p = plot_result(fc; var=1)   # Single variable
```

```@raw html
<iframe src="../assets/plots/forecast_vecm.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Factor Forecast

```julia
# Large panel from FRED-MD (safe variables only, first 20 columns)
fred = load_example(:fred_md)
safe_idx = [i for i in 1:nvars(fred)
            if fred.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred.data[:, i])]
fred_safe = fred[:, varnames(fred)[safe_idx]]
X = to_matrix(apply_tcode(fred_safe))
X = X[all.(isfinite, eachrow(X)), 1:min(20, size(X, 2))]

fm = estimate_dynamic_factors(X, 2, 1)
fc = forecast(fm, 10)

p = plot_result(fc)                          # Factor forecasts
p = plot_result(fc; type=:observable, var=1) # Observable forecast
```

```@raw html
<iframe src="../assets/plots/forecast_factor.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### LP Forecast

```julia
# Use last 100 rows of the 3-variable FRED-MD data
Y_lp = Y[end-99:end, :]
lp_fc = estimate_lp(Y_lp, 1, 10; lags=4)
shock_path = zeros(10); shock_path[1] = 1.0
fc = forecast(lp_fc, shock_path)
p = plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_lp.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Volatility Models

ARCH, GARCH, EGARCH, and GJR-GARCH models produce a three-panel diagnostic figure: return series, conditional volatility, and standardized residuals with +/-2 standard deviation bounds.

```julia
# S&P 500 returns (or INDPRO growth as fallback)
p = plot_result(estimate_arch(y_vol, 2))
p = plot_result(estimate_garch(y_vol, 1, 1))
p = plot_result(estimate_egarch(y_vol, 1, 1))
p = plot_result(estimate_gjr_garch(y_vol, 1, 1))
```

```@raw html
<iframe src="../assets/plots/model_garch.html" width="100%" height="700" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Stochastic Volatility

The SV model shows posterior volatility with quantile credible bands:

```julia
m = estimate_sv(y_vol; n_samples=2000, burnin=1000)
p = plot_result(m)
```

```@raw html
<iframe src="../assets/plots/model_sv.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Factor Models

Factor models display a scree plot of eigenvalues and the extracted factor time series:

```julia
# Static factors (reusing X from factor forecast above)
p = plot_result(estimate_factors(X, 3))

# Dynamic factors
p = plot_result(estimate_dynamic_factors(X, 2, 1))
```

```@raw html
<iframe src="../assets/plots/model_factor_static.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Data Containers

### TimeSeriesData

```julia
# Plot raw FRED-MD series
fred = load_example(:fred_md)
d = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
p = plot_result(d)                               # All variables
p = plot_result(d; vars=["INDPRO", "CPIAUCSL"])  # Subset
```

```@raw html
<iframe src="../assets/plots/data_timeseries.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### PanelData

```julia
# Penn World Table: real GDP, population, employment, human capital
pwt = load_example(:pwt)
p = plot_result(pwt; vars=["rgdpna", "pop", "emp", "hc"])
```

```@raw html
<iframe src="../assets/plots/data_panel.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Panel data plots show each variable in a separate panel with one line per group.

---

## Nowcasting

### Nowcast Result

```julia
# Mixed-frequency panel from FRED-MD
fred = load_example(:fred_md)
nc_sub = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y_nc = to_matrix(apply_tcode(nc_sub))
Y_nc = Y_nc[all.(isfinite, eachrow(Y_nc)), :]
Y_nc = Y_nc[end-99:end, :]
Y_nc[end, end] = NaN  # Simulate missing observation

dfm_nc = nowcast_dfm(Y_nc, 4, 1; r=2, p=1)
nr = nowcast(dfm_nc)
p = plot_result(nr)
```

```@raw html
<iframe src="../assets/plots/nowcast_result.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Displays smoothed target variable with nowcast and forecast values annotated in the title.

### Nowcast News

```julia
X_old = copy(Y_nc)
X_new = copy(X_old); X_new[end, end] = X_old[end-1, end]  # Fill with previous value
dfm_news = nowcast_dfm(X_old, 4, 1; r=2, p=1)
nn = nowcast_news(X_new, X_old, dfm_news, 5)
p = plot_result(nn)
```

```@raw html
<iframe src="../assets/plots/nowcast_news.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Bar chart showing per-release impact on the nowcast revision.

---

## Common Options

All `plot_result()` methods accept these keyword arguments:

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `title` | `String` | `""` | Override auto-generated title |
| `save_path` | `String` or `nothing` | `nothing` | Auto-save HTML to path |
| `ncols` | `Int` | `0` (auto) | Number of columns in multi-panel grid |

Type-specific kwargs (e.g., `var`, `shock`, `history`, `stat`, `bias_corrected`) are documented for each method above.

---

## Chart Types

Three D3.js chart types cover all use cases:

| Chart | Used for | Features |
|---|---|---|
| **Line** | IRF, forecasts, filters, volatility, data series | CI bands, dashed lines, zero reference line, tooltips |
| **Stacked area** | FEVD | Proportions summing to 1.0, per-shock coloring |
| **Bar** | Historical decomposition, nowcast news | Stacked or grouped, diverging stack for negative values |

All charts include interactive tooltips and a consistent color palette.

## References

- Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents. *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301–2309. DOI: 10.1109/TVCG.2011.185
