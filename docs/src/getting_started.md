# [Installation & First Model](@id getting_started_page)

This page takes you from a clean Julia installation to an estimated vector autoregression with identified impulse responses. Every example runs against the built-in FRED-MD panel, so no downloads, API keys, or external files are involved. When the workflow is clear, [Choosing a Method](@ref method_guide_page) routes you to the estimator your research question actually calls for.

MacroEconometricModels.jl requires **Julia 1.10 or newer**. A single `using MacroEconometricModels` brings every estimator, test, transformation, table, and plotting function into scope.

## Quick Start

```@setup gs
using MacroEconometricModels, Random
Random.seed!(42)
```

**Recipe 1: Install the package and confirm the load**

Install from the Julia General registry:

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

Load the package and check which version is active:

```@example gs
using MacroEconometricModels
pkgversion(MacroEconometricModels)
```

**Recipe 2: Estimate your first VAR**

A **vector autoregression** (Sims 1980) regresses each variable on lags of every variable in the system. FRED-MD (McCracken & Ng 2016) is the monthly US macro panel that ships with the package: `apply_tcode` applies each series' codebook transformation to stationarity, `fix` drops the leading rows that differencing fills with `NaN`, and `estimate_var` fits the system by equation-wise OLS:

```@example gs
fred = load_example(:fred_md)
d = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
model = estimate_var(d, 4)
report(model)
```

`report` prints the specification, the per-equation fit statistics, a seven-column coefficient table with significance stars for each equation, the residual covariance and correlation matrices, and a stationarity check. Each equation describes how one variable responds to the recent history of the whole system: the `FEDFUNDS` equation is the estimated monetary policy rule. The largest companion eigenvalue is 0.70, comfortably inside the unit circle, so the system is stable and impulse responses converge.

**Recipe 3: Identify shocks and trace impulse responses**

Reduced-form coefficients are not shocks. Identification maps the estimated residuals into structural disturbances, and an **impulse response function** traces one structural shock through the system over time (Lutkepohl 2005). The recursive (Cholesky) scheme orders the variables so that each shock affects only itself and the variables below it on impact:

```@example gs
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
report(result)
```

The horizon-``h`` entry of the `FEDFUNDS` block is the response of each variable to a one-standard-deviation monetary policy shock ``h`` months after impact; starred entries have bootstrap confidence intervals excluding zero. The impact response of `FEDFUNDS` to its own shock, 0.43, is the estimated standard deviation of that shock in percentage points. Responses decay within a year because the transformed series are growth rates and differences with little persistence. With this ordering the policy shock is the `FEDFUNDS` innovation orthogonal to contemporaneous output and prices.

**Recipe 4: Stay inside the data containers**

`apply_tcode` returns a `TimeSeriesData` that carries variable names, frequency, and transformation codes, and every estimator accepts it directly — no manual matrix extraction. `describe_data` summarizes it and `diagnose` reports `NaN`, `Inf`, and constant columns before you estimate anything:

```@example gs
describe_data(d)
```

The codebook assigns a different transformation to each series: `INDPRO` becomes a log difference (monthly output growth), `CPIAUCSL` a second log difference (the change in monthly inflation), and `FEDFUNDS` a first difference in percentage points. `inverse_tcode` maps forecasts back to levels. [Data Management](@ref data_page) covers the containers, all seven transformation codes, panel construction with `xtset`, and repair strategies in full.

---

## From Data to Results

Every workflow in the package follows the same four stages. The functions differ by model family; the shape does not.

| Stage | Function | Page |
|-------|----------|------|
| Load and inspect | `load_example`, `describe_data`, `diagnose` | [Data Management](@ref data_page) |
| Transform to stationarity | `apply_tcode`, `fix`, `apply_filter` | [Data Management](@ref data_page) |
| Estimate | `estimate_var`, `estimate_bvar`, `estimate_lp`, ... | [Choosing a Method](@ref method_guide_page) |
| Analyze and report | `irf`, `fevd`, `historical_decomposition`, `report`, `plot_result` | [Innovation Accounting](@ref innovation_accounting_page) |

Results render as interactive, self-contained D3.js documents:

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/quickstart_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

`plot_result` returns a `PlotOutput` that displays inline in Pluto, Jupyter, and VS Code, and writes to disk with `save_plot(p, "irf.html")`. The file has no external dependencies. [Visualization](@ref plotting_page) lists every supported result type.

---

## [Where to Go Next](@id getting_started_next)

- [Choosing a Method](@ref method_guide_page) — decision tables mapping a research question to the right estimator and page
- [Data Management](@ref data_page) — containers, transformations, validation, and panel operations
- [VAR](@ref var_page) — the full VAR toolkit beyond this page: forecasting, conditional forecasts, stability
- [Structural Identification](@ref structural_identification_page) — sign, narrative, long-run, and heteroskedasticity-based schemes
- [Innovation Accounting](@ref innovation_accounting_page) — IRF, FEVD, and historical decomposition workflows
- [Hypothesis Tests](@ref tests_page) — unit-root, cointegration, break, and diagnostic tests
- [Notation](@ref notation) — the symbol dictionary used throughout the documentation

---

## Complete Example

A four-variable monetary VAR, from raw vintage to a variance decomposition. The lag order is chosen by BIC rather than assumed, and the Cholesky ordering places the policy rate last so that the monetary shock is orthogonal to contemporaneous real activity, prices, and unemployment:

```@example gs
fred = load_example(:fred_md)
d = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "UNRATE", "FEDFUNDS"]]))

p = select_lag_order(to_matrix(d), 12; criterion=:bic)
m = estimate_var(d, p)
report(m)
```

BIC selects two lags out of a search range of twelve: once the transformation codes have rendered each series stationary, two months of own- and cross-dynamics absorb most of the predictable variation, and BIC's system-wide parameter penalty makes the remaining gains too expensive. Always check that the selected order is interior to the search range — a criterion that returns `max_p` itself means the range is too narrow. Decompose the forecast error variance over a two-year horizon:

```@example gs
vd = fevd(m, 24; method=:cholesky)
report(vd)
```

Each row gives the share of one variable's ``h``-step forecast error variance attributable to each structural shock, and the shares sum to 100% across the row at every horizon. Own shocks dominate throughout: 94% for industrial production growth and 97% for the CPI series at two years. Unemployment is the exception — 36% of its forecast error traces to industrial production shocks even on impact, the Okun's-law link recovered without imposing it. The monetary shock's contribution to industrial production grows from zero on impact to 1.2% at two years, the delayed transmission of policy to real activity.

---

## Common Pitfalls

1. **Julia below 1.10.** Check with `VERSION` at the REPL. `Pkg.add` refuses to install the package on older releases, and the resolver reports an unsatisfiable `julia` compatibility bound rather than a missing package.

2. **The first `using` is slow.** Precompilation runs once per package version and takes on the order of a minute. Every subsequent session loads in a few seconds. Do not interrupt the first load.

3. **Data must be ``T \times n``.** Rows are time periods, columns are variables. A transposed matrix estimates a different model or throws a dimension error. Passing a `TimeSeriesData` removes the ambiguity entirely.

4. **Raw FRED-MD levels are non-stationary.** Estimating a VAR on untransformed levels produces spurious dynamics and an unstable companion matrix. Always run `apply_tcode` first, or apply your own differencing.

5. **Differencing leaves `NaN` in the first rows.** Estimators reject non-finite input. `fix(d)` drops those rows by listwise deletion; `fix(d; method=:interpolate)` fills interior gaps instead.

6. **Cholesky identification depends on the ordering.** The recursive scheme is not invariant to the column order of the data. Order variables from slowest- to fastest-moving, and check the robustness of your conclusions against the schemes in [Structural Identification](@ref structural_identification_page).

7. **Bootstrap bands are random.** Set `seed=` on `irf` (or seed the global RNG) when confidence intervals must be reproducible across sessions.

---

## References

- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574--589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1--48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)
