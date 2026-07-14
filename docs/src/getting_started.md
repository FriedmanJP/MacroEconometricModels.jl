# [Installation & First Model](@id getting_started_page)

This tutorial takes you from a clean Julia install to an estimated vector autoregression with plotted impulse responses in about ten minutes. It uses the built-in FRED-MD dataset, so no external files or downloads are required. Every code block below runs exactly as shown.

Once you finish, the [Choosing a Method](@ref method_guide_page) guide routes you to the right estimator for your data, and the per-topic pages listed under [Where to Next](@ref getting_started_next) cover each model family in depth.

---

## Install

MacroEconometricModels.jl requires **Julia 1.10 or newer**. Install the package from the Julia General registry:

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

Then load it into your session:

```julia
using MacroEconometricModels
```

That single `using` brings every estimator, test, and plotting function into scope.

---

## Load Data

```@setup gettingstarted
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
```

The package ships with several standard macroeconometric datasets. Here we use **FRED-MD**, the monthly US macro panel of McCracken & Ng (2016). We pull three series — industrial production (`INDPRO`), the consumer price index (`CPIAUCSL`), and the federal funds rate (`FEDFUNDS`) — and apply each series' recommended stationarity transformation with `apply_tcode`:

```@example gettingstarted
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]  # drop rows with missing/transform gaps
size(Y)
```

`Y` is now a numeric matrix with one column per variable and one row per month. `apply_tcode` differences and log-transforms each series as prescribed by the FRED-MD codebook, so the columns are output growth, inflation, and the change in the policy rate.

---

## Estimate a VAR

A **vector autoregression** (Sims 1980) regresses each variable on lags of every variable, capturing the joint dynamics of the system. Fit a VAR with four lags:

```@example gettingstarted
model = estimate_var(Y, 4)
report(model)
```

The `report` table shows the estimated coefficient matrices, equation-by-equation fit statistics, and the AIC/BIC information criteria used for lag selection. Each column of coefficients describes how one variable responds to the recent history of the whole system.

---

## Compute Impulse Responses

An **impulse response function (IRF)** traces how a one-time structural shock propagates through the system over time. We identify the shocks recursively (Cholesky) and attach bootstrap confidence bands:

```@example gettingstarted
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
nothing # hide
```

`result` holds the point-estimate responses over a 20-month horizon together with bootstrap confidence intervals for every variable-shock pair. With the Cholesky ordering used here, a monetary-policy shock (the `FEDFUNDS` innovation, ordered last) feeds through to output and prices, letting you read off the sign and persistence of the transmission.

---

## Visualize

Render the impulse responses as an interactive, self-contained HTML figure:

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/quickstart_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

`plot_result` returns a `PlotOutput` you can display inline or write to disk with `save_plot(p, "irf.html")`. The figure is a standalone D3.js document with no external dependencies.

---

## [Where to Next](@id getting_started_next)

You have estimated a model and plotted its dynamics. From here:

- [Data Management](@ref data_page) — loading, transforming, and reshaping time-series, panel, and cross-sectional data
- [Time Series Filters](@ref filters_page) — HP, Hamilton, Baxter-King, and Beveridge-Nelson trend-cycle decompositions
- [VAR](@ref var_page) — the full VAR toolkit: identification, FEVD, historical decomposition, forecasting
- [Linear Regression](@ref regression_page) — OLS/WLS/IV with robust and clustered standard errors
- [Panel VAR](@ref pvar_page) — dynamic panels with GMM estimation
- [DSGE Models](@ref dsge_page) — structural equilibrium models: solving, estimation, and analysis
- [Structural Identification](@ref structural_identification_page) — sign, narrative, long-run, and heteroskedasticity-based schemes
- [Statistical Identification](@ref nongaussian_page) — ICA and non-Gaussian SVAR identification
- [Innovation Accounting](@ref innovation_accounting_page) — IRF, FEVD, and historical decomposition workflows
- [Nowcasting](@ref nowcast_page) — real-time DFM, BVAR, and bridge-equation forecasts
- [Hypothesis Tests](@ref tests_page) — unit-root, cointegration, break, and diagnostic tests
- [Visualization](@ref plotting_page) — the full `plot_result` dispatch surface

---

## References

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574-589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)
- Sims, C. A. (1980). Macroeconomics and Reality.
  *Econometrica*, 48(1), 1-48. [DOI](https://doi.org/10.2307/1912017)
