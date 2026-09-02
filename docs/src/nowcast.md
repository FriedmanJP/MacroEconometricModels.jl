# [Nowcasting](@id nowcast_page)

Central banks and forecasters need current-quarter GDP estimates weeks before the official release. Nowcasting closes this gap by extracting signal from timely high-frequency indicators --- monthly industrial production, employment, and financial data --- to produce real-time estimates of quarterly aggregates. **MacroEconometricModels.jl** implements four estimators and one revision-attribution tool, all sharing the same mixed-frequency data layout: a dynamic factor model for large panels, a large Bayesian VAR for medium panels, bridge equations for a fast baseline, MIDAS for a single high-frequency indicator, and a news decomposition that attributes each nowcast revision to individual data releases.

This page owns the shared material --- the mixed-frequency data layout, the `nowcast()` extraction interface, the StatsAPI methods, and the result visualizations. Each child page owns one estimator: its model specification, keyword table, return fields, and worked examples.

```@setup nc
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN
```

## Quick Start

Estimate a two-factor dynamic factor model on a mixed-frequency FRED-MD panel --- four monthly indicators (`nM = 4`) and one quarterly target (`nQ = 1`) with a ragged edge in the most recent month --- and read off the current-quarter estimate:

```@example nc
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
report(dfm)
```

---

## Choosing a Method

The size of the panel and the question being asked determine the estimator:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Large panel, 50--200 indicators | [DFM Nowcasting](@ref nowcast_dfm_page) | Factors compress the cross-section |
| Medium panel with shrinkage priors | [BVAR Nowcasting](@ref nowcast_bvar_page) | Marginal likelihood tunes the shrinkage |
| Fast, transparent baseline | [Bridge Equations](@ref nowcast_bridge_page) | Plain OLS on aggregated indicators |
| One indicator, within-period timing | [MIDAS Regression](@ref midas_page) | Weight function keeps every high-frequency lag |
| Attribute a revision to releases | [News Decomposition](@ref nowcast_news_page) | Kalman gain weights each new observation |

### Method Comparison

| Criterion | DFM | BVAR | Bridge |
|-----------|-----|------|--------|
| **Cross-section size** | Large (50--200) | Medium (10--50) | Small (5--20) |
| **Interpretability** | Latent factors | Direct coefficients | Simple OLS |
| **News decomposition** | Native | --- | --- |
| **Computational cost** | Moderate (EM) | Moderate (optimization) | Fast (OLS) |
| **Best for** | Large mixed-frequency panels | Medium panels with priors | Quick baseline |

MIDAS sits outside this comparison: it relates one high-frequency indicator to the target through a parametric weight function rather than pooling a panel.

---

## Child Pages

- [DFM Nowcasting](@ref nowcast_dfm_page) --- EM algorithm, Kalman smoother, Mariano-Murasawa temporal aggregation, block structure, idiosyncratic dynamics
- [BVAR Nowcasting](@ref nowcast_bvar_page) --- GLP prior, dummy observations, hyperparameter optimization, Kalman smoothing
- [Bridge Equations](@ref nowcast_bridge_page) --- quarterly aggregation, pairwise OLS, median combination, interpolation
- [News Decomposition](@ref nowcast_news_page) --- revision attribution, per-release impact, group aggregation, data vintage comparison
- [MIDAS Regression](@ref midas_page) --- exponential-Almon, Beta, and polynomial weights, ADL-MIDAS, U-MIDAS, direct forecasting

---

## The Nowcasting Problem

Quarterly aggregates like GDP are released with a 4--8 week delay, while dozens of monthly indicators are available in real time. Nowcasting produces current-quarter estimates by exploiting this timely high-frequency information:

```math
\underbrace{Y_t}_{\text{target (quarterly)}} = f\big(\underbrace{X_{1,t}, \ldots, X_{N,t}}_{\text{monthly indicators}}\big) + \varepsilon_t
```

where:
- ``Y_t`` is the quarterly target variable (e.g., GDP growth)
- ``X_{j,t}`` are monthly indicator variables
- ``\varepsilon_t`` is the forecast error

Three challenges define the problem:

1. **Mixed frequencies** --- monthly indicators and quarterly targets coexist in the same model
2. **Ragged edges** --- not all series update simultaneously; the most recent months have missing observations for slower-release variables
3. **Large cross-sections** --- dozens to hundreds of indicators provide complementary information

!!! note "Data Layout Convention"
    All nowcasting functions expect a ``T \times N`` matrix where the first `nM` columns are monthly variables and the last `nQ` columns are quarterly variables. Quarterly observations appear every 3rd row (months 3, 6, 9, 12) with `NaN` for non-quarter-end months. The ragged edge is represented by trailing `NaN` values in the most recent rows.

Scale matters as much as layout. The DFM standardizes internally, but series with vastly different scales slow EM convergence. Apply `apply_tcode()` to FRED-MD data first to obtain stationary, comparable-scale series.

---

## Nowcast Extraction

The `nowcast()` function extracts the current-quarter estimate and a one-quarter-ahead forecast from any `AbstractNowcastModel`:

```@example nc
result = nowcast(dfm)
result.nowcast    # current-quarter value
result.forecast   # next-quarter forecast
result.method     # :dfm, :bvar, or :bridge
```

Each method computes the forecast differently: DFM projects the state vector 3 months forward, BVAR iterates the VAR one step, and Bridge uses the median of individual equation nowcasts. Multi-step forecasts are available via `forecast(model, h)` for DFM and BVAR models.

---

## StatsAPI Interface

| Function | DFM | BVAR | Bridge |
|----------|-----|------|--------|
| `loglikelihood(m)` | Log-likelihood at convergence | Marginal log-likelihood | --- |
| `predict(m)` | Smoothed data `X_sm` | Smoothed data `X_sm` | Smoothed data `X_sm` |
| `nobs(m)` | Number of time periods | Number of time periods | Number of time periods |

---

## Visualization

The `plot_result` function supports multiple views for a `NowcastResult`, following the visualization patterns of the ECB Nowcasting Toolbox (Linzenich and Meunier 2024). The default view leads with the target panel --- the smoothed quarterly series extended by the nowcast and the one-quarter-ahead forecast --- then adds one panel per monthly indicator and, for DFM models, one panel per extracted factor.

```julia
plot_result(result)                      # default: target + factors (DFM)
plot_result(result; view=:heatmap,       # z-score heatmap with ragged edge
            variable_names=["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"])
plot_result(result; view=:contributions) # group contributions (DFM only)
```

```@raw html
<iframe src="../assets/plots/nowcast_result.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **heatmap view** computes z-scores for each variable and renders the last `n_periods` (default 18) as a color-coded matrix, labelling the rows with `variable_names`. Missing values appear in grey, revealing the ragged edge --- here the two months in every three for which the quarterly target is unobserved, plus the unpublished current quarter.

```@raw html
<iframe src="../assets/plots/nowcast_heatmap.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **contributions view** draws three stacked bars --- the target's sample mean, the factor contribution to the current-quarter nowcast, and the factor contribution to the one-quarter-ahead forecast --- with one stack segment per factor block. A single-block DFM like this one yields one segment; pass `groups` and `group_names` to split the factors into named blocks.

```@raw html
<iframe src="../assets/plots/nowcast_contributions.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

`NowcastNews` objects carry their own three views (`:releases`, `:groups`, `:individual`); see [News Decomposition](@ref nowcast_news_page).

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example nc
path = joinpath(mktempdir(), "nowcast_dfm.jld2")
save_model(dfm, path)
dfm2 = load_model(path)
typeof(dfm2)
```

---

## References

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [DOI: 10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Banbura, Marta, Irina Belousova, Katalin Bodnar, and Mate Barnabas Toth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [DOI: 10.1162/REST\_a\_00483](https://doi.org/10.1162/REST_a_00483)
- Linzenich, Jan, and Baptiste Meunier. 2024. "Nowcasting with Mixed Frequency Data Using a Simple Modelling Setup: An Update of the ECB Nowcasting Framework." *ECB Working Paper* No. 3004.
- Mariano, Roberto S., and Yasutomo Murasawa. 2003. "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18 (4): 427--443. [DOI: 10.1002/jae.695](https://doi.org/10.1002/jae.695)
