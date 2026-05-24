# [Nowcasting](@id nowcast_page)

Central banks and forecasters need current-quarter GDP estimates weeks before official release. Nowcasting bridges this gap by extracting signal from timely high-frequency indicators --- monthly industrial production, employment, and financial data --- to produce real-time estimates of quarterly aggregates. **MacroEconometricModels.jl** implements three complementary approaches:

- **Dynamic Factor Model (DFM)**: EM + Kalman smoother on latent factors with Mariano-Murasawa temporal aggregation; native news decomposition; see [DFM Nowcasting](@ref nowcast_dfm_page)
- **Large Bayesian VAR**: GLP-style Normal-Inverse-Wishart prior with hyperparameter optimization via marginal likelihood; see [BVAR Nowcasting](@ref nowcast_bvar_page)
- **Bridge Equations**: OLS regressions on quarterly-aggregated monthly indicators, combined via median; see [Bridge Equations](@ref nowcast_bridge_page)
- **News Decomposition**: attribute nowcast revisions to individual data releases (Banbura and Modugno 2014); see [News Decomposition](@ref nowcast_news_page)

```@setup nc
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

All recipes use the following FRED-MD mixed-frequency setup:

```@example nc
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
Y[end, end] = NaN       # simulate ragged edge
nothing # hide
```

**Recipe 1: DFM nowcasting**

```@example nc
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
report(dfm)
```

**Recipe 2: BVAR nowcasting**

```@example nc
bvar = nowcast_bvar(Y, nM, nQ; lags=5)
report(bvar)
```

**Recipe 3: Bridge equation nowcasting**

```@example nc
bridge = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1)
report(bridge)
```

**Recipe 4: Nowcast extraction and comparison**

```@example nc
r_dfm = nowcast(dfm)
r_bvar = nowcast(bvar)
r_bridge = nowcast(bridge)
report(r_dfm)
```

**Recipe 5: News decomposition**

```@example nc
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN   # simulate 3 releases arriving

news = nowcast_news(X_new, X_old, dfm, size(Y, 1); target_var=size(Y, 2))
report(news)
```

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

---

## Method Comparison

| Criterion | DFM | BVAR | Bridge |
|-----------|-----|------|--------|
| **Cross-section size** | Large (50--200) | Medium (10--50) | Small (5--20) |
| **Interpretability** | Latent factors | Direct coefficients | Simple OLS |
| **News decomposition** | Native | --- | --- |
| **Computational cost** | Moderate (EM) | Moderate (optimization) | Fast (OLS) |
| **Best for** | Large mixed-frequency panels | Medium panels with priors | Quick baseline |

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

The `plot_result` function supports multiple views for both `NowcastResult` and `NowcastNews` objects, following the visualization patterns of the ECB Nowcasting Toolbox (Linzenich and Meunier 2024).

### Nowcast Result Views

The default view displays the smoothed target variable with nowcast/forecast extension. For DFM models, extracted factor panels are appended automatically.

```julia
plot_result(nr)                          # default: target + factors (DFM)
plot_result(nr; view=:heatmap)           # z-score heatmap with ragged edge
plot_result(nr; view=:contributions)     # group contributions (DFM only)
```

```@raw html
<iframe src="../assets/plots/nowcast_result.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **heatmap view** computes z-scores for each variable and renders the last `n_periods` (default 18) as a color-coded matrix. Missing values appear in grey, revealing the ragged edge --- the characteristic pattern of staggered data releases.

```@raw html
<iframe src="../assets/plots/nowcast_heatmap.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **contributions view** decomposes the nowcast into factor/block group contributions, showing how each group drives the current-quarter estimate and the one-quarter-ahead forecast.

```@raw html
<iframe src="../assets/plots/nowcast_contributions.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### News Decomposition Views

```julia
plot_result(news)                        # default: per-release impact bar chart
plot_result(news; view=:groups)          # stacked bar by variable group
plot_result(news; view=:individual)      # sorted bar by absolute impact
```

```@raw html
<iframe src="../assets/plots/nowcast_news.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **group view** aggregates news impacts by variable group, immediately revealing which sector drove the revision.

```@raw html
<iframe src="../assets/plots/nowcast_news_groups.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The **individual view** sorts releases by absolute impact, identifying the single most influential data points.

```@raw html
<iframe src="../assets/plots/nowcast_news_individual.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Sub-Page Guide

For detailed treatment of each method --- theory, model specification, keyword tables, return value tables, and advanced usage:

- [DFM Nowcasting](@ref nowcast_dfm_page) --- EM algorithm, Kalman smoother, Mariano-Murasawa temporal aggregation, block structure, idiosyncratic dynamics
- [BVAR Nowcasting](@ref nowcast_bvar_page) --- GLP prior, dummy observations, hyperparameter optimization, Kalman smoothing
- [Bridge Equations](@ref nowcast_bridge_page) --- quarterly aggregation, pairwise OLS, median combination, interpolation
- [News Decomposition](@ref nowcast_news_page) --- revision attribution, per-release impact, group aggregation, data vintage comparison

---

## Complete Example

This workflow produces a real-time GDP nowcast from mixed-frequency FRED-MD data, compares DFM and BVAR estimates, and attributes the revision to individual data releases via news decomposition.

```@example nc
# 1. DFM estimation with 2 factors and AR(1) idiosyncratic dynamics
dfm_full = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=200)
report(dfm_full)
```

```@example nc
# 2. BVAR estimation with GLP prior
bvar_full = nowcast_bvar(Y, nM, nQ; lags=4, max_iter=200)
report(bvar_full)
```

```@example nc
# 3. Compare point estimates
r_dfm = nowcast(dfm_full)
r_bvar = nowcast(bvar_full)
(dfm=round(r_dfm.nowcast, digits=4), bvar=round(r_bvar.nowcast, digits=4))
```

```@example nc
# 4. News decomposition: simulate a data vintage update
X_old = copy(Y)
X_old[end-2, 1:nM] .= NaN   # mask the most recent monthly releases
X_new = copy(Y)
news = nowcast_news(X_new, X_old, dfm_full, size(Y, 1);
                    groups=[1, 2, 1, 2, 3],
                    group_names=["Real", "Nominal", "Target"])
report(news)
```

```julia
plot_result(dfm_full)
plot_result(news)
```

The DFM and BVAR nowcasts provide independent estimates of the target quarter. Agreement between the two methods increases confidence in the point forecast. The news decomposition traces the nowcast revision to specific data releases, identifying which monthly indicator contributed most to the update. The group-level attribution reveals whether the revision is driven by real activity, nominal prices, or financial conditions.

---

## Common Pitfalls

1. **NaN placement determines quarterly alignment.** Quarterly values must appear at rows divisible by 3 (months 3, 6, 9, 12). Placing quarterly observations at other rows produces silently incorrect temporal aggregation weights.

2. **Column ordering matters.** Monthly variables occupy columns `1:nM` and quarterly variables occupy columns `nM+1:nM+nQ`. Swapping this order causes the Mariano-Murasawa weights to apply to the wrong variables.

3. **Standardize before estimation.** The DFM internally standardizes data, but raw data with vastly different scales can slow EM convergence. Apply `apply_tcode()` to FRED-MD data to obtain stationary, comparable-scale series.

4. **News decomposition requires two vintages.** The `nowcast_news` function compares two data matrices (`X_new` and `X_old`) that differ only in newly released observations. Both matrices must have the same dimensions; the old vintage has `NaN` where the new vintage has observed values.

5. **Bridge equations need sufficient quarterly observations.** With `lagM`, `lagQ`, and `lagY` lags, the effective sample shrinks. Ensure at least 20 quarterly observations remain after lag truncation to avoid overfitting.

---

## References

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [DOI: 10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Banbura, Marta, Irina Belousova, Katalin Bodnar, and Mate Barnabas Toth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [DOI: 10.1162/REST\_a\_00483](https://doi.org/10.1162/REST_a_00483)
- Linzenich, Jan, and Baptiste Meunier. 2024. "Nowcasting with Mixed Frequency Data Using a Simple Modelling Setup: An Update of the ECB Nowcasting Framework." *ECB Working Paper* No. 3004.
- Mariano, Roberto S., and Yasutomo Murasawa. 2003. "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18 (4): 427--443. [DOI: 10.1002/jae.695](https://doi.org/10.1002/jae.695)
