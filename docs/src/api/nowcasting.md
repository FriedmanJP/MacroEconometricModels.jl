# [Nowcasting API](@id api_nowcasting)

Mixed-frequency nowcasting via dynamic factor models, large Bayesian VARs, and bridge equations, plus the news decomposition that attributes a nowcast revision to individual data releases. See [Nowcasting](@ref nowcast_page) for theory and examples, and its children [DFM](@ref nowcast_dfm_page), [BVAR](@ref nowcast_bvar_page), [Bridge](@ref nowcast_bridge_page), and [News](@ref nowcast_news_page).

---

## Nowcasting Types

```@docs
AbstractNowcastModel
NowcastDFM
NowcastBVAR
NowcastBridge
NowcastResult
NowcastNews
NowcastForecast
```

---

## Estimation

Three nowcasting models over the same ragged-edge panel: a dynamic factor model estimated
by EM plus the Kalman smoother, a large Bayesian VAR with Giannone-Lenza-Primiceri priors,
and a set of bridge equations combined by OLS.

```@docs
nowcast_dfm
nowcast_bvar
nowcast_bridge
```

---

## Nowcast and Forecast

`nowcast` extracts the current-quarter nowcast and the next-quarter forecast from a fitted
model; `forecast` extends the horizon further.

```@docs
nowcast
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["nowcast/forecast.jl"]
Order   = [:function]
```

---

## News Decomposition

Attributes the revision between two data vintages to the individual releases that caused
it. See [Nowcast News](@ref nowcast_news_page) for the release- and group-level views.

```@docs
nowcast_news
```

---

## Panel Balancing

Fills the missing entries of a ragged panel by DFM imputation, returning a container of
the same type.

```@docs
balance_panel
```

---

## Nowcast Display

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["summary_nowcast.jl"]
Order   = [:function]
```
