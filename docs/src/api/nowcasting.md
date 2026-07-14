# [Nowcasting API](@id api_nowcasting)

Mixed-frequency nowcasting via DFM, BVAR, and bridge equations with news decomposition. See [Nowcasting](../nowcast.md) for theory and examples.

---

## Nowcasting Types

```@docs
AbstractNowcastModel
NowcastDFM
NowcastBVAR
NowcastBridge
NowcastResult
NowcastNews
```

---

## Estimation

```@docs
nowcast_dfm
nowcast_bvar
nowcast_bridge
```

---

## Nowcast and Forecast

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

```@docs
nowcast_news
```

---

## Panel Balancing

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
