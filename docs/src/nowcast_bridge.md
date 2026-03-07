# [Bridge Equations](@id nowcast_bridge_page)

Bridge equations provide a simple, transparent approach to nowcasting by regressing a quarterly target on aggregated monthly indicators via OLS. Multiple equations --- one per pair of monthly indicators, plus univariate equations --- are combined through the **median**, producing a robust current-quarter estimate. For the broader nowcasting framework, see [Nowcasting](@ref).

```@setup nc_bridge
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

```@example nc_bridge
# Standard mixed-frequency data setup (used throughout this page)
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
nothing # hide
```

**Recipe 1: Basic bridge equation nowcast**

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ)
report(bridge)
```

**Recipe 2: Bridge with custom lags**

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ; lagM=2, lagQ=1, lagY=2)
report(bridge)
```

**Recipe 3: Individual equation results**

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ)
n_quarters = length(bridge.Y_nowcast)
for eq in 1:bridge.n_equations
    val = bridge.Y_individual[n_quarters, eq]
    isnan(val) || println("Equation $eq: ", round(val, digits=4))
end
println("Combined (median): ", round(bridge.Y_nowcast[n_quarters], digits=4))
```

**Recipe 4: Bridge with TimeSeriesData**

```@example nc_bridge
ts = TimeSeriesData(Y; varnames=["INDPRO", "UNRATE", "CPI", "M2", "FEDFUNDS"], frequency=Monthly)
bridge = nowcast_bridge(ts, nM, nQ)
result = nowcast(bridge)
report(result)
```

---

## Model Specification

Bridge equations translate monthly indicator information into a quarterly forecast through a two-step procedure: aggregate monthly data to quarterly frequency, then regress the quarterly target on these aggregated regressors using OLS. The method constructs multiple regression equations and combines them via the median (Banbura et al. 2023).

For each pair ``(m_1, m_2)`` of monthly indicators (or a single indicator when ``m_1 = m_2``):

```math
Y_t^Q = \beta_0 + \sum_{l=0}^{L_M} \beta_{m_1,l} \, X_{m_1,t-l}^Q + \sum_{l=0}^{L_M} \beta_{m_2,l} \, X_{m_2,t-l}^Q + \sum_{l=1}^{L_Q} \gamma_l \, X_{t-l}^Q + \sum_{l=1}^{L_Y} \delta_l \, Y_{t-l}^Q + \varepsilon_t
```

where:
- ``Y_t^Q`` is the quarterly target variable (last column of the data)
- ``X_{m,t-l}^Q`` are monthly indicators aggregated to quarterly frequency at lag ``l``
- ``X_{t-l}^Q`` are quarterly covariates (columns between monthly and target)
- ``L_M``, ``L_Q``, ``L_Y`` are the lag orders for monthly, quarterly, and autoregressive terms
- ``\beta_0`` is the intercept

**Quarterly aggregation.** Monthly data is converted to quarterly frequency via 3-month averages: for quarter ``q``, the aggregated value is the arithmetic mean of months ``3(q-1)+1`` through ``3q``. Missing monthly values are filled by linear interpolation before aggregation.

**Combination.** The model constructs ``\binom{n_M}{2} + n_M`` bridge equations (all pairwise combinations plus univariate equations). The final nowcast is the **median** across all individual equation predictions, providing robustness to individual equation failures or outlying forecasts.

---

## Estimation

Each bridge equation is estimated by OLS with regularization. The procedure is fully automatic: construct regressors, solve the normal equations, and combine predictions via median.

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)
report(bridge)
println("Bridge equations: ", bridge.n_equations)  # C(4,2) + 4 = 10
```

The median combination provides robustness because individual equations may overfit or suffer from collinearity. When two monthly indicators are highly correlated, their pairwise equation may produce an extreme forecast, but the median is unaffected.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lagM` | `Int` | `1` | Monthly indicator lags after quarterly aggregation |
| `lagQ` | `Int` | `1` | Quarterly indicator lags |
| `lagY` | `Int` | `1` | Autoregressive lags for target |

### `NowcastBridge` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data with NaN filled by interpolation |
| `Y_nowcast` | `Vector{T}` | Combined nowcast per quarter (median across equations) |
| `Y_individual` | `Matrix{T}` | Individual equation nowcasts (``n_{\text{quarters}} \times n_{\text{equations}}``) |
| `n_equations` | `Int` | Number of bridge equations |
| `coefficients` | `Vector{Vector{T}}` | OLS coefficients per equation |
| `nM` | `Int` | Number of monthly variables |
| `nQ` | `Int` | Number of quarterly variables |
| `lagM` | `Int` | Monthly indicator lags used |
| `lagQ` | `Int` | Quarterly indicator lags used |
| `lagY` | `Int` | Autoregressive lags used |
| `data` | `Matrix{T}` | Original data with NaN |

---

## Complete Example

```@example nc_bridge
# === Step 1: Estimate bridge models with different lag structures ===
bridge_1 = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)
bridge_2 = nowcast_bridge(Y, nM, nQ; lagM=2, lagQ=1, lagY=2)

# === Step 2: Compare nowcasts ===
r1 = nowcast(bridge_1)
r2 = nowcast(bridge_2)
println("Bridge (lagM=1, lagY=1): ", round(r1.nowcast, digits=4))
println("Bridge (lagM=2, lagY=2): ", round(r2.nowcast, digits=4))

# === Step 3: Inspect individual equations ===
report(bridge_1)
n_quarters = length(bridge_1.Y_nowcast)
valid_preds = filter(!isnan, bridge_1.Y_individual[n_quarters, :])
println("Spread: min=", round(minimum(valid_preds), digits=4),
        " median=", round(bridge_1.Y_nowcast[n_quarters], digits=4),
        " max=", round(maximum(valid_preds), digits=4))

# === Step 4: Compare with DFM ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
r_dfm = nowcast(dfm)
println("Bridge: ", round(r1.nowcast, digits=4), "  DFM: ", round(r_dfm.nowcast, digits=4))
```

**Interpretation.** The bridge approach constructs 10 equations from the 4 monthly FRED-MD indicators (6 pairwise + 4 univariate). Each equation aggregates monthly indicators to quarterly frequency via 3-month averages, then runs OLS on the quarterly target. The spread between minimum and maximum individual equation nowcasts reveals disagreement across indicator combinations --- a wide spread suggests the choice of indicators matters, while a narrow spread indicates consensus.

---

## Common Pitfalls

1. **Too few monthly indicators.** With ``n_M = 2``, the model constructs only 3 equations (1 pair + 2 univariate), which is too few for the median to provide meaningful robustness. Use at least 4 monthly indicators.

2. **Excessive lags with short samples.** Each additional lag consumes one quarterly observation. With 100 monthly observations (33 quarters), setting `lagM=3, lagQ=3, lagY=3` leaves only about 30 observations per equation. Keep total lag count below one-third of the available quarters.

3. **Non-stationary data.** Bridge equations assume stationary relationships. Apply appropriate transformations (differencing, log-differencing) before estimation. The `apply_tcode()` function handles standard FRED-MD transformation codes.

4. **Confusing quarterly indexing.** The `Y_nowcast` vector has length ``\lfloor T/3 \rfloor``, not ``T``. Quarter ``q`` corresponds to months ``3(q-1)+1`` through ``3q``.

---

## References

- Banbura, Marta, Irina Belousova, Katalin Bodnar, and Mate Barnabas Toth. 2023.
  "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.

