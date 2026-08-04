# [Bridge Equations](@id nowcast_bridge_page)

Bridge equations are the transparent baseline of nowcasting: aggregate the monthly indicators to quarterly frequency, regress the quarterly target on them by OLS, and read off the fitted value for the current quarter. The fragility of any single such regression is handled by estimating many of them — one per pair of monthly indicators plus one per indicator alone — and combining their predictions with the **median**, which is unmoved by an equation that overfits or collapses under collinearity (Bańbura et al. 2023). No latent states, no priors, no iteration.

For the shared data layout, the `nowcast()` interface, and the result visualizations, see [Nowcasting](@ref nowcast_page). Sibling estimators: [DFM Nowcasting](@ref nowcast_dfm_page) and [BVAR Nowcasting](@ref nowcast_bvar_page).

```@setup nc_bridge
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
nM, nQ = 4, 1
N = nM + nQ
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN
T_obs = size(Y, 1)
```

The examples run on a FRED-MD panel in the standard mixed-frequency layout: four monthly indicators (`INDPRO`, `UNRATE`, `CPIAUCSL`, `M2SL`) in the leading columns and one quarterly target (`FEDFUNDS`) in the last, observed only at quarter-end months and missing in the most recent month.

## Quick Start

**Recipe 1: Bridge nowcast with default lags**

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ)
report(bridge)
```

**Recipe 2: Richer lag structure**

```@example nc_bridge
bridge_2 = nowcast_bridge(Y, nM, nQ; lagM=2, lagQ=1, lagY=2)
report(bridge_2)
```

**Recipe 3: The individual equations behind the median**

```@example nc_bridge
q = length(bridge.Y_nowcast)
preds = bridge.Y_individual[q, :]
(equations=bridge.n_equations, min=round(minimum(preds), digits=4),
 median=round(bridge.Y_nowcast[q], digits=4), max=round(maximum(preds), digits=4))
```

**Recipe 4: Read off the nowcast**

```@example nc_bridge
report(nowcast(bridge))
```

**Recipe 5: `TimeSeriesData` dispatch**

```@example nc_bridge
ts = TimeSeriesData(Y; varnames=["INDPRO", "UNRATE", "CPI", "M2", "FEDFUNDS"],
                    frequency=Monthly)
report(nowcast(nowcast_bridge(ts, nM, nQ)))
```

---

## Model Specification

A bridge equation is an ordinary quarterly regression whose regressors happen to be built from monthly data. For a pair ``(m_1, m_2)`` of monthly indicators — or a single indicator when ``m_1 = m_2`` — the equation is

```math
Y_q = \beta_0 + \sum_{l=0}^{L_M} \beta_{m_1,l} \, X_{m_1,q-l} + \sum_{l=0}^{L_M} \beta_{m_2,l} \, X_{m_2,q-l}
      + \sum_{l=1}^{L_Q} \sum_{k} \gamma_{k,l} \, Z_{k,q-l} + \sum_{l=1}^{L_Y} \delta_l \, Y_{q-l} + \varepsilon_q
```

where:
- ``Y_q`` is the quarterly target, the last column of the data
- ``X_{m,q-l}`` is monthly indicator ``m`` aggregated to quarter ``q-l``
- ``Z_{k,q-l}`` are the non-target quarterly variables, the columns between the monthly block and the target
- ``L_M``, ``L_Q``, ``L_Y`` are `lagM`, `lagQ` and `lagY`
- ``\beta_0`` is the intercept

The monthly terms start at lag ``0``: the current quarter's own indicator values are the whole point of the exercise. Quarterly and autoregressive terms start at lag ``1``, since the contemporaneous target is what is being predicted. In a univariate equation the two indicator sums collapse to one — the duplicate regressor is dropped rather than entered twice, which would make the design rank-deficient.

**Quarterly aggregation.** Monthly series are averaged three months at a time: quarter ``q`` takes the mean of months ``3(q-1)+1`` through ``3q``, or of however many of those months exist for a partial final quarter. Non-target quarterly columns are read at the quarter-end month rather than averaged. Missing monthly values are filled beforehand by linear interpolation between the neighbouring observations, with forward fill after the last observation and backward fill before the first.

**Combination.** The model builds ``\binom{n_M}{2} + n_M`` equations — every unordered pair plus every singleton — and reports the median of their predictions. With four monthly indicators that is ``6 + 4 = 10`` equations. The median is what makes the procedure robust: two highly correlated indicators entering the same equation can produce a wild coefficient pair and an extreme forecast, and the median simply steps over it.

!!! note "Quarter indexing"
    The number of quarters is ``\lceil T_{\text{obs}} / 3 \rceil``, using ceiling division so that the current, partially observed quarter — the very quarter a nowcast exists to produce — gets a row. A 60-month panel therefore yields 20 quarters, and `Y_nowcast[20]` is the current-quarter estimate.

---

## Estimation

Each equation is estimated by OLS on the quarters where the target is observed, with a ``10^{-6}`` ridge on the normal equations to survive near-singular designs and a `robust_inv` fallback if that still fails. Quarters before ``\max(L_M, L_Q, L_Y) + 1`` cannot form a complete lag vector and are skipped; an equation with fewer than three usable quarters, or with fewer observations than coefficients, is abandoned and contributes `NaN` to every quarter rather than a spurious fit.

```@example nc_bridge
bridge = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)
(equations=bridge.n_equations, quarters=length(bridge.Y_nowcast),
 coef_counts=unique(length.(bridge.coefficients)),
 skipped_quarters=count(isnan, bridge.Y_nowcast))
```

Ten equations are fitted over 20 quarters. The pairwise equations carry six coefficients — an intercept, two indicators at lags 0 and 1, and one autoregressive term — and the univariate equations four, because the collapsed duplicate removes two columns. One quarter is `NaN`: the first, which has no lagged quarter to draw on. There are no quarterly covariates here, since `nQ = 1` means the only quarterly column is the target itself.

Predictions are then formed for **every** quarter from the same coefficients, including quarters where the target is missing. That is what produces the nowcast: the final quarter's regressors are complete because the monthly indicators are observed, even though the target is not.

---

## Reading the Individual Equations

`Y_individual` keeps every equation's prediction, so the dispersion across equations is available as a diagnostic that the median alone hides.

```@example nc_bridge
q = length(bridge.Y_nowcast)
preds = bridge.Y_individual[q, :]
(n=count(!isnan, preds), min=round(minimum(preds), digits=4),
 median=round(bridge.Y_nowcast[q], digits=4), max=round(maximum(preds), digits=4))
```

All ten equations produce a current-quarter number, spanning 0.0082 to 0.0525 around a median of 0.0301. A spread that wide relative to the median means the answer depends materially on which indicators are used, and the combined nowcast should be treated as one draw from a genuinely uncertain range rather than a point estimate. A tight cluster would say the opposite: the indicators agree and the choice among them is immaterial.

Lag structure moves the answer as much as indicator choice does:

```@example nc_bridge
bridge_2 = nowcast_bridge(Y, nM, nQ; lagM=2, lagQ=1, lagY=2)
(lags_1=round(nowcast(bridge).nowcast, digits=4),
 lags_2=round(nowcast(bridge_2).nowcast, digits=4))
```

Going from one monthly and one autoregressive lag to two of each moves the nowcast from 0.0301 to 0.0055. With 20 quarters and up to eight regressors, the richer specification spends a quarter of its degrees of freedom on lags that a 60-month sample cannot pin down. Prefer the parsimonious setting unless the target is known to respond to indicators with a delay.

---

## Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lagM` | `Int` | `1` | Monthly indicator lags after quarterly aggregation; lag 0 is always included |
| `lagQ` | `Int` | `1` | Lags of the non-target quarterly variables |
| `lagY` | `Int` | `1` | Autoregressive lags of the target |

---

## NowcastBridge Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Monthly panel with missing values filled by interpolation |
| `Y_nowcast` | `Vector{T}` | Median across equations per quarter, length ``\lceil T_{\text{obs}}/3 \rceil`` |
| `Y_individual` | `Matrix{T}` | Per-equation predictions, ``n_{\text{quarters}} \times n_{\text{equations}}`` |
| `n_equations` | `Int` | Number of bridge equations, ``\binom{n_M}{2} + n_M`` |
| `coefficients` | `Vector{Vector{T}}` | OLS coefficients per equation; empty for equations that could not be fitted |
| `nM` | `Int` | Number of monthly variables |
| `nQ` | `Int` | Number of quarterly variables |
| `lagM` | `Int` | Monthly lags used |
| `lagQ` | `Int` | Quarterly lags used |
| `lagY` | `Int` | Autoregressive lags used |
| `data` | `Matrix{T}` | Original input panel, NaN included |

`predict(bridge)` returns `X_sm` and `nobs(bridge)` the number of months. `loglikelihood` is not defined — bridge equations are estimated by OLS, not by maximum likelihood over a joint model.

!!! warning "`forecast` is not defined for bridge models"
    There is no `forecast(::NowcastBridge, h)` method: a bridge equation needs quarterly aggregates of monthly indicators as inputs, which do not exist beyond the current quarter. `nowcast(bridge)` returns a `NowcastResult` whose `forecast` field repeats the nowcast for that reason.

---

## Complete Example

```@example nc_bridge
# === Step 1: Estimate under two lag structures ===
bridge_1 = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)
bridge_2 = nowcast_bridge(Y, nM, nQ; lagM=2, lagQ=1, lagY=2)
report(bridge_1)
```

```@example nc_bridge
# === Step 2: Compare the combined nowcasts ===
(lags_1=round(nowcast(bridge_1).nowcast, digits=4),
 lags_2=round(nowcast(bridge_2).nowcast, digits=4))
```

```@example nc_bridge
# === Step 3: Dispersion across equations in the current quarter ===
q = length(bridge_1.Y_nowcast)
preds = bridge_1.Y_individual[q, :]
(min=round(minimum(preds), digits=4), median=round(bridge_1.Y_nowcast[q], digits=4),
 max=round(maximum(preds), digits=4))
```

```@example nc_bridge
# === Step 4: Cross-check against the DFM on the same panel ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
(bridge=round(nowcast(bridge_1).nowcast, digits=4),
 dfm=round(nowcast(dfm).nowcast, digits=4))
```

**Interpretation.** Ten equations built from the four monthly FRED-MD indicators produce a current-quarter median of 0.0301, with individual predictions between 0.0082 and 0.0525. The DFM gives 0.0375 on the same panel — inside the bridge equations' own spread, so the two methods agree to within the disagreement the bridge combination already reports. That is the useful reading of a bridge nowcast: the median is the estimate and the spread is its credibility. The lag comparison is the warning attached to it, since doubling the monthly and autoregressive lags moves the median to 0.0055, further than any of the indicator combinations moved it.

---

## Common Pitfalls

1. **Too few monthly indicators leaves nothing to combine.** With ``n_M = 2`` there are only ``1 + 2 = 3`` equations and the median is the middle of three highly overlapping fits. Four indicators (ten equations) is a practical minimum for the combination to add robustness.

2. **Lags are expensive at quarterly frequency.** A 60-month panel is 20 quarters, and each lag both consumes an initial quarter and adds regressors to every equation. Keep the total lag count below a third of the available quarters; the default `lagM=lagQ=lagY=1` is the right starting point for a sample this size.

3. **Bridge equations assume stationarity.** OLS on trending levels produces a spurious fit that the median will happily propagate. Transform first — `apply_tcode()` applies the FRED-MD transformation codes.

4. **`Y_nowcast` is indexed by quarter, not by month.** Its length is ``\lceil T_{\text{obs}}/3 \rceil``, and quarter ``q`` covers months ``3(q-1)+1`` through ``3q``. Indexing it with a month number silently reads the wrong quarter.

5. **The leading quarters are `NaN` by construction.** Quarters before ``\max(L_M, L_Q, L_Y) + 1`` have no complete lag vector and are left missing. Filter with `!isnan` before taking statistics over `Y_nowcast`, and do not read a leading `NaN` as an estimation failure.

6. **Interpolation runs before aggregation, not after.** Gaps in the monthly indicators are filled by straight-line interpolation, and the edges are carried flat. A long gap at the end of an indicator therefore enters the quarterly average as a repeated last value, which damps that indicator's contribution rather than dropping it.

---

## References

- Bańbura, Marta, Irina Belousova, Katalin Bodnár, and Máté Barnabás Tóth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.
- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2011. "Nowcasting." In *The Oxford Handbook of Economic Forecasting*, 193--224. Oxford: Oxford University Press. [https://doi.org/10.1093/oxfordhb/9780195398649.013.0008](https://doi.org/10.1093/oxfordhb/9780195398649.013.0008)
- Giannone, Domenico, Lucrezia Reichlin, and David Small. 2008. "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55 (4): 665--676. [https://doi.org/10.1016/j.jmoneco.2008.05.010](https://doi.org/10.1016/j.jmoneco.2008.05.010)
