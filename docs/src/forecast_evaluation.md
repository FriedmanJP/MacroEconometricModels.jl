# Forecast Evaluation & Combination

Every model in MacroEconometricModels.jl exposes a `forecast`, but comparing those forecasts — the daily work of a forecasting desk — needs a separate toolkit. This page documents `src/fceval`, a model-agnostic suite that scores point forecasts, tests whether one forecast beats another, and combines competing forecasts into a single series. It consumes any model's output through a light duck-typed interface: a vector of realized values plus a vector or matrix of forecasts. No forecast type is touched.

The suite provides:

- **Accuracy metrics** — ME, MAE, RMSE, MAPE, sMAPE, MASE, Theil's `U1`/`U2`, and the Theil MSE bias/variance/covariance decomposition ([`forecast_evaluate`](@ref)).
- **Equal-accuracy test** — Diebold–Mariano (1995) with the Harvey–Leybourne–Newbold (1997) small-sample correction ([`diebold_mariano`](@ref)).
- **Nested-model test** — Clark–West (2007), the correct test when one model nests the other ([`clark_west`](@ref)).
- **Efficiency & encompassing** — Mincer–Zarnowitz (1969) efficiency regression ([`mincer_zarnowitz`](@ref)) and the Harvey–Leybourne–Newbold (1998) encompassing test ([`forecast_encompassing`](@ref)).
- **Combination** — equal, Bates–Granger inverse-MSE, and Granger–Ramanathan constrained least squares ([`combine_forecasts`](@ref)).

## Quick Start

```@setup fceval
using MacroEconometricModels
t = collect(1.0:120.0)
# Two competing forecast-error series and a realized series
e1 = sin.(0.3 .* t) .+ 0.5 .* cos.(0.1 .* t)
e2 = 0.8 .* sin.(0.3 .* t .+ 0.5) .+ 0.2
actual = 2.0 .+ 0.5 .* sin.(0.2 .* t) .+ cos.(0.15 .* t)
f1 = actual .+ e1        # two forecasts of `actual`; errors are `actual .- fⱼ`
f2 = actual .+ e2
```

**Recipe 1: Score competing forecasts**

```@example fceval
ev = forecast_evaluate(actual, hcat(f1, f2); model_names=["Model A", "Model B"])
report(ev)
```

**Recipe 2: Test equal predictive accuracy (Diebold–Mariano)**

```@example fceval
dm = diebold_mariano(actual .- f1, actual .- f2; h=1, loss=:se)
report(dm)
```

**Recipe 3: Combine forecasts**

```@example fceval
comb = combine_forecasts(hcat(f1, f2), actual; method=:bates_granger)
report(comb)
```

---

## Point Accuracy Metrics

[`forecast_evaluate`](@ref) reports the standard scale-dependent and scale-free accuracy measures for one forecast (an `AbstractVector`) or several (a `T×M` `AbstractMatrix` whose columns are competing forecasts). Forecast errors follow the convention ``e_t = y_t - \hat{y}_t``.

```math
\text{RMSE} = \sqrt{\tfrac{1}{T}\textstyle\sum_t e_t^2}, \quad
\text{MASE} = \frac{\text{MAE}}{\tfrac{1}{T-m}\sum_{t>m}|y_t - y_{t-m}|}, \quad
U2 = \sqrt{\frac{\sum_t \big((\hat y_{t+1}-y_{t+1})/y_t\big)^2}{\sum_t \big((y_{t+1}-y_t)/y_t\big)^2}}
```

where

- ``e_t = y_t - \hat{y}_t`` — forecast error,
- ``m`` — `seasonal_period`, the lag of the in-sample naive benchmark used to scale MASE (Hyndman & Koehler 2006),
- ``U2`` — Theil's inequality coefficient relative to the no-change (random-walk) forecast; ``U2 = 1`` exactly for the naive forecast, ``U2 < 1`` when the forecast beats it.

!!! note "MSE decomposition"
    The mean squared error splits into three parts, ``\text{MSE} = (\bar{\hat y} - \bar y)^2 + (s_{\hat y} - s_y)^2 + 2(1-\rho)s_{\hat y}s_y``, whose proportions (bias, variance, covariance) sum to one. A well-specified forecast concentrates its error in the covariance term; large bias or variance proportions signal systematic error.

```@example fceval
ev = forecast_evaluate(actual, f1)
(rmse = ev.values[1, 3], u2 = ev.values[1, 8], decomp_sum = sum(ev.decomp[1, :]))
```

MAPE and sMAPE skip observations with (near-)zero denominators, so a series that touches zero does not blow up the percentage errors.

**Return value** ([`ForecastEvaluation`](@ref)):

| Field | Type | Description |
|---|---|---|
| `models` | `Vector{String}` | Model labels |
| `metrics` | `Vector{String}` | Metric names, in table order |
| `values` | `Matrix{T}` | `n_models × 8` metric values |
| `decomp` | `Matrix{T}` | `n_models × 3` Theil MSE decomposition proportions |
| `n` | `Int` | Number of evaluation points |

---

## Diebold–Mariano Test

[`diebold_mariano`](@ref) tests the null of equal predictive accuracy between two forecasts using their error series. With loss differential ``d_t = g(e_{1t}) - g(e_{2t})``,

```math
\text{DM} = \frac{\bar d}{\sqrt{\hat V / T}}, \qquad
\hat V = \hat\gamma_0 + 2\sum_{j=1}^{h-1} \hat\gamma_j
```

where

- ``g`` — the loss: squared (`loss=:se`), absolute (`loss=:ad`), or a user-supplied function,
- ``\hat V`` — the truncated (rectangular-kernel) HAC long-run variance of ``d_t`` at lag ``h-1``,
- ``h`` — the forecast horizon.

With `hln=true` (default) the Harvey–Leybourne–Newbold (1997) factor ``\sqrt{(T+1-2h+h(h-1)/T)/T}`` multiplies the statistic, which is then referenced to ``t_{T-1}`` — matching R's `forecast::dm.test`. A positive `DM` means model 1 has the larger average loss.

```@example fceval
dm = diebold_mariano(actual .- f1, actual .- f2; h=4, loss=:se)
(stat = dm.statistic, pvalue = dm.pvalue)
```

!!! warning "Nested models"
    The DM test is **invalid for nested models** — under the null the loss differential is degenerate. Use [`clark_west`](@ref) instead.

---

## Clark–West Test (Nested Models)

When the small (restricted) model is nested in the big (unrestricted) one, [`clark_west`](@ref) forms the adjusted MSPE differential

```math
\hat f_t = e_{\text{small},t}^2 - \Big(e_{\text{big},t}^2 - (\hat y_{\text{small},t} - \hat y_{\text{big},t})^2\Big)
```

and tests ``E[\hat f]\le 0`` (the big model does not improve MSPE) against the one-sided `greater` alternative, referencing the standard normal (Clark & West 2007). The third argument is the gap between the two point forecasts.

```@example fceval
# Nested example: small = sample mean, big = a genuine signal
y = actual
f_small = fill(sum(y)/length(y), length(y))
f_big = f1
cw = clark_west(y .- f_small, y .- f_big, f_small .- f_big; h=1)
(stat = cw.statistic, pvalue = cw.pvalue)
```

---

## Efficiency & Encompassing

[`mincer_zarnowitz`](@ref) runs the efficiency regression ``y_t = a + b\,\hat y_t + u_t`` and jointly tests ``(a,b)=(0,1)`` with a Newey–West HAC covariance (truncation lag `lags`). A weakly efficient forecast satisfies ``a=0``, ``b=1``.

```@example fceval
mz = mincer_zarnowitz(actual, f1; lags=4)
report(mz)
```

[`forecast_encompassing`](@ref) estimates ``y_t = a + b_1\hat y_{1t} + b_2\hat y_{2t} + u_t`` and tests ``b_2 = 0`` (Harvey, Leybourne & Newbold 1998). Non-rejection means forecast 1 encompasses forecast 2 — the second forecast adds no incremental information.

```@example fceval
enc = forecast_encompassing(actual, f1, f2; lags=4)
(b2 = enc.b2, tstat = enc.tstat, pvalue = enc.pvalue)
```

---

## Forecast Combination

[`combine_forecasts`](@ref) blends the columns of a `T×M` forecast matrix into a single series:

- `:equal` — the simple average, ``w_i = 1/M`` (Bates & Granger 1969); robust and estimation-free.
- `:bates_granger` — inverse-MSE weights ``w_i \propto 1/\text{MSE}_i``, normalized; ignores cross-forecast error correlation.
- `:granger_ramanathan` — constrained least squares minimizing ``\|y - Fw\|^2`` subject to ``\mathbf{1}'w = 1`` (Granger & Ramanathan 1984), solved in closed form via the KKT system. Weights may be **negative** — this is intended and no clamping is applied.

```@example fceval
comb = combine_forecasts(hcat(f1, f2), actual; method=:granger_ramanathan)
(weights = comb.weights, sum_w = sum(comb.weights))
```

The equal-weights combination is guaranteed to beat the worst individual forecast on RMSE, which is why it is a hard baseline in forecasting competitions.

---

## Complete Example

```@example fceval
# Three forecasts of one target, scored, raced, and combined.
using MacroEconometricModels
models = ["A", "B", "C"]
f3 = actual .+ 0.3 .* e1 .- 0.2 .* e2    # a third, distinct forecast
F = hcat(f1, f2, f3)

# 1. Accuracy table
ev = forecast_evaluate(actual, F; model_names=models)
report(ev)
```

```@example fceval
# 2. Is Model A significantly better than Model B?
dm = diebold_mariano(actual .- F[:,1], actual .- F[:,2]; h=1, loss=:se)
println("DM stat = ", round(dm.statistic, digits=3), ", p = ", round(dm.pvalue, digits=4))

# 3. Optimal linear combination
comb = combine_forecasts(F, actual; method=:granger_ramanathan, model_names=models)
report(comb)
```

```julia
# 4. Visualize the accuracy race
plot_result(ev; metric="RMSE")
```

---

## Common Pitfalls

1. **Passing forecasts where errors are expected.** [`diebold_mariano`](@ref) and [`clark_west`](@ref) take forecast **error** series (`actual .- forecast`), while [`forecast_evaluate`](@ref), [`mincer_zarnowitz`](@ref), and [`combine_forecasts`](@ref) take the forecasts themselves alongside `actual`.
2. **Using DM on nested models.** The DM statistic is degenerate under the null when one model nests the other. Switch to [`clark_west`](@ref).
3. **Forgetting the horizon.** For `h`-step forecasts the loss differential is serially correlated up to lag `h-1`; pass `h` so the HAC variance uses the right truncation lag. The default `h=1` assumes no autocorrelation.
4. **Expecting non-negative Granger–Ramanathan weights.** Only the sum-to-one constraint is imposed; negative weights are a genuine feature of the constrained least-squares solution.
5. **MASE without an in-sample series.** By default MASE scales by the naive-forecast MAE of the *evaluation* actuals. Pass `insample=` to scale by the true in-sample benchmark, as intended by Hyndman & Koehler (2006).

---

## API

```@docs
forecast_evaluate
diebold_mariano
clark_west
mincer_zarnowitz
forecast_encompassing
combine_forecasts
ForecastEvaluation
DMTestResult
ClarkWestResult
MincerZarnowitzResult
ForecastEncompassingResult
ForecastCombination
```

---

## References

- Bates, J. M. and Granger, C. W. J. (1969). "The Combination of Forecasts." *Operational Research Quarterly* 20(4), 451–468. [doi:10.1057/jors.1969.103](https://doi.org/10.1057/jors.1969.103)
- Clark, T. E. and West, K. D. (2007). "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *Journal of Econometrics* 138(1), 291–311. [doi:10.1016/j.jeconom.2006.05.023](https://doi.org/10.1016/j.jeconom.2006.05.023)
- Diebold, F. X. and Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3), 253–263. [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- Granger, C. W. J. and Ramanathan, R. (1984). "Improved Methods of Combining Forecasts." *Journal of Forecasting* 3(2), 197–204. [doi:10.1002/for.3980030207](https://doi.org/10.1002/for.3980030207)
- Harvey, D., Leybourne, S. and Newbold, P. (1997). "Testing the Equality of Prediction Mean Squared Errors." *International Journal of Forecasting* 13(2), 281–291. [doi:10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4)
- Harvey, D. I., Leybourne, S. J. and Newbold, P. (1998). "Tests for Forecast Encompassing." *Journal of Business & Economic Statistics* 16(2), 254–259. [doi:10.1080/07350015.1998.10524759](https://doi.org/10.1080/07350015.1998.10524759)
- Hyndman, R. J. and Koehler, A. B. (2006). "Another Look at Measures of Forecast Accuracy." *International Journal of Forecasting* 22(4), 679–688. [doi:10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
- Mincer, J. and Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts." In J. Mincer (ed.), *Economic Forecasts and Expectations*, NBER, 3–46.
- Theil, H. (1966). *Applied Economic Forecasting*. North-Holland, Amsterdam.
