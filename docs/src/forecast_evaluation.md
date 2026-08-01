# [Forecast Evaluation & Combination](@id forecast_evaluation_page)

Estimators across the package return forecasts; deciding which of them to believe is a
separate job. `src/fceval` is a model-agnostic suite that scores point forecasts, tests
whether one forecast beats another, and blends competing forecasts into a single series.
It touches no forecast type: every function consumes a plain vector of realized values
together with a vector or matrix of forecasts, so output from a [VAR](@ref var_page), an
[ARIMA](@ref arima_page), a [MIDAS](@ref midas_page) regression, or a spreadsheet is
scored the same way.

The suite provides:

- **Accuracy metrics** --- ME, MAE, RMSE, MAPE, sMAPE, MASE, Theil's ``U_1``/``U_2``, and the Theil MSE bias/variance/covariance decomposition ([`forecast_evaluate`](@ref)).
- **Equal-accuracy test** --- Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction ([`diebold_mariano`](@ref)).
- **Nested-model test** --- Clark-West (2007), the correct test when one model nests the other ([`clark_west`](@ref)).
- **Efficiency and encompassing** --- the Mincer-Zarnowitz (1969) efficiency regression ([`mincer_zarnowitz`](@ref)) and the Harvey-Leybourne-Newbold (1998) encompassing test ([`forecast_encompassing`](@ref)).
- **Combination** --- equal, Bates-Granger inverse-MSE, and Granger-Ramanathan constrained least-squares weights ([`combine_forecasts`](@ref)).

## Quick Start

The examples race three one-step-ahead forecasts of quarterly US real GDP growth
(FRED-QD `GDPC1`, log differences in percent) over the 80 quarters ending in 2019Q4: an
expanding-window AR(2), the no-change forecast, and the historical mean. The sample
stops before 2020 deliberately --- the ``-8.2\%`` pandemic quarter is a single
observation that would dominate every squared-error metric on the page.

```@setup fceval
using MacroEconometricModels, Statistics
qd = load_example(:fred_qd)
gdp = vec(to_matrix(apply_tcode(qd[:, ["GDPC1"]]))) .* 100
gdp = gdp[1:243]     # through 2019Q4
```

**Recipe 1: Build a pseudo-out-of-sample race**

```@example fceval
f_ar, f_rw, f_mean = Float64[], Float64[], Float64[]
for t in (length(gdp) - 79):length(gdp)
    train = gdp[1:t-1]
    push!(f_ar, forecast(estimate_ar(train, 2), 1).forecast[1])  # expanding-window AR(2)
    push!(f_rw, train[end])                                      # no-change forecast
    push!(f_mean, mean(train))                                   # historical mean
end
actual = gdp[end-79:end]
F, models = hcat(f_ar, f_rw, f_mean), ["AR(2)", "Random walk", "Mean"]
size(F)
```

**Recipe 2: Score the competitors**

```@example fceval
ev = forecast_evaluate(actual, F; model_names=models)
report(ev)
```

**Recipe 3: Test equal predictive accuracy**

```@example fceval
dm = diebold_mariano(actual .- f_ar, actual .- f_rw; h=1, loss=:se)
report(dm)
```

**Recipe 4: Combine the three forecasts**

```@example fceval
comb = combine_forecasts(F, actual; method=:granger_ramanathan, model_names=models)
report(comb)
```

---

## Point Accuracy Metrics

[`forecast_evaluate`](@ref) reports the standard scale-dependent and scale-free accuracy
measures for one forecast (an `AbstractVector`) or several (a ``T \times M``
`AbstractMatrix` whose columns are competing forecasts). Errors follow the convention
``e_t = y_t - \hat y_t``.

```math
\text{RMSE} = \sqrt{\tfrac{1}{T}\textstyle\sum_t e_t^2}, \qquad
\text{MASE} = \frac{\text{MAE}}{\tfrac{1}{T-m}\sum_{t>m}|y_t - y_{t-m}|}, \qquad
U_2 = \sqrt{\frac{\sum_t \big((\hat y_{t+1}-y_{t+1})/y_t\big)^2}
                 {\sum_t \big((y_{t+1}-y_t)/y_t\big)^2}}
```

where

- ``e_t = y_t - \hat y_t`` is the forecast error,
- ``m`` is `seasonal_period`, the lag of the naive benchmark that scales MASE (Hyndman & Koehler 2006),
- ``U_2`` is Theil's inequality coefficient against the no-change forecast: ``U_2 = 1`` exactly for the naive forecast, ``U_2 < 1`` when the forecast beats it.

In the race above the AR(2) posts an RMSE of 0.5669 against 0.6783 for the random walk
and 0.6429 for the historical mean --- improvements of 16% and 12%. Its MASE of 0.7643
confirms the same ranking against the naive benchmark, and its ``U_2`` of 0.5161 puts its
squared error at roughly a quarter of the no-change forecast's. The random walk's
``U_2`` is exactly 1.0000 because it *is* the no-change forecast, which is the cheapest
available check that the metrics line up with the data. MAPE, by contrast, reads 134% for
the best model on the page: quarterly growth crosses zero repeatedly, and percentage
errors carry no information on a series whose denominator sits near zero.

!!! note "MSE decomposition"
    The mean squared error splits into three parts,
    ``\text{MSE} = (\bar{\hat y} - \bar y)^2 + (s_{\hat y} - s_y)^2 + 2(1-\rho)s_{\hat y}s_y``,
    whose proportions --- bias, variance, covariance --- sum to one. A well-specified
    forecast concentrates its error in the covariance term; large bias or variance
    proportions signal systematic error rather than bad luck.

The decomposition printed below the accuracy table separates the three failure modes.
The AR(2) carries 41% of its MSE in the variance term: its forecasts have a standard
deviation of 0.214 against 0.578 for realized growth, the familiar shrinkage of a
mean-reverting model that will not predict a recession. The historical mean is worse
still at 69%, since it barely varies at all. The random walk puts 99.9% of its error in
the covariance term --- unbiased and correctly scaled, its problem is purely timing.

```@example fceval
ev1 = forecast_evaluate(actual, f_ar)
(rmse = ev1.values[1, 3], u2 = ev1.values[1, 8], decomp_sum = sum(ev1.decomp[1, :]))
```

MAPE and sMAPE skip observations with (near-)zero denominators, so a series that touches
zero does not blow up the percentage errors --- but as the 134% above shows, surviving is
not the same as being informative.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `seasonal_period` | `Int` | `1` | Lag of the naive benchmark that scales MASE |
| `insample` | `AbstractVector` or `nothing` | `nothing` | In-sample series for the MASE scale; defaults to the evaluation actuals |
| `model_names` | `AbstractVector{<:AbstractString}` or `nothing` | `nothing` | Column labels, defaulting to `"Model j"` |

**Return value** ([`ForecastEvaluation`](@ref)):

| Field | Type | Description |
|---|---|---|
| `models` | `Vector{String}` | Model labels |
| `metrics` | `Vector{String}` | Metric names in table order |
| `values` | `Matrix{T}` | ``n_{models} \times 8`` metric values |
| `decomp` | `Matrix{T}` | ``n_{models} \times 3`` Theil MSE decomposition proportions |
| `n` | `Int` | Number of evaluation points |

---

## Diebold-Mariano Test

[`diebold_mariano`](@ref) tests the null of equal predictive accuracy between two
forecasts from their error series. With loss differential
``d_t = g(e_{1t}) - g(e_{2t})``,

```math
\text{DM} = \frac{\bar d}{\sqrt{\hat V / T}}, \qquad
\hat V = \hat\gamma_0 + 2\sum_{j=1}^{h-1} w_j \hat\gamma_j
```

where

- ``g`` is the loss: squared (`loss=:se`), absolute (`loss=:ad`), or any user-supplied function of a scalar error,
- ``\hat V`` is the truncated HAC long-run variance of ``d_t`` at lag ``h-1``, with ``w_j = 1`` under the default rectangular kernel and ``w_j = 1 - j/h`` under `kernel=:bartlett`,
- ``h`` is the forecast horizon.

With `hln=true` (the default) the Harvey-Leybourne-Newbold (1997) factor
``\sqrt{(T+1-2h+h(h-1)/T)/T}`` multiplies the statistic, which is then referenced to
``t_{T-1}``, matching R's `forecast::dm.test`. A **positive** DM means model 1 carries the
larger average loss.

Recipe 3 returns ``\text{DM} = -1.8217`` with ``p = 0.0723``: the AR(2) carries the
smaller loss --- its mean squared error is 0.1388 below the random walk's, which is
exactly the reported ``\bar d`` --- but at 80 quarters a 16% RMSE advantage still falls
short of significance at the 5% level. The DM test is a statement about *average* loss,
and squared loss here is dominated by a handful of recession quarters that both models
miss. Switching to absolute loss makes the same comparison decisive:

```@example fceval
dm_ad = diebold_mariano(actual .- f_ar, actual .- f_rw; h=1, loss=:ad)
(statistic = round(dm_ad.statistic; digits=4), pvalue = round(dm_ad.pvalue; digits=4))
```

Under absolute loss the statistic is ``-3.68`` with ``p = 0.0004``. The AR(2) is reliably
better through the middle of the error distribution, and the squared-loss test is paying
for a few large misses common to both forecasts. Reporting both losses is standard
practice for exactly this reason.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `h` | `Int` | `1` | Forecast horizon; sets the HAC truncation lag ``h-1`` |
| `loss` | `Symbol` or `Function` | `:se` | `:se`, `:ad`, or a user-supplied loss of a scalar error |
| `hln` | `Bool` | `true` | Apply the Harvey-Leybourne-Newbold correction and reference ``t_{T-1}`` |
| `kernel` | `Symbol` | `:rectangular` | `:rectangular` (matching R) or `:bartlett` |
| `alternative` | `Symbol` | `:two_sided` | `:two_sided`, `:less`, or `:greater` |

**Return value** ([`DMTestResult`](@ref)): `statistic`, `pvalue`, `dbar` (the mean loss
differential), `lrvar` (``\hat V``), `h`, `loss`, `hln`, `alternative`, `T_obs`.

!!! warning "Nested models"
    The DM test is **invalid for nested models** --- under the null the loss differential
    is degenerate and the statistic is not asymptotically normal. Use
    [`clark_west`](@ref) instead.

---

## Clark-West Test (Nested Models)

When the small (restricted) model is nested in the big (unrestricted) one,
[`clark_west`](@ref) forms the adjusted MSPE differential

```math
\hat f_t = e_{\text{small},t}^2
           - \Big(e_{\text{big},t}^2 - (\hat y_{\text{small},t} - \hat y_{\text{big},t})^2\Big)
```

and tests ``E[\hat f] \le 0`` --- the big model does not improve MSPE --- against the
one-sided `greater` alternative, referencing the standard normal (Clark & West 2007). The
third argument is the gap between the two point forecasts, not an error series.

The historical-mean forecast is a constant-only regression, so it is nested in the AR(2),
and this is exactly the comparison the DM test may not be used for:

```@example fceval
cw = clark_west(actual .- f_mean, actual .- f_ar, f_mean .- f_ar; h=1)
report(cw)
```

The statistic of 2.4407 rejects at ``p = 0.0073``: the two autoregressive coefficients
earn their keep. The mechanics are visible in ``\bar{\hat f} = 0.1483``. The raw MSPE gap
between the mean forecast and the AR(2) is only 0.0920 (0.4133 against 0.3213); the
adjustment term, the mean squared distance between the two forecasts, supplies the
remaining 0.0563. That term is the estimation noise the larger model necessarily injects
under the null, and adding it back is what makes the test valid where DM is not.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `h` | `Int` | `1` | Forecast horizon; sets the HAC truncation lag ``h-1`` |
| `alternative` | `Symbol` | `:greater` | `:greater` (the Clark-West default), `:two_sided`, or `:less` |

**Return value** ([`ClarkWestResult`](@ref)): `statistic`, `pvalue`, `fbar`, `lrvar`,
`h`, `alternative`, `T_obs`.

---

## Efficiency & Encompassing

[`mincer_zarnowitz`](@ref) runs the efficiency regression ``y_t = a + b\,\hat y_t + u_t``
and jointly tests ``(a,b) = (0,1)`` with a Newey-West HAC covariance at truncation lag
`lags`. A weakly efficient forecast satisfies ``a = 0`` and ``b = 1``: it is unbiased and
its variation is scaled correctly.

```@example fceval
mz = mincer_zarnowitz(actual, f_ar; lags=4)
report(mz)
```

Read the two blocks against each other. Individually the AR(2) looks efficient: the
intercept of ``-0.094`` is nowhere near significant (``p = 0.73``) and the slope 0.8967
has a 95% interval of ``[0.260, 1.533]`` that comfortably contains one. Jointly the
forecast fails, with ``\chi^2(2) = 13.89`` and ``p < 0.001``. The two readings are not in
conflict: the HAC covariance of ``(a, b)`` has a correlation of ``-0.98``, so the pair
travels a long way along that ridge without either marginal statistic moving, while the
direction that separates it from ``(0, 1)`` is estimated sharply. This is why the joint
Wald test, and not two ``t``-tests, is the Mincer-Zarnowitz test.

The HAC lag matters. The default `lags=0` gives the White (HC0) sandwich rather than a
Newey-West covariance and reports ``\chi^2(2) = 8.98`` (``p = 0.011``) --- the same
verdict, less sharply drawn. For quarterly one-step forecasts `lags=4` is the
conventional choice.

[`forecast_encompassing`](@ref) estimates
``y_t = a + b_1 \hat y_{1t} + b_2 \hat y_{2t} + u_t`` and tests ``b_2 = 0`` (Harvey,
Leybourne & Newbold 1998). Non-rejection means forecast 1 encompasses forecast 2 --- the
second forecast adds no incremental information.

```@example fceval
enc_ar = forecast_encompassing(actual, f_ar, f_rw; lags=4)   # does AR(2) encompass RW?
enc_rw = forecast_encompassing(actual, f_rw, f_ar; lags=4)   # and the reverse?
(ar_encompasses_rw = (b2 = round(enc_ar.b2; digits=4), p = round(enc_ar.pvalue; digits=4)),
 rw_encompasses_ar = (b2 = round(enc_rw.b2; digits=4), p = round(enc_rw.pvalue; digits=4)))
```

The first test gives ``b_2 = 0.133`` with ``p = 0.645``, which reads as the AR(2)
encompassing the random walk. The reverse test gives ``b_2 = 0.590`` with ``p = 0.222``
and also fails to reject. Both directions surviving is a power problem rather than a
paradox: with 80 quarters and two forecasts that share most of their information, the
encompassing regression cannot separate them. A non-rejection is evidence of a weak test
at least as often as it is evidence of encompassing.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `lags` | `Int` | `0` | Newey-West truncation lag; `lags=0` gives the White (HC0) sandwich |
| `kernel` | `Symbol` | `:bartlett` | Kernel passed to `newey_west` when `lags >= 1` |

**Return values**: [`MincerZarnowitzResult`](@ref) carries `a`, `b`, `se`, `wald`,
`pvalue_wald`, `fstat`, `pvalue_f`, `lags`, `kernel`, `T_obs`;
[`ForecastEncompassingResult`](@ref) carries `b1`, `b2`, `se_b2`, `tstat`, `pvalue`
(two-sided, referenced to ``t_{T-3}``), `lags`, `kernel`, `T_obs`.

---

## Forecast Combination

[`combine_forecasts`](@ref) blends the columns of a ``T \times M`` forecast matrix into a
single series (Timmermann 2006):

- `:equal` (the default) --- the simple average ``w_i = 1/M`` (Bates & Granger 1969); robust and estimation-free.
- `:bates_granger` --- inverse-MSE weights ``w_i \propto 1/\text{MSE}_i``, normalized to sum to one; ignores cross-forecast error correlation and requires strictly positive MSEs.
- `:granger_ramanathan` --- constrained least squares minimizing ``\|y - Fw\|^2`` subject to ``\mathbf{1}'w = 1`` (Granger & Ramanathan 1984), solved in closed form through the KKT system. Weights may be **negative**; this is intended, and no clamping is applied.

```@example fceval
schemes = map((:equal, :bates_granger, :granger_ramanathan)) do method
    c = combine_forecasts(F, actual; method=method, model_names=models)
    (method = method, weights = round.(c.weights; digits=4),
     mse = round(mean(abs2, actual .- c.combined); digits=4))
end
```

The three schemes tell one story. Equal weights land at an MSE of 0.3230 --- better than
the average of the individual MSEs (0.3983, as Jensen's inequality guarantees) and far
better than the worst single model (0.4601), but not better than the AR(2) alone
(0.3213). Inverse-MSE weights tilt towards the AR(2) (0.404 against 0.282 and 0.314) and
gain almost nothing, because they ignore that all three forecasts err together. Only
Granger-Ramanathan improves on the best individual model, reaching 0.3150 by loading
1.114 on the AR(2) and shorting the historical mean at ``-0.199`` --- and it does so by
fitting three weights on the same 80 observations it is then scored against. Out of
sample that advantage routinely evaporates, which is why the equal-weight combination's
stubborn competitiveness is known as the forecast combination puzzle.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `method` | `Symbol` | `:equal` | `:equal`, `:bates_granger`, or `:granger_ramanathan` |
| `model_names` | `AbstractVector{<:AbstractString}` or `nothing` | `nothing` | Model labels, defaulting to `"Model j"` |

**Return value** ([`ForecastCombination`](@ref)): `weights`, `combined` (the series
``Fw``), `method`, `mse` (the individual-model MSEs), `models`.

---

## Visualization

Every result type on this page has a `plot_result` dispatch:

```julia
plot_result(ev)                      # grouped bar: all models within each metric
plot_result(ev; metric="RMSE")       # a single metric, ranked best first
plot_result(ev; view=:theil)         # stacked Theil bias/variance/covariance bars
plot_result(mz)                      # efficiency line against the 45-degree reference
plot_result(dm)                      # loss differential with its 95% interval
plot_result(comb)                    # combination weights and standalone MSEs
```

The `MincerZarnowitzResult` plot draws the fitted efficiency line alone: the result
stores no forecast or actual series, so there is no scatter to overlay.

---

## Complete Example

```@example fceval
# Score the three individual forecasts alongside two combinations of them
ceq = combine_forecasts(F, actual; method=:equal, model_names=models)
cgr = combine_forecasts(F, actual; method=:granger_ramanathan, model_names=models)

ev_all = forecast_evaluate(actual, hcat(F, ceq.combined, cgr.combined);
                           model_names=vcat(models, ["Equal-weight", "Granger-Ramanathan"]))
report(ev_all)
```

```@example fceval
# Is the Granger-Ramanathan combination significantly better than the AR(2) alone?
dm_comb = diebold_mariano(actual .- cgr.combined, actual .- f_ar; h=1, loss=:se)
report(dm_comb)
```

The combined rows confirm what the weights implied: Granger-Ramanathan trims RMSE from
0.5669 to 0.5613 and cuts the variance share of the MSE decomposition from 41% to 28%,
while the equal-weight combination lands slightly behind the AR(2) at 0.5684. The DM test
of the combination against its own strongest component returns ``-0.457`` with
``p = 0.649``: the improvement cannot be distinguished from noise, the expected verdict
when the weights were fitted on the same 80 observations used to score them. Note
also that ``U_2`` ranks both combinations *below* the AR(2) even though RMSE ranks
Granger-Ramanathan above it: ``U_2`` divides each error by the previous quarter's growth
rate, so it re-weights the sample towards quarters that RMSE treats as ordinary.

---

## Common Pitfalls

1. **Passing forecasts where errors are expected.** [`diebold_mariano`](@ref) and
   [`clark_west`](@ref) take forecast **error** series (`actual .- forecast`), while
   [`forecast_evaluate`](@ref), [`mincer_zarnowitz`](@ref), and
   [`combine_forecasts`](@ref) take the forecasts themselves alongside `actual`. Both
   run without complaint on the wrong input.
2. **Using DM on nested models.** The DM statistic is degenerate under the null when one
   model nests the other. A constant-only forecast against an AR, or an AR(1) against an
   AR(2), needs [`clark_west`](@ref).
3. **Forgetting the horizon.** For ``h``-step forecasts the loss differential is serially
   correlated up to lag ``h-1``; pass `h` so the HAC variance uses the right truncation
   lag. The default `h=1` assumes no autocorrelation.
4. **Reading percentage errors on a near-zero series.** MAPE and sMAPE carry no
   information for growth rates that cross zero --- 134% for the winning model above. Use
   RMSE, MASE, or ``U_2`` on such series.
5. **Letting one observation decide.** A single pandemic-sized outlier reverses every
   squared-error ranking on this page. Report absolute-loss results alongside
   squared-loss ones, and state the evaluation window explicitly.
6. **Expecting non-negative Granger-Ramanathan weights.** Only the sum-to-one constraint
   is imposed; negative weights are a genuine feature of the constrained least-squares
   solution, not a numerical failure.
7. **MASE without an in-sample series.** By default MASE scales by the naive-forecast MAE
   of the *evaluation* actuals, which is why the random walk above scores 1.0172 rather
   than exactly 1. Pass `insample=` to scale by the true in-sample benchmark, as intended
   by Hyndman & Koehler (2006).

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

- Bates, J. M., & Granger, C. W. J. (1969). The Combination of Forecasts.
  *Operational Research Quarterly*, 20(4), 451--468.
  [10.1057/jors.1969.103](https://doi.org/10.1057/jors.1969.103)
- Clark, T. E., & West, K. D. (2007). Approximately Normal Tests for Equal Predictive
  Accuracy in Nested Models. *Journal of Econometrics*, 138(1), 291--311.
  [10.1016/j.jeconom.2006.05.023](https://doi.org/10.1016/j.jeconom.2006.05.023)
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of
  Business & Economic Statistics*, 13(3), 253--263.
  [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- Granger, C. W. J., & Ramanathan, R. (1984). Improved Methods of Combining Forecasts.
  *Journal of Forecasting*, 3(2), 197--204.
  [10.1002/for.3980030207](https://doi.org/10.1002/for.3980030207)
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of Prediction
  Mean Squared Errors. *International Journal of Forecasting*, 13(2), 281--291.
  [10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4)
- Harvey, D. I., Leybourne, S. J., & Newbold, P. (1998). Tests for Forecast Encompassing.
  *Journal of Business & Economic Statistics*, 16(2), 254--259.
  [10.1080/07350015.1998.10524759](https://doi.org/10.1080/07350015.1998.10524759)
- Hyndman, R. J., & Koehler, A. B. (2006). Another Look at Measures of Forecast Accuracy.
  *International Journal of Forecasting*, 22(4), 679--688.
  [10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
- Mincer, J., & Zarnowitz, V. (1969). The Evaluation of Economic Forecasts. In J. Mincer
  (ed.), *Economic Forecasts and Expectations: Analysis of Forecasting Behavior and
  Performance*. New York: NBER. ISBN 0-87014-202-X.
  [NBER chapter c1214](https://www.nber.org/chapters/c1214)
- Theil, H. (1966). *Applied Economic Forecasting*. Amsterdam: North-Holland.
- Timmermann, A. (2006). Forecast Combinations. In G. Elliott, C. W. J. Granger, &
  A. Timmermann (eds.), *Handbook of Economic Forecasting*, Vol. 1, 135--196. Amsterdam:
  Elsevier. [10.1016/S1574-0706(05)01004-9](https://doi.org/10.1016/S1574-0706(05)01004-9)
