# [MIDAS Regression](@id midas_page)

**MIDAS** (MIxed-DAta Sampling) regression relates a low-frequency target to many
high-frequency lags of an indicator through a parsimonious weighting function
(Ghysels, Santa-Clara & Valkanov 2006). Where [bridge equations](@ref nowcast_bridge_page)
time-aggregate a monthly indicator to the target frequency before regressing, MIDAS
keeps every high-frequency observation and lets a small parameter vector ``\theta``
shape their weights, so a monthly indicator's within-quarter timing enters the
regression without a discretionary aggregation rule. Central banks use MIDAS and
ADL-MIDAS to nowcast quarterly GDP from monthly and daily indicators
(Clements & Galvao 2008).

For an overview of all nowcasting methods and a method-comparison table, see
[Nowcasting](@ref nowcast_page).

This page covers:

- **Restricted MIDAS** with exponential-Almon, Beta, and polynomial-Almon weights
- **ADL-MIDAS**, adding autoregressive lags of the target
- **U-MIDAS**, the unrestricted lag polynomial estimated by OLS
- **Direct forecasting** with a Gaussian NLS prediction interval

## Quick Start

The examples regress quarterly real GDP growth (FRED-QD `GDPC1`) on monthly
industrial-production growth (FRED-MD `INDPRO`), both as log differences in percent.
In the January 2026 vintage the monthly indicator runs three months past the last
published quarter of GDP --- the ragged edge that MIDAS exists to exploit.

```@setup midas
using MacroEconometricModels, Statistics

md = load_example(:fred_md)
qd = load_example(:fred_qd)

# Monthly IP growth and quarterly GDP growth, in percent.
ip_monthly = vec(to_matrix(apply_tcode(md[:, ["INDPRO"]]))) .* 100
gdp_all    = vec(to_matrix(apply_tcode(qd[:, ["GDPC1"]]))) .* 100

# Estimation sample: drop the quarter whose GDP is not yet published, and the three
# monthly observations belonging to it, so both series end in the same period. The
# last 30 years keep the build fast.
gdp_q = gdp_all[1:end-1][end-119:end]
ip_m  = ip_monthly[1:end-3][end-359:end]
```

**Recipe 1: ADL-MIDAS with exponential-Almon weights**

```@example midas
# Quarterly GDP growth on six monthly IP lags plus one own lag
model = estimate_midas(gdp_q, ip_m; m=3, K=6, weights=:expalmon, p_ar=1)
report(model)
```

**Recipe 2: Read the estimated weight curve**

```@example midas
round.(midas_weights(model); digits=4)   # w(θ̂), most-recent-first
```

**Recipe 3: Nowcast the quarter that has no published GDP**

```@example midas
# Six monthly readings covering the unpublished quarter and the one before it
ip_new = reverse(ip_monthly[end-5:end])   # most-recent-first
fc = forecast(model, ip_new)
report(fc)
```

**Recipe 4: Compare the restricted fit against an unrestricted lag polynomial**

```@example midas
umodel = estimate_midas(gdp_q, ip_m; m=3, K=6, weights=:umidas, p_ar=1)
(expalmon_bic = round(model.bic; digits=2), umidas_bic = round(umodel.bic; digits=2),
 expalmon_par = dof(model), umidas_par = dof(umodel))
```

The restricted fit spends five parameters against eight and wins on BIC by 7.5 points,
so the three extra degrees of freedom the unrestricted polynomial buys do not pay for
themselves.

---

## Model and Frequency Alignment

Restricted MIDAS aggregates the ``K`` high-frequency lags through a normalized weight
``w_k(\theta)`` and estimates

```math
y_t = \beta_0 + \beta_1 \sum_{k=1}^{K} w_k(\theta)\, x_{t,k}
      + \sum_{j=1}^{p} \rho_j\, y_{t-j} + u_t,
\qquad \sum_{k=1}^{K} w_k(\theta) = 1,
```

where

- ``y_t`` is the low-frequency target in period ``t``,
- ``x_{t,k}`` is the ``k``-th high-frequency lag within period ``t``, counted most-recent-first,
- ``w_k(\theta)`` is the weight function, normalized to sum to one,
- ``\beta_1`` is the aggregate loading of the indicator and ``\rho_j`` the autoregressive coefficients,
- ``m`` is the frequency ratio: 3 for monthly-to-quarterly, roughly 66 for daily-to-quarterly.

Because the weights sum to one, ``\beta_1`` carries the entire scale of the
relationship --- it is the response of ``y_t`` to a one-unit move in the weighted
average of the indicator --- while ``\theta`` controls only the shape of that average.
The two are therefore separately interpretable, which is the practical reason to
normalize.

Alignment is positional, not calendar-based. `estimate_midas` anchors the **last**
high-frequency observation to the **last** low-frequency period and walks backwards in
blocks of ``m``, so period ``t`` receives ``[x_{t,m}, x_{t,m-1}, \ldots, x_{t,m-K+1}]``.
Early periods without a complete block of ``K`` lags, and periods without the requested
`p_ar` own lags, are dropped: the fit above retains 119 of the 120 quarters supplied.

!!! warning "Both series must end in the same period"
    Trim the high-frequency vector so that its final observation belongs to the final
    low-frequency period. A monthly indicator running three months past the last
    published quarter shifts *every* block by one quarter if it is passed untrimmed.

---

## Weight Functions

The `weights` keyword selects the functional form of ``w_k(\theta)``:

| `weights` | Form | Parameters |
|---|---|---|
| `:expalmon` | ``w_k \propto \exp(\theta_1 k + \theta_2 k^2)`` | 2 |
| `:beta2` | ``w_k \propto x_k^{\theta_1-1}(1-x_k)^{\theta_2-1}``, ``x_k`` on a grid over ``(0,1)`` | 2 |
| `:beta3` | Beta plus a level constant ``\theta_3`` | 3 |
| `:almon` | ``w_k \propto \sum_{d=0}^{D} \theta_{d+1}\, k^{d}``, then normalized | `poly_degree`+1 |
| `:umidas` | unrestricted lag coefficients, no weight function | ``K`` |

The exponential-Almon weight reduces to equal weights ``1/K`` at ``\theta = 0``, which
makes ``\theta = 0`` both the natural starting value and a useful sanity check.
`midas_weights` evaluates any scheme directly from parameters:

```@example midas
w_flat  = midas_weights([0.0, 0.0], 6)                 # θ = 0 ⇒ equal weights
w_decay = midas_weights([0.3, -0.05], 6)               # gentle hump, then decay
w_beta  = midas_weights([2.0, 4.0], 6; kind=:beta2)    # front-loaded Beta
round.(hcat(w_flat, w_decay, w_beta); digits=4)
```

Column 1 is flat at ``1/6 = 0.1667``. Column 2 peaks mildly at lag 3 (0.1931) and
decays to 0.1232 by lag 6 --- a near-uniform shape, because ``\theta_2 = -0.05`` bends
the exponent only slightly. Column 3 puts 81% of the mass on lags 2 and 3 and exactly
zero on both endpoints: the Beta grid runs from ``\delta`` to ``1-\delta`` with
``\delta = 10^{-8}``, so ``x_1^{\theta_1-1}`` and ``(1-x_K)^{\theta_2-1}`` underflow
whenever both shape parameters exceed one.

The estimated exponential-Almon curve from Recipe 1 is
``(0.011, 0.128, 0.410, 0.360, 0.086, 0.006)``, peaking at lags 3 and 4 --- the first
month of the target quarter and the last month of the preceding one, which together
carry 77% of the weight. That hump is what temporal aggregation predicts: the growth
rate of a quarterly average of a monthly series is a triangular ``(1,2,3,2,1)`` filter
over the five most recent monthly growth rates (Mariano & Murasawa 2003), and its peak
also falls on the third lag. Both ``\theta`` estimates are significant at 1%, so the
data reject the flat ``\theta = 0`` shape outright.

A Beta weight recovers the same hump:

```@example midas
bmodel = estimate_midas(gdp_q, ip_m; m=3, K=6, weights=:beta2, p_ar=1)
(theta = round.(bmodel.theta; digits=3), r2 = round(bmodel.r2; digits=4),
 bic = round(bmodel.bic; digits=2), weights = round.(midas_weights(bmodel); digits=3))
```

The Beta curve reaches 0.409 at lag 3 against 0.410 for exponential-Almon, and its BIC
is 197.81 against 197.45 --- a gap far below the conventional threshold of 2. When two
shapes with the same parameter count agree this closely, the weight family is not the
binding modelling choice; ``K`` and `p_ar` are.

---

## Estimation

Restricted MIDAS is nonlinear least squares. Given ``\theta``, the linear coefficients
``(\beta_0, \beta_1, \rho)`` are concentrated out by OLS and the profiled sum of squared
residuals is minimized over ``\theta`` with `Optim.LBFGS` and analytic gradients. The
exponential-Almon and Beta profiled surfaces carry flat ridges, so the estimator
restarts from a fixed grid of starting values --- five for `:expalmon` and `:beta2`,
four for `:beta3`, two for `:almon` --- and keeps the best minimum. Standard errors come
from the Gauss-Newton sandwich ``\hat\sigma^2 (J'J)^{-1}`` evaluated at the full
parameter vector ``[\beta; \theta]``. `:umidas` bypasses all of this: it is plain OLS on
the ``K`` stacked lags (Foroni, Marcellino & Schumacher 2015).

| Keyword | Type | Default | Description |
|---|---|---|---|
| `m` | `Int` | required | High-to-low frequency ratio |
| `K` | `Int` | required | Number of high-frequency lags |
| `weights` | `Symbol` | `:expalmon` | `:expalmon`, `:beta2`, `:beta3`, `:almon`, or `:umidas` |
| `p_ar` | `Int` | `0` | Autoregressive lags of the target (ADL-MIDAS) |
| `poly_degree` | `Int` | `2` | Polynomial degree for `:almon` |
| `h` | `Int` | `1` | Direct forecast horizon recorded on the model |
| `max_iter` | `Int` | `500` | LBFGS iteration cap per starting value |

!!! warning "`h` labels the horizon, it does not shift the data"
    `h` is stored on the model and printed by `report`, but the estimator always aligns
    the high-frequency block to the *same* low-frequency period. For a genuine direct
    ``h``-step regression, shift the inputs: estimate on `y_lf[(1+h):end]` against
    `X_hf[1:(end - h*m)]`.

**Return value** (`MidasModel`):

| Field | Type | Description |
|---|---|---|
| `beta` | `Vector{T}` | ``[\beta_0, \beta_1, \rho_1, \ldots]``, or ``[\beta_0, b_1, \ldots, b_K, \rho_1, \ldots]`` under `:umidas` |
| `theta` | `Vector{T}` | Weight parameters, empty under `:umidas` |
| `w` | `Vector{T}` | Realized weight curve ``w(\hat\theta)``, length ``K``, most-recent-first |
| `vcov_mat` | `Matrix{T}` | Gauss-Newton covariance of ``[\beta; \theta]`` |
| `fitted` / `residuals` | `Vector{T}` | In-sample fit and residuals over the retained periods |
| `ssr` / `sigma2` | `T` | Sum of squared residuals and ``\hat\sigma^2 = \text{SSR}/(n-p)`` |
| `r2` / `adj_r2` | `T` | Coefficient of determination, unadjusted and adjusted |
| `loglik` / `aic` / `bic` | `T` | Gaussian log-likelihood and information criteria |
| `converged` | `Bool` | NLS convergence flag, always `true` under `:umidas` |

The `StatsAPI` interface is available throughout: `coef`, `vcov`, `stderror`, `nobs`,
`dof`, `dof_residual`, `residuals`, `fitted`, `predict`, `aic`, `bic`,
`loglikelihood`, `r2`.

---

## ADL-MIDAS and U-MIDAS

Setting `p_ar > 0` adds autoregressive lags of the target, giving the **ADL-MIDAS**
specification used for persistent series (Clements & Galvao 2008). Setting
`weights=:umidas` drops the weight function and estimates the ``K`` lag coefficients by
ordinary least squares, which is competitive when the frequency ratio ``m`` is small
(Foroni, Marcellino & Schumacher 2015).

```@example midas
report(umodel)
```

The unrestricted coefficients trace the same hump the weight function imposed: lags 3,
4 and 5 are significant (0.541, 0.374, 0.151) while lags 1, 2 and 6 are not. That the
free polynomial *chooses* the shape the exponential-Almon function *assumes* is the
strongest available evidence that the restriction is not distorting the fit. U-MIDAS
buys a marginally higher ``R^2`` --- 0.8312 against 0.8213 --- with three more
parameters, so it wins on AIC (182.75 against 183.55) and loses on BIC (204.98 against
197.45). At ``m = 3`` the two are close; the gap widens with ``K``, and at
daily-to-quarterly frequencies the unrestricted polynomial is unusable.

The autoregressive term is small but informative. With the monthly indicator in the
equation the own lag enters at ``-0.141`` (``t = -3.39``): a mild correction to the
overshooting that industrial production alone produces, since manufacturing swings
harder than aggregate output. In the unrestricted fit the same term is insignificant,
absorbed by the six free lag coefficients.

---

## Forecasting

`forecast` produces a direct point forecast from a fresh high-frequency block, ordered
most-recent-first, together with a Gaussian prediction interval that combines residual
variance with parameter uncertainty:

```math
\widehat{\text{Var}}(\hat y_f) = \hat\sigma^2 + x_f' V x_f,
```

where

- ``x_f`` stacks the linear design row ``[1, s_f, y_{t-1}, \ldots]`` with the ``\theta``-gradient ``\beta_1 (\partial w/\partial\theta)' x^{hf}_f``,
- ``V`` is the estimated covariance of ``[\beta; \theta]``,
- ``s_f = \sum_k w_k(\hat\theta)\, x^{hf}_{f,k}`` is the weighted indicator for the forecast period.

The interval uses normal quantiles rather than ``t``: at the default `level=0.95` the
half-width is ``1.96\,\widehat{\text{se}}``.

```@example midas
fc = forecast(model, ip_new; level=0.68)
report(fc)
```

The point nowcast for the unpublished quarter is 0.453% quarter-on-quarter, below both
the sample mean (0.618%) and the last published quarter (1.071%): industrial production
over the six relevant months was weak enough to pull the weighted average below its own
mean. The prediction standard error of 0.515 barely exceeds the residual standard
deviation of 0.513, so parameter uncertainty contributes almost nothing at 119
observations. Even the 68% interval, ``[-0.059, 0.965]``, spans zero --- a contracting
quarter is not ruled out at one standard error.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `y_lags` | `AbstractVector` or `nothing` | `nothing` | Own lags, most-recent-first; defaults to the last retained in-sample targets |
| `level` | `Real` | `0.95` | Nominal coverage of the prediction interval |

**Return value** (`MidasForecast`): `forecast`, `se`, `ci_lower` and `ci_upper` (each a
length-1 `Vector{T}`), plus `horizon::Int` and `conf_level::T`.

---

## Visualization

`plot_result` offers three views of a fitted model:

```julia
plot_result(model; view=:weights)       # w_k against high-frequency lag k
plot_result(model; view=:fit)           # actual vs fitted low-frequency target
plot_result(model; view=:diagnostics)   # four-panel residual diagnostics
```

Check the `:weights` view first. A curve that has collapsed onto a single lag, or that
has run flat at ``\theta = 0``, signals that the profiled objective stalled on a ridge
rather than that the data prefer a degenerate shape.

```@raw html
<iframe src="../assets/plots/midas_weights.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The estimated exponential-Almon curve is single-peaked and interior: weight rises from
0.011 on the most recent month to 0.410 at lag 3, then decays to 0.006 by lag 6. The
quarter's information about GDP growth is concentrated in the middle of the preceding
two quarters rather than in the latest month — the hump shape that motivates a
parametric weight function over six free coefficients.

---

## Complete Example

```@example midas
# Estimate, verify the weight shape, then nowcast the unpublished quarter
model = estimate_midas(gdp_q, ip_m; m=3, K=6, weights=:expalmon, p_ar=1)
w = midas_weights(model)

(n = nobs(model), r2 = round(model.r2; digits=4), converged = model.converged,
 peak_lag = argmax(w), peak_weight = round(maximum(w); digits=4),
 own_quarter_share = round(sum(w[1:3]); digits=4))
```

```@example midas
fc = forecast(model, reverse(ip_monthly[end-5:end]); level=0.95)
report(fc)
```

The three months of the target quarter carry 55% of the weight and the three months of
the preceding quarter carry the remaining 45%, so this nowcast is a genuinely
mixed-frequency object: nearly half its content is information the target quarter's own
data have not yet supplied. To score a sequence of such nowcasts against a benchmark,
pass them to [Forecast Evaluation](@ref forecast_evaluation_page).

---

## Common Pitfalls

1. **High-frequency ordering.** `X_hf` is chronological and its **last** observation
   must belong to the **last** target period. Within a period the lags enter
   most-recent-first, so `midas_weights(model)[1]` is the weight on the most recent
   high-frequency observation, not the oldest.
2. **Ragged edges are trimmed silently.** Periods without a full block of ``K`` lags, or
   without the requested `p_ar` own lags, are dropped, so the estimation sample is
   shorter than `length(y_lf)`. Check `nobs(model)` against your expectation.
3. **Flat ridges in the profiled objective.** The exponential-Almon and Beta surfaces
   have flat regions. Estimation restarts LBFGS from a fixed grid to guard against local
   minima, but check `model.converged` and inspect the weight curve before interpreting
   ``\theta``.
4. **Beta endpoints.** The Beta grid is guarded away from 0 and 1 to avoid
   ``0^{\text{negative}}``, which forces the first and last weights to zero whenever both
   shape parameters exceed one. Use `:expalmon` when the endpoints must carry mass.
5. **`h` is a label, not a specification.** Setting `h=4` does not estimate a
   four-quarter-ahead regression; shift the inputs as described under Estimation.
6. **One indicator only.** `estimate_midas` handles a single high-frequency series. For
   several indicators, use [bridge equations](@ref nowcast_bridge_page) or a
   [mixed-frequency DFM](@ref nowcast_dfm_page).

---

## References

- Andreou, E., Ghysels, E., & Kourtellos, A. (2010). Regression Models with Mixed
  Sampling Frequencies. *Journal of Econometrics*, 158(2), 246--261.
  [10.1016/j.jeconom.2010.01.004](https://doi.org/10.1016/j.jeconom.2010.01.004)
- Clements, M. P., & Galvao, A. B. (2008). Macroeconomic Forecasting With
  Mixed-Frequency Data: Forecasting Output Growth in the United States.
  *Journal of Business & Economic Statistics*, 26(4), 546--554.
  [10.1198/073500108000000015](https://doi.org/10.1198/073500108000000015)
- Foroni, C., Marcellino, M., & Schumacher, C. (2015). Unrestricted Mixed Data Sampling
  (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials. *Journal of the Royal
  Statistical Society Series A*, 178(1), 57--82.
  [10.1111/rssa.12043](https://doi.org/10.1111/rssa.12043)
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). Predicting Volatility: Getting
  the Most Out of Return Data Sampled at Different Frequencies. *Journal of
  Econometrics*, 131(1--2), 59--95.
  [10.1016/j.jeconom.2005.01.004](https://doi.org/10.1016/j.jeconom.2005.01.004)
- Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS Regressions: Further Results and
  New Directions. *Econometric Reviews*, 26(1), 53--90.
  [10.1080/07474930600972467](https://doi.org/10.1080/07474930600972467)
- Mariano, R. S., & Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based
  on Monthly and Quarterly Series. *Journal of Applied Econometrics*, 18(4), 427--443.
  [10.1002/jae.695](https://doi.org/10.1002/jae.695)
