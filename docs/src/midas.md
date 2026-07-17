# MIDAS Regression

**MIDAS** (MIxed-DAta Sampling) regression relates a low-frequency target to many
high-frequency lags of an indicator through a parsimonious weighting function.
Where the [bridge equations](@ref nowcast_bridge_page) approach time-aggregates a monthly indicator to
the target frequency before regressing, MIDAS keeps every high-frequency
observation and lets a small parameter vector ``\theta`` shape their weights — so a
monthly indicator's within-quarter timing is exploited without discretionary
aggregation. Central banks use MIDAS and ADL-MIDAS for GDP nowcasting from monthly
and daily data.

This page covers:

- **Restricted MIDAS** with exponential-Almon, Beta, and polynomial-Almon weights
- **ADL-MIDAS**, adding autoregressive lags of the target
- **U-MIDAS**, the unrestricted lag polynomial estimated by OLS
- **Direct forecasting** with a nonlinear-least-squares prediction interval

## Quick Start

```@setup midas
using MacroEconometricModels
using Random, LinearAlgebra, Statistics

# Simulate a monthly -> quarterly DGP (m = 3): a quarterly target driven by six
# monthly lags through an exponential-Almon weight, plus one AR lag.
rng = MersenneTwister(20240716)
m, K, T_lf = 3, 6, 160
theta_true = [0.35, -0.06]
w_true = midas_weights(theta_true, K; kind=:expalmon)

x_hf = randn(rng, m * T_lf)                     # monthly indicator
s = zeros(T_lf)
for t in 1:T_lf
    hi = t * m
    hi - K + 1 >= 1 || continue
    s[t] = dot(x_hf[hi:-1:(hi - K + 1)], w_true)
end
y_lf = zeros(T_lf)                              # quarterly target
for t in 2:T_lf
    y_lf[t] = 0.5 + 2.0 * s[t] + 0.4 * y_lf[t-1] + 0.2 * randn(rng)
end
```

```@example midas
# Exponential-Almon ADL-MIDAS: quarterly y on 6 monthly lags of x + 1 AR lag
model = estimate_midas(y_lf, x_hf; m=3, K=6, weights=:expalmon, p_ar=1)
report(model)
```

```@example midas
# Realized weight curve w(θ̂), most-recent-first
round.(midas_weights(model); digits=4)
```

---

## Weight Functions

Restricted MIDAS aggregates the ``K`` high-frequency lags through a normalized
weight ``w_k(\theta)`` with ``\sum_k w_k = 1``. The estimated equation is

```math
y_t = \beta_0 + \beta_1 \sum_{k=1}^{K} w_k(\theta)\, x_{t,k} + \sum_{j=1}^{p} \rho_j\, y_{t-j} + u_t,
```

where

- ``y_t`` is the low-frequency target in period ``t``,
- ``x_{t,k}`` is the ``k``-th high-frequency lag within period ``t`` (most-recent-first),
- ``w_k(\theta)`` is the weight function, normalized to sum to 1,
- ``\beta_1`` is the aggregate loading and ``\rho_j`` the autoregressive coefficients.

The `weights` keyword selects the functional form:

| `weights` | Form | Parameters |
|---|---|---|
| `:expalmon` | ``w_k \propto \exp(\theta_1 k + \theta_2 k^2)`` | 2 |
| `:beta2` | ``w_k \propto x_k^{\theta_1-1}(1-x_k)^{\theta_2-1}`` | 2 |
| `:beta3` | Beta plus a nonnegative level constant ``\theta_3`` | 3 |
| `:almon` | polynomial ``w_k \propto \sum_{d} \theta_d k^{d}`` (then normalized) | `poly_degree`+1 |
| `:umidas` | unrestricted lag coefficients (OLS) | ``K`` |

The exponential-Almon weight reduces to **equal weights** ``1/K`` at ``\theta = 0``,
a useful sanity check. Evaluate any weight function directly with `midas_weights`:

```@example midas
# Exponential Almon with a mild decay
midas_weights([0.3, -0.05], 6; kind=:expalmon)
```

```@example midas
# Beta weight peaking early in the window
round.(midas_weights([2.0, 4.0], 6; kind=:beta2); digits=4)
```

---

## ADL-MIDAS and U-MIDAS

Setting `p_ar > 0` adds autoregressive lags of the target, giving the
**ADL-MIDAS** specification used for nowcasting persistent series. Setting
`weights=:umidas` drops the weight function entirely and estimates the ``K`` lag
coefficients by ordinary least squares (Foroni, Marcellino & Schumacher 2015),
which is competitive when the frequency ratio ``m`` is small.

```@example midas
# Unrestricted MIDAS: plain OLS on the six stacked monthly lags
umodel = estimate_midas(y_lf, x_hf; m=3, K=6, weights=:umidas, p_ar=1)
report(umodel)
```

The restricted exponential-Almon fit uses three free parameters
(``\beta_1``, ``\theta_1``, ``\theta_2``) where U-MIDAS spends six, so the
restricted model typically has the lower BIC when the weight shape is smooth.

---

## Forecasting

`forecast` produces a direct point forecast from a fresh high-frequency block
(most-recent-first) together with a Gaussian NLS prediction interval that
combines residual variance with parameter uncertainty.

```@example midas
# Six new monthly observations for the quarter to be nowcast
x_new = randn(MersenneTwister(1), 6)
fc = forecast(model, x_new)
report(fc)
```

The interval widens with both the residual standard error and the Gauss–Newton
sandwich term ``x_f^\top V x_f``, so forecasts made far from the in-sample
regressor range carry wider bands.

---

## Visualization

The weight curve and the actual-versus-fitted series are available through
`plot_result`:

```julia
plot_result(model; view=:weights)   # w_k versus high-frequency lag k
plot_result(model; view=:fit)       # actual vs fitted low-frequency target
```

---

## Complete Example

```@example midas
# End-to-end: estimate, inspect the weight shape, and nowcast one quarter ahead
model = estimate_midas(y_lf, x_hf; m=3, K=6, weights=:expalmon, p_ar=1)

# Weight concentration on the most recent months
w = midas_weights(model)
println("Weight on most recent month: ", round(w[1]; digits=3))
println("R-squared: ", round(model.r2; digits=3))

# Direct nowcast with 95% prediction interval
x_new = randn(MersenneTwister(2), 6)
fc = forecast(model, x_new; level=0.95)
report(fc)
```

---

## Common Pitfalls

1. **High-frequency ordering.** `X_hf` is chronological and the **last**
   observation aligns to the **last** target period. Within each period the lags
   enter most-recent-first, so `midas_weights(model)[1]` is the weight on the most
   recent high-frequency observation.
2. **Ragged edges are trimmed.** Periods without a full block of `K` lags (or
   without the requested `p_ar` autoregressive lags) are dropped automatically;
   the estimation sample is therefore shorter than `length(y_lf)`.
3. **Flat ridges in the profiled objective.** The exponential-Almon and Beta
   surfaces have flat regions. Estimation uses a multi-start LBFGS search to guard
   against local minima; check `model.converged` and inspect the weight curve.
4. **Beta endpoints.** The Beta grid is guarded away from 0 and 1 to avoid
   ``0^{\text{negative}}``; extreme ``\theta`` still produce very peaked weights.
5. **Single indicator only.** `estimate_midas` handles one high-frequency
   indicator; multi-indicator MIDAS is future work. For aggregate-then-regress
   nowcasting with several indicators, use [bridge equations](@ref nowcast_bridge_page).

---

## References

- Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS Regressions: Further
  Results and New Directions. *Econometric Reviews*, 26(1), 53–90.
  [10.1080/07474930600972467](https://doi.org/10.1080/07474930600972467)
- Andreou, E., Ghysels, E., & Kourtellos, A. (2010). Regression Models with Mixed
  Sampling Frequencies. *Journal of Econometrics*, 158(2), 246–261.
  [10.1016/j.jeconom.2010.01.004](https://doi.org/10.1016/j.jeconom.2010.01.004)
- Foroni, C., Marcellino, M., & Schumacher, C. (2015). Unrestricted Mixed Data
  Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials.
  *Journal of the Royal Statistical Society: Series A*, 178(1), 57–82.
  [10.1111/rssa.12043](https://doi.org/10.1111/rssa.12043)
```
