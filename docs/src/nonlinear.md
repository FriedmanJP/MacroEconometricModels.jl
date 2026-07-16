# [Nonlinear Time Series](@id nonlinear_page)

**MacroEconometricModels.jl** models regime-switching dynamics in the conditional mean through threshold autoregression. A threshold model is piecewise linear: the process follows one autoregression while a threshold variable sits below a boundary and a different autoregression once it crosses. This is the workhorse for asymmetric business-cycle and interest-rate dynamics, where expansions and contractions — or high- and low-rate states — obey distinct laws of motion.

- **Two-regime threshold least squares** — `estimate_threshold` fits ``y_t = X_t'\beta_1\,\mathbf{1}\{q_t \le \gamma\} + X_t'\beta_2\,\mathbf{1}\{q_t > \gamma\} + u_t`` with the threshold ``\gamma`` chosen by grid search over the order statistics of the threshold variable
- **SETAR** — `estimate_setar` is the self-exciting special case ``q_t = y_{t-d}``, ``X_t = [1, y_{t-1}, \dots, y_{t-p}]``, with optional joint selection of the delay ``d`` (Tong 1990)
- **Hansen (1996) linearity test** — `hansen_linearity_test` reports a heteroskedasticity-robust sup-LM (and sup-Wald) statistic with a fixed-regressor bootstrap p-value, the correct inference under the Davies nuisance-parameter problem
- **Hansen (2000) threshold confidence interval** — the reported interval inverts the likelihood-ratio statistic with the tabulated non-standard critical values ``c(.90)=5.94``, ``c(.95)=7.35``, ``c(.99)=10.59``
- **Bootstrap forecasting** — `forecast` iterates the fitted piecewise model forward, resampling residuals within regime, and returns a mean path with percentile bands

This is the entry point of the nonlinear module; smooth-transition (STAR) and Markov-switching models extend the same scaffold. All models return a [`ThresholdModel`](@ref) and integrate with `report`, `refs`, `forecast`, and `plot_result`.

```@setup nonlinear
using MacroEconometricModels, Random, Statistics
# A fixed-seed two-regime SETAR(2;1,1): the process switches on y_{t-1}.
Random.seed!(20240716)
n = 400
y = zeros(n)
for t in 2:n
    if y[t-1] <= 0.0
        y[t] = 0.6 + 0.5 * y[t-1] + 0.4 * randn()
    else
        y[t] = -0.6 - 0.4 * y[t-1] + 0.4 * randn()
    end
end
```

## Quick Start

**Recipe 1: Fit a SETAR model**

```@example nonlinear
# SETAR(2; 1, 1): AR(1) in each regime, threshold on y_{t-1}
m = estimate_setar(y, 1, 1)
report(m)
```

**Recipe 2: Test linearity before committing to a threshold**

```@example nonlinear
# Hansen (1996) sup-LM / sup-Wald with fixed-regressor bootstrap
X = hcat(ones(length(y) - 1), y[1:end-1])
lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=500)
println("sup-LM   = ", round(lt.sup_lm, digits=3),
        "  (bootstrap p = ", round(lt.pvalue_lm, digits=3), ")")
println("sup-Wald = ", round(lt.sup_wald, digits=3),
        "  (bootstrap p = ", round(lt.pvalue_wald, digits=3), ")")
```

**Recipe 3: Select the threshold delay automatically**

```@example nonlinear
# Search the delay d jointly with the threshold γ
m_auto = estimate_setar(y, 2, :auto)
println("selected delay d = ", m_auto.d)
```

**Recipe 4: Forecast**

```@example nonlinear
# Bootstrap-simulation forecast with 90% bands
f = forecast(m, 8; reps=1000)
report(f)
```

---

## Threshold Least Squares

The general two-regime threshold regression splits the sample by a threshold variable ``q_t``:

```math
y_t = X_t'\beta_1\,\mathbf{1}\{q_t \le \gamma\} + X_t'\beta_2\,\mathbf{1}\{q_t > \gamma\} + u_t.
```

For a fixed ``\gamma`` the model is linear in each regime, so the coefficients are concentrated out by regime OLS and the estimator reduces to a one-dimensional search for ``\gamma``. `estimate_threshold` minimises the concentrated sum of squared residuals

```math
S(\gamma) = \mathrm{SSR}_1(\gamma) + \mathrm{SSR}_2(\gamma)
```

over a grid of candidate thresholds.

!!! note "Grid over order statistics"
    The grid is the set of distinct sample values of ``q`` between the `trim` and ``1 - \text{trim}`` quantiles. Searching the order statistics of ``q`` — rather than an evenly spaced grid — visits every distinct sample split exactly once and never wastes a point on a boundary that reproduces its neighbour.

```@example nonlinear
# Generic threshold regression: supply y, the regressors X, and the threshold q
X = hcat(ones(length(y) - 1), y[1:end-1])
q = y[1:end-1]
mt = estimate_threshold(y[2:end], X, q; trim=0.15)
println("threshold γ̂ = ", round(mt.gamma, digits=3))
println("regime-1 obs = ", mt.n1, ", regime-2 obs = ", mt.n2)
```

The `trim` fraction guards each regime against rank deficiency: it must leave enough observations on each side for the regime OLS fits. A too-aggressive trim, or too few observations, raises an informative error.

---

## SETAR

The self-exciting threshold autoregression is the leading special case: the threshold variable is a lag of the series itself, ``q_t = y_{t-d}``, and the regressors are the model's own lags, ``X_t = [1, y_{t-1}, \dots, y_{t-p}]``. `estimate_setar(y, p, d)` builds this design and delegates to the threshold estimator.

```@example nonlinear
# SETAR(2; 2, 2) with a fixed delay d = 1
m2 = estimate_setar(y, 2, 1)
println("regime 1 (y[t-1] ≤ γ̂): ", round.(m2.beta1, digits=3))
println("regime 2 (y[t-1] > γ̂): ", round.(m2.beta2, digits=3))
```

Passing `d = :auto` (or a range such as `1:p`) selects the delay jointly with the threshold by minimising the pooled SSR over the ``(d, \gamma)`` grid. The selected delay is stored in `m.d`.

---

## Testing Linearity

Because the threshold ``\gamma`` is unidentified under the null of linearity (``\beta_1 = \beta_2``), the score test statistic has a nonstandard distribution — the Davies (1987) nuisance-parameter problem. `hansen_linearity_test` maximises the heteroskedasticity-robust LM statistic over the threshold grid,

```math
\sup_\gamma \; LM(\gamma), \qquad LM(\gamma) = S(\gamma)'\,V(\gamma)^{-1}\,S(\gamma),
```

where ``S(\gamma)`` is the score of the regime interaction evaluated at the linear-model residuals and ``V(\gamma)`` is its White heteroskedasticity-robust variance. The p-value comes from the **fixed-regressor bootstrap** of Hansen (1996): draw iid ``N(0,1)`` weights, form the simulated score process, recompute the supremum, and report the exceedance frequency.

```@example nonlinear
X = hcat(ones(length(y) - 1), y[1:end-1])
lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=500)
lt.pvalue_lm < 0.05   # reject linearity for this regime-switching series
```

!!! warning "Do not use χ² p-values"
    An asymptotic χ² approximation is invalid here because ``\gamma`` is not identified under the null. The fixed-regressor bootstrap is mandatory; `hansen_linearity_test` always reports bootstrap p-values.

---

## Threshold Confidence Interval

`estimate_threshold` and `estimate_setar` report a confidence interval for ``\gamma`` by inverting the likelihood-ratio statistic (Hansen 2000):

```math
LR(\gamma) = n\,\frac{S(\gamma) - S(\hat\gamma)}{S(\hat\gamma)},
\qquad
\text{CI} = \{\gamma : LR(\gamma) \le c(\alpha)\}.
```

The critical values are non-standard — they are quantiles of the distribution with CDF ``(1 - e^{-x/2})^2`` — and are tabulated exactly as

| Level ``\alpha`` | ``c(\alpha)`` |
|:----------------:|:-------------:|
| 0.90 | 5.94 |
| 0.95 | 7.35 |
| 0.99 | 10.59 |

```@example nonlinear
m = estimate_setar(y, 1, 1; ci_level=0.95)
println("γ̂  = ", round(m.gamma, digits=3))
println("95% CI = [", round(m.gamma_ci[1], digits=3), ", ",
        round(m.gamma_ci[2], digits=3), "]")
```

Set `het=true` to scale ``LR(\gamma)`` by an estimate of the heteroskedasticity ratio ``\eta^2`` at the threshold (Hansen 2000, §3.4); under homoskedasticity ``\eta^2 \approx 1`` and the interval is unchanged.

---

## Forecasting and Visualization

`forecast(m, h)` produces multi-step forecasts of a SETAR model by bootstrap simulation: it iterates the fitted piecewise model forward, drawing residuals from the regime realised at each step, and returns the mean path with percentile bands.

```@example nonlinear
f = forecast(m, 12; reps=1000, level=0.90)
f.forecast[1:3]   # first three horizons of the mean path
```

`plot_result` visualises the fit. The `:regimes` view colours the series by regime; the `:ssr` view plots the concentrated SSR profile ``S(\gamma)``, marking the minimiser ``\hat\gamma``.

```julia
plot_result(m; view=:regimes)   # series shaded by regime
plot_result(m; view=:ssr)       # SSR profile S(γ) with γ̂ annotated
```

---

## Complete Example

```@example nonlinear
# 1. Test linearity
X = hcat(ones(length(y) - 1), y[1:end-1])
lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=500)
reject = lt.pvalue_lm < 0.05

# 2. Fit the SETAR model with automatic delay selection
m = estimate_setar(y, 1, :auto)

# 3. Inspect the regimes, threshold, and its CI
report(m)
```

```@example nonlinear
# 4. Forecast forward
f = forecast(m, 8; reps=1000)
report(f)
```

---

## Common Pitfalls

1. **Trusting χ² p-values for linearity.** The threshold is unidentified under the null; only the fixed-regressor bootstrap p-values from `hansen_linearity_test` are valid.
2. **Too aggressive a `trim`.** Trimming must leave enough observations for two OLS fits per regime. If a regime would be rank-deficient the estimator raises an error — lower `trim` or supply more data.
3. **Forecasting a generic threshold model.** `forecast` is defined only for SETAR models (from `estimate_setar`); a generic `estimate_threshold` fit would require future exogenous ``X`` and ``q`` and raises an error.
4. **Reading the threshold CI as symmetric.** The Hansen (2000) interval inverts a nonstandard LR statistic and is generally asymmetric around ``\hat\gamma``; a very sharp threshold in a large sample can produce a degenerate interval at ``\hat\gamma``.
5. **Confusing the delay `d` with the AR order `p`.** In `estimate_setar(y, p, d)`, `p` is the number of autoregressive lags per regime and `d` is the delay of the threshold variable ``y_{t-d}``.

---

## References

- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is present only under the alternative. *Biometrika* 74(1), 33–43.
- Hansen, B. E. (1996). Inference when a nuisance parameter is not identified under the null hypothesis. *Econometrica* 64(2), 413–430. [doi:10.2307/2171789](https://doi.org/10.2307/2171789)
- Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68(3), 575–603. [doi:10.1111/1468-0262.00124](https://doi.org/10.1111/1468-0262.00124)
- Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach.* Oxford University Press. ISBN 978-0-19-852300-6.
