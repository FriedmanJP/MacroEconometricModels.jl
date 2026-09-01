# [Nonlinear Time Series](@id nonlinear_page)

**MacroEconometricModels.jl** models regime-switching dynamics in the conditional mean through threshold autoregression. A threshold model is piecewise linear: the process follows one autoregression while a threshold variable sits below a boundary and a different autoregression once it crosses. This is the workhorse for asymmetric business-cycle and interest-rate dynamics, where expansions and contractions — or high- and low-rate states — obey distinct laws of motion.

- **Two-regime threshold least squares** — `estimate_threshold` fits ``y_t = X_t'\beta_1\,\mathbf{1}\{q_t \le \gamma\} + X_t'\beta_2\,\mathbf{1}\{q_t > \gamma\} + u_t`` with the threshold ``\gamma`` chosen by grid search over the order statistics of the threshold variable
- **SETAR** — `estimate_setar` is the self-exciting special case ``q_t = y_{t-d}``, ``X_t = [1, y_{t-1}, \dots, y_{t-p}]``, with optional joint selection of the delay ``d`` (Tong 1990)
- **Hansen (1996) linearity test** — `hansen_linearity_test` reports a heteroskedasticity-robust sup-LM (and sup-Wald) statistic with a fixed-regressor bootstrap p-value, the correct inference under the Davies nuisance-parameter problem
- **Hansen (2000) threshold confidence interval** — the reported interval inverts the likelihood-ratio statistic with the tabulated non-standard critical values ``c(.90)=5.94``, ``c(.95)=7.35``, ``c(.99)=10.59``
- **Bootstrap forecasting** — `forecast` iterates the fitted piecewise model forward, resampling residuals within regime, and returns a mean path with percentile bands
- **Smooth-transition autoregression (STAR)** — `estimate_star` replaces the abrupt indicator with a continuous logistic (LSTR1/LSTR2) or exponential (ESTR) transition, `star_linearity_test` runs the Luukkonen–Saikkonen–Teräsvirta LM3 test, and `type=:auto` selects the transition shape by Teräsvirta's (1994) sequential procedure
- **Markov-switching regression and MS-AR** — `estimate_ms` and `estimate_ms_ar` make the regime a latent Markov chain rather than a function of an observed variable, filtered by Hamilton (1989) and smoothed by Kim (1994)

Threshold and SETAR models return a [`ThresholdModel`](@ref); the smooth-transition estimator returns a [`STARModel`](@ref); the Markov-switching estimators return an [`MSRegModel`](@ref). All three integrate with `report`, `refs`, `forecast`, and `plot_result`. For models that switch the shock *variance* rather than the conditional mean, see [Statistical Identification](@ref nongaussian_page); for locally-weighted conditional means that never commit to a regime at all, see [Nonparametric Regression](@ref nonparametric_page).

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
(sup_lm = round(lt.sup_lm; digits=3), p_lm = lt.pvalue_lm,
 sup_wald = round(lt.sup_wald; digits=3), p_wald = lt.pvalue_wald)
```

**Recipe 3: Select the threshold delay automatically**

```@example nonlinear
# Search the delay d jointly with the threshold γ
m_auto = estimate_setar(y, 2, :auto)
(selected_delay = m_auto.d, threshold = round(m_auto.gamma; digits=3))
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
report(mt)
```

The estimator splits the 399 effective observations into 226 below the threshold and 173 above it, at ``\hat\gamma = 0.000`` — the true switch point of the simulated process. The two regimes are qualitatively different rather than merely shifted: below the threshold the process reverts toward a positive level (intercept 0.636, slope 0.513), above it toward a negative one (intercept ``-0.551``, slope ``-0.504``). Every coefficient clears the 1% level, and the attached Hansen sup-LM of 107.2 rejects linearity with a bootstrap p-value below 0.002. Because the design here is generic, the coefficient labels default to `x1`, `x2`; `estimate_setar` supplies `const`, `y[t-1]`, … instead.

The `trim` fraction guards each regime against rank deficiency: it must leave enough observations on each side for the regime OLS fits. A too-aggressive trim, or too few observations, raises an informative error.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `trim` | `Real` | `0.15` | Fraction of extreme ``q`` values excluded from the ``\gamma`` grid |
| `linearity` | `Bool` | `true` | Run `hansen_linearity_test` and attach the result |
| `reps` | `Int` | `1000` | Bootstrap replications for the linearity test |
| `ci_level` | `Real` | `0.95` | Confidence level for the threshold CI (`0.90`, `0.95`, or `0.99`) |
| `het` | `Bool` | `false` | Heteroskedasticity-correct the threshold CI |
| `rng` | `AbstractRNG` | `default_rng()` | Generator for the linearity bootstrap |

`estimate_threshold` and `estimate_setar` both return a [`ThresholdModel`](@ref):

| Field | Type | Description |
|---|---|---|
| `gamma` | `T` | Estimated threshold ``\hat\gamma`` |
| `gamma_ci` | `Tuple{T,T}` | Hansen (2000) LR-inversion interval for ``\gamma`` |
| `beta1`, `beta2` | `Vector{T}` | Regime coefficients (``q \le \hat\gamma`` / ``q > \hat\gamma``) |
| `se1`, `se2` | `Vector{T}` | Classical per-regime standard errors |
| `regime` | `Vector{Bool}` | Regime-1 indicator ``\mathbf{1}\{q_t \le \hat\gamma\}`` |
| `n1`, `n2` | `Int` | Per-regime observation counts |
| `ssr1`, `ssr2`, `ssr` | `T` | Regime and pooled sums of squared residuals |
| `sigma2` | `T` | Pooled residual variance ``S(\hat\gamma)/n`` |
| `p`, `d` | `Int` | SETAR order and delay (`0` for a generic threshold fit) |
| `is_setar` | `Bool` | Whether the fit came from `estimate_setar` |
| `aic`, `bic` | `T` | Information criteria (``2k+1`` parameters) |
| `linearity` | `Union{Nothing,HansenLinearityTest{T}}` | Attached linearity test |

---

## SETAR

The self-exciting threshold autoregression is the leading special case: the threshold variable is a lag of the series itself, ``q_t = y_{t-d}``, and the regressors are the model's own lags, ``X_t = [1, y_{t-1}, \dots, y_{t-p}]``. `estimate_setar(y, p, d)` builds this design and delegates to the threshold estimator.

```@example nonlinear
# SETAR(2; 2, 2) with a fixed delay d = 1
m2 = estimate_setar(y, 2, 1)
(gamma = round(m2.gamma; digits=3),
 regime1 = round.(m2.beta1; digits=3),      # [const, y[t-1], y[t-2]] for y[t-1] ≤ γ̂
 regime2 = round.(m2.beta2; digits=3))      # same block for y[t-1] > γ̂
```

Adding a second lag leaves the first-regime picture almost unchanged — intercept 0.635 and first-lag slope 0.514, against 0.636 and 0.513 in the SETAR(2;1,1) fit — and the second lag enters at 0.006 in regime 1 and ``-0.078`` in regime 2, both negligible because the data-generating process has no second lag. The estimated threshold moves from 0.000 to ``-0.002``: with `p = 2` the effective sample starts one period later, so the grid of candidate order statistics is not identical.

Passing `d = :auto` (or a range such as `1:p`) selects the delay jointly with the threshold by minimising the pooled SSR over the ``(d, \gamma)`` grid, on a common effective sample ``t > \max(p, d_{\max})`` so the SSRs are comparable. The selected delay is stored in `m.d`.

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
(gamma_at_sup = round(lt.gamma_sup; digits=3),   # threshold attaining the supremum
 grid_points = lt.n_grid, reps = lt.reps,
 bootstrap_p_lm = lt.pvalue_lm, bootstrap_p_wald = lt.pvalue_wald)
```

The supremum is attained at ``\gamma = 0.000``, searched over 282 candidate order statistics after 15% trimming. Not one of the 500 fixed-regressor bootstrap draws produced a sup-LM or sup-Wald as large as the observed 107.2 and 392.3, so both p-values are 0.000 — linearity is rejected as decisively as 500 replications can express. A bootstrap p-value of zero means "below ``1/500``", not "exactly zero"; raise `reps` if that distinction matters for a marginal case.

The returned [`HansenLinearityTest`](@ref) carries `sup_lm`, `sup_wald`, their bootstrap p-values, `gamma_sup`, `reps`, `trim`, and `n_grid`. When `estimate_threshold` runs with `linearity=true` (the default) the same object is attached to the fitted model's `linearity` field and printed in the `report` footer.

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
short = estimate_setar(y[1:80], 1, 1; ci_level=0.95, linearity=false)
(full_sample = (gamma = round(m.gamma; digits=3), ci = round.(m.gamma_ci; digits=3)),
 short_sample = (gamma = round(short.gamma; digits=3), ci = round.(short.gamma_ci; digits=3)))
```

On the full sample the interval collapses to the single point ``\hat\gamma = 0.000``. That is not a failure of the routine: with 399 observations and a genuinely discontinuous switch, every neighbouring candidate raises the SSR enough that ``LR(\gamma)`` clears the critical value 7.35 immediately. Re-fitting on the first 80 observations makes the threshold much less sharply identified — ``\hat\gamma = 0.089`` with a 95% interval of ``[0.000, 0.143]``, visibly asymmetric around the point estimate because ``LR(\gamma)`` is a step function over the order statistics rather than a smooth quadratic.

Set `het=true` to scale ``LR(\gamma)`` by an estimate of the heteroskedasticity ratio ``\eta^2`` at the threshold (Hansen 2000, §3.4); under homoskedasticity ``\eta^2 \approx 1`` and the interval is unchanged.

---

## Forecasting and Visualization

`forecast(m, h)` produces multi-step forecasts of a SETAR model by bootstrap simulation: it iterates the fitted piecewise model forward, drawing residuals from the regime realised at each step, and returns the mean path with percentile bands.

```@example nonlinear
f = forecast(m, 12; reps=1000, level=0.90)
(mean_path = round.(f.forecast[1:4]; digits=3),
 lower_90 = round.(f.ci_lower[1:4]; digits=3),
 upper_90 = round.(f.ci_upper[1:4]; digits=3))
```

The one-step forecast of ``-1.163`` sits deep in regime 2, because the last observed value is above the threshold and the regime-2 intercept is negative. By horizon 2 the mean has jumped back to ``0.019``: a large negative value at ``t+1`` puts the process below the threshold again, so the regime-1 law of motion takes over. From horizon 3 the mean settles near ``-0.15``, the unconditional mean of the two-regime mixture, while the 90% bands stay wide — roughly ``[-1.35, 0.92]`` at horizon 4 — because the regime itself remains uncertain at every step.

The returned [`ThresholdForecast`](@ref) carries `forecast`, `ci_lower`, `ci_upper`, `se`, `horizon`, `conf_level`, and `reps`. `forecast` is defined only for SETAR fits: a generic threshold model would need future paths for ``X`` and ``q``, which the estimator has no way to supply.

`plot_result` visualises the fit. The `:regimes` view colours the series by regime; the `:ssr` view plots the concentrated SSR profile ``S(\gamma)``, marking the minimiser ``\hat\gamma``; `:diagnostics` draws the residual panel.

```julia
plot_result(m; view=:regimes)       # series shaded by regime
plot_result(m; view=:ssr)           # SSR profile S(γ) with γ̂ annotated
plot_result(m; view=:diagnostics)   # residual diagnostics
```

```@raw html
<iframe src="../assets/plots/nonlinear_regimes.html" width="100%" height="420" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The two colours alternate at almost every date rather than clustering into long spells: with ``q_t = y_{t-1}`` and ``\hat\gamma = 0``, regime membership is decided by the sign of the previous observation, and the fitted laws of motion push the process back across zero at once. That is the signature of a self-exciting threshold in a mean-reverting series, and it is what separates the SETAR reading from a Markov-switching one, where the regime persists because the chain does.

---

## Smooth-Transition Autoregression (STAR)

The threshold model switches regime abruptly at ``\gamma``. A smooth-transition autoregression (STAR) replaces the indicator ``\mathbf{1}\{s_t > c\}`` with a continuous transition function ``G(s_t; \gamma, c) \in [0, 1]``, so the process moves *gradually* between two autoregressions as the transition variable ``s_t`` crosses the location ``c``:

```math
y_t = \phi_1'z_t\,(1 - G(s_t;\gamma,c)) + \phi_2'z_t\,G(s_t;\gamma,c) + u_t,
\qquad z_t = [1, y_{t-1}, \dots, y_{t-p}]'.
```

This is the standard model for gradual business-cycle asymmetry — expansions and contractions blend into one another rather than snapping at a boundary. The transition ``G`` takes one of three shapes:

| Type | ``G(s_t;\gamma,c)`` | Shape |
|:-----|:--------------------|:------|
| `:lstr1` | ``1/(1 + e^{-(\gamma/\hat\sigma_s)(s_t - c)})`` | Logistic, one location — monotone asymmetry |
| `:lstr2` | ``1/(1 + e^{-(\gamma/\hat\sigma_s^2)(s_t - c_1)(s_t - c_2)})`` | Logistic, two locations — outer/inner regimes |
| `:estr`  | ``1 - e^{-(\gamma/\hat\sigma_s^2)(s_t - c)^2}`` | Exponential — symmetric about ``c`` |

!!! note "The ``1/\hat\sigma_s`` slope scaling"
    The slope ``\gamma`` is divided by the sample standard deviation of ``s`` (squared for the quadratic transitions). This makes ``\gamma`` dimension-free and comparable across series — and it is not optional: without it the optimiser stalls on the flat region of ``G`` where the objective is nearly constant in ``\gamma`` (Teräsvirta 1994). The reported ``\hat\gamma`` is the scaled slope.

`estimate_star` fits the model by nonlinear least squares. Because the STAR objective is multimodal, starting values come from a 2-D grid over ``(\gamma, c)`` — ``\gamma`` log-spaced, ``c`` on the sample quantiles of ``s`` — with the linear coefficients ``(\phi_1, \phi_2)`` concentrated out by OLS at each node; the best node is refined with L-BFGS and a ForwardDiff gradient. Standard errors are the Gauss–Newton delta-method SEs.

```@example nonlinear
# LSTR1 smooth-transition AR(1), transition on y_{t-1}
ms = estimate_star(y, 1; d=1, type=:lstr1)
report(ms)
```

The estimated slope ``\hat\gamma = 203.8`` carries a standard error of 258.5 and a p-value of 0.43. Read that as an identification result, not an insignificance result: as ``\gamma \to \infty`` the logistic transition converges to the SETAR indicator, so the objective flattens once ``\gamma`` is large and the data cannot pin down *how* large. What the fit does identify is the regime coefficients, and they reproduce the SETAR estimates almost exactly (0.644 and 0.520 against 0.636 and 0.513; ``-0.539`` and ``-0.522`` against ``-0.551`` and ``-0.504``), with the location ``\hat c = -0.001`` recovering the true switch point. `report` prints `Converged  false` for the same reason — L-BFGS cannot meet its gradient tolerance on a flat ridge — and the fitted model is still the right one.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `s` | `AbstractVector` | `nothing` | External transition variable aligned with `y`; `nothing` uses ``y_{t-d}`` |
| `d` | `Int` | `1` | Transition delay for the self-exciting case |
| `type` | `Symbol` | `:auto` | `:lstr1`, `:lstr2`, `:estr`, or `:auto` for Teräsvirta selection |
| `n_gamma` | `Int` | `15` | Log-spaced ``\gamma`` nodes in the start-value grid |
| `n_c` | `Int` | `15` | Quantile nodes for the location ``c`` |

`estimate_star` returns a [`STARModel`](@ref):

| Field | Type | Description |
|---|---|---|
| `phi1`, `phi2` | `Vector{T}` | Regime coefficients weighted by ``1-G`` and ``G`` |
| `se_phi1`, `se_phi2` | `Vector{T}` | Gauss–Newton delta-method standard errors |
| `gamma`, `c` | `T`, `Vector{T}` | Scaled transition slope and location(s) |
| `se_gamma`, `se_c` | `T`, `Vector{T}` | Standard errors of the transition parameters |
| `G` | `Vector{T}` | Fitted transition weights ``G(s_t;\hat\gamma,\hat c)`` |
| `trans_type` | `Symbol` | Transition shape actually fitted |
| `sigma_s` | `T` | Scaling ``\hat\sigma_s`` applied to ``\gamma`` |
| `lm3_stat`, `lm3_pvalue` | `T` | LM3 linearity statistic (``\chi^2``) and p-value |
| `lm3_fstat`, `lm3_fpvalue` | `T` | F-form of the LM3 test and its p-value |
| `sel_pvalues` | `Union{Nothing,NTuple{3,T}}` | Teräsvirta ``(H_{04}, H_{03}, H_{02})`` p-values under `type=:auto` |
| `converged` | `Bool` | L-BFGS convergence flag |

### Testing linearity and selecting the transition

Under the null of linearity (``\phi_1 = \phi_2``) the transition parameters ``\gamma`` and ``c`` are unidentified. `star_linearity_test` sidesteps this with the Luukkonen–Saikkonen–Teräsvirta LM3 test: it regresses the linear-AR residuals on ``z_t`` augmented with ``\tilde z_t s_t``, ``\tilde z_t s_t^2``, ``\tilde z_t s_t^3`` — the third-order Taylor expansion of ``G`` around ``\gamma = 0`` — and forms the ``n R^2 \sim \chi^2(3p)`` statistic (plus an F-form with better small-sample size).

```@example nonlinear
lt3 = star_linearity_test(y, 1; d=1)
(df = lt3.df,
 lm3_chi2 = round(lt3.stat; digits=2), p_chi2 = round(lt3.pvalue; sigdigits=2),
 lm3_F = round(lt3.fstat; digits=2), p_F = round(lt3.fpvalue; sigdigits=2))
```

The LM3 statistic is 111.5 on ``3p = 3`` degrees of freedom (``p \approx 5\times10^{-24}``), and the F-form gives 50.96 with an even smaller p-value. Linearity is rejected overwhelmingly, which is what the third-order Taylor expansion is built to detect: a sharp threshold is a limiting smooth transition, so the test has power against SETAR alternatives as well as genuinely gradual ones.

Passing `type=:auto` runs Teräsvirta's (1994) sequential F-test on the same auxiliary regression to choose the transition shape, storing the three hypothesis p-values in `sel_pvalues`. The rule is the one in van Dijk, Teräsvirta & Franses (2002): the symmetric ESTR transition is selected when ``H_{03}`` is the most strongly rejected of the three, otherwise LSTR1.

```@example nonlinear
star_auto = estimate_star(y, 1; type=:auto)
(transition = star_auto.trans_type,
 terasvirta_p = round.(star_auto.sel_pvalues; sigdigits=2))   # (H₀₄, H₀₃, H₀₂)
```

The sequential test returns ``p_4 = 5.4\times10^{-11}``, ``p_3 = 1.5\times10^{-4}``, and ``p_2 = 2.2\times10^{-17}``. Since ``p_3`` is the *largest* of the three rather than the smallest, the symmetric ESTR shape is not selected and the procedure returns LSTR1 — correctly, because the simulated process switches monotonically at a single point rather than symmetrically around one.

A fitted self-exciting STAR model forecasts by the same bootstrap-simulation route as SETAR, returning a [`STARForecast`](@ref) with the same fields:

```@example nonlinear
fs = forecast(star_auto, 8; reps=1000, level=0.90)
round.(fs.forecast; digits=3)
```

The path starts at ``-1.164``, rebounds to 0.038 at horizon 2, and settles near ``-0.18``, tracking the SETAR forecast closely — unsurprising, since the fitted transition is effectively an indicator. Simulation is unavoidable here: the smooth transition makes the ``h``-step conditional mean a nonlinear function of the entire simulated history, with no closed form to evaluate.

The `:transition` view plots the fitted ``G(s_t;\hat\gamma,\hat c)`` against the transition variable, showing how sharply the process moves between regimes.

```julia
plot_result(ms; view=:transition)     # G(s) against s
plot_result(ms; view=:weights)        # G over time
plot_result(ms; view=:diagnostics)    # residual diagnostics
```

```@raw html
<iframe src="../assets/plots/nonlinear_star_transition.html" width="100%" height="420" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

At ``\hat\gamma = 203.8`` the curve is a step: ``G`` sits at 0 for every ``s_t`` below ``\hat c = -0.001`` and at 1 above it, with 5 of the 399 observations landing anywhere on the ramp between. This is the picture behind the flat objective — the plot is the diagnostic that tells a genuinely smooth transition apart from an indicator that the logistic has merely reparameterized.

!!! note "Distinct from smooth-transition heteroskedasticity"
    STAR models a smooth transition in the conditional *mean*. The smooth-transition SVAR used for statistical identification in [Statistical Identification](@ref nongaussian_page) drives the shock *covariance* with the same logistic function and leaves the autoregression alone. The two share the functional form and nothing else.

---

## Markov-Switching Regression and MS-AR

Where threshold and STAR models make the regime a deterministic function of an observed transition variable, a **Markov-switching** model treats the regime ``s_t`` as a *latent* ``K``-state Markov chain with transition matrix ``P`` (Hamilton 1989). The likelihood integrates over the unobserved regime path with the scaled Hamilton forward filter; the Kim (1994) backward smoother recovers ``\Pr(s_t=k\mid\mathcal{F}_T)``; and parameters are estimated by EM (Baum–Welch) with an `Optim` maximum-likelihood polish and delta-method standard errors.

Two entry points cover the common cases:

- [`estimate_ms`](@ref) — a switching *regression* ``y_t = x_t'\beta_{s_t} + \varepsilon_t``, ``\varepsilon_t\sim N(0,\sigma^2_{s_t})``, where every coefficient (and optionally the variance) switches with the regime.
- [`estimate_ms_ar`](@ref) — the Hamilton (1989) mean-switching autoregression

```math
(y_t - \mu_{s_t}) = \sum_{j=1}^{p} \phi_j\,(y_{t-j} - \mu_{s_{t-j}}) + \varepsilon_t,
```

where only the level ``\mu`` switches; the AR coefficients ``\phi`` are common across regimes. Because the density depends on the regime *path* ``(s_t,\dots,s_{t-p})``, the filter runs on the ``K^{p+1}`` expanded state space and reports the ``s_t`` marginals.

Regimes are labelled deterministically in order of increasing conditional mean, so regime 1 is always the lowest-mean state — reproducible across runs.

The canonical illustration is Hamilton's (1989) MS(2)-AR(4) model of US real GNP growth, shipped as `load_example(:gnp_hamilton)`:

```@example nonlinear
ts = load_example(:gnp_hamilton)          # 135 quarters of 100×Δlog US real GNP, 1951Q2–1984Q4
g  = vec(ts.data)
mh = estimate_ms_ar(g, 4)                 # 2-regime mean-switching AR(4)
report(mh)
```

Regime 1 is the low-growth recession state with ``\hat\mu = -0.36``, regime 2 the expansion with ``\hat\mu = 1.16``; the staying probabilities are ``\hat p_{11} = 0.755`` and ``\hat p_{22} = 0.904``. These are the classic Hamilton (1989, Table I) estimates. Among the common AR coefficients only ``\phi_3 = -0.247`` and ``\phi_4 = -0.213`` are individually significant, and the two reported regime variances are identical at 0.591 because `estimate_ms_ar` defaults to `switching_variance=false`, the Hamilton (1989) specification.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `k_regimes` | `Int` | `2` | Number of latent regimes ``K`` |
| `switching_variance` | `Bool` | `false` (MS-AR) / `true` (regression) | Give each regime its own ``\sigma^2`` |
| `max_iter` | `Int` | `1000` (MS-AR) / `500` (regression) | Optimiser / EM iteration cap |
| `yname` | `String` | `"y"` | Dependent-variable label (MS-AR only) |
| `xnames` | `Vector{String}` | `nothing` | Regressor labels (regression only) |
| `tol` | `Real` | `1e-8` | EM log-likelihood tolerance (regression only) |

The `smoothed_prob` field holds the inferred recession probabilities ``\Pr(s_t=1\mid\mathcal{F}_T)``, and `ergodic`/`expected_durations` summarise the estimated chain:

```@example nonlinear
(max_recession_prob = round(maximum(mh.smoothed_prob[:, 1]); digits=3),
 ergodic_recession_share = round(mh.ergodic[1]; digits=3),
 expected_recession_quarters = round(mh.expected_durations[1]; digits=1),
 expected_expansion_quarters = round(mh.expected_durations[2]; digits=1))
```

The smoothed probability of the recession state reaches 0.999 at its peak — the filter identifies specific quarters as recessions with near certainty rather than hedging across the sample. The ergodic distribution puts 28.1% of the mass on the recession state, and the implied durations are 4.1 quarters for a recession against 10.4 for an expansion. The estimated chain therefore reproduces the standard business-cycle asymmetry, in which contractions are short and expansions long.

For a general switching regression, pass the design matrix directly. With `switching_variance=true` (the default for `estimate_ms`) each regime carries its own error variance:

```@example nonlinear
mr = estimate_ms(g; k_regimes=2, switching_variance=true)   # intercept-only: switching mean+var
(regime_means = round.(mr.mu; digits=3),
 regime_variances = round.(mr.sigma2; digits=3),
 staying_probs = round.([mr.P[1, 1], mr.P[2, 2]]; digits=3))
```

The intercept-only switching regression finds means of ``-0.224`` and 1.177 with staying probabilities of 0.753 and 0.892 — close to the MS-AR estimates, because most of what separates the two states is the level. What changes is the variance: the recession state carries 0.942 against 0.620 in the expansion, so downturns are the more volatile as well as the lower-mean regime, a feature the fixed-variance MS-AR cannot express.

The `:probabilities` view plots the Kim-smoothed regime probabilities as a stacked area (each layer a regime, summing to 1 at every date):

```julia
plot_result(mh; view=:probabilities)   # smoothed regime timeline
plot_result(mh; view=:filtered)        # filtered probabilities
plot_result(mh; view=:diagnostics)     # residual diagnostics
```

```@raw html
<iframe src="../assets/plots/nonlinear_ms_probabilities.html" width="100%" height="420" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The low-mean state takes more than half the smoothed probability in 36 of the 131 estimation quarters, arriving in seven separate spells — the same count as the NBER recessions dated between 1953 and 1982, and the reason Hamilton's model is read as a statistical business-cycle chronology rather than a two-mean mixture. The layers stack to 1 at every date, so the width of the lower band is the probability the quarter belongs to the recession state.

Both entry points return an [`MSRegModel`](@ref):

| Field | Type | Description |
|---|---|---|
| `model_type` | `Symbol` | `:regression` or `:ms_ar` |
| `mu` | `Vector{T}` | Regime conditional means, increasing |
| `coefs`, `se_coefs` | `Matrix{T}` | Per-regime coefficients (``k_x \times K``) and standard errors |
| `ar`, `se_ar` | `Vector{T}` | Common AR coefficients ``\phi`` (empty for `:regression`) |
| `sigma2`, `se_sigma2` | `Vector{T}` | Regime variances (equal entries when the variance does not switch) |
| `P` | `Matrix{T}` | ``K \times K`` transition matrix, rows summing to 1 |
| `ergodic` | `Vector{T}` | Stationary distribution of ``P`` |
| `expected_durations` | `Vector{T}` | ``1/(1 - P_{kk})`` |
| `filtered_prob`, `smoothed_prob` | `Matrix{T}` | ``\Pr(s_t=k\mid\mathcal F_t)`` and ``\Pr(s_t=k\mid\mathcal F_T)`` |
| `fitted`, `fitted_filtered` | `Vector{T}` | Smoothed- and filtered-weighted conditional means |
| `loglik`, `aic`, `bic` | `T` | Maximised log-likelihood and information criteria |
| `converged`, `iterations` | `Bool`, `Int` | Convergence flag and iteration count |

!!! note "Distinct from variance-regime identification"
    `estimate_ms`/`estimate_ms_ar` switch the conditional *mean*. The Markov-switching *variance* regimes used for SVAR identification in [Statistical Identification](@ref nongaussian_page), and the Hamilton (2018) detrending [`hamilton_filter`](@ref), are separate code paths. All three run a Hamilton–Kim filter but model different objects.

### Fitted Values and Forecasts

The fitted value of a Markov-switching model is the regime-probability-weighted conditional mean

```math
\hat y_t = \sum_k \Pr(s_t = k \mid \mathcal I)\, E[y_t \mid s_t = k, \mathcal F_{t-1}],
```

where

- ``\Pr(s_t = k \mid \mathcal I)`` is the regime probability under information set ``\mathcal I``,
- ``E[y_t \mid s_t = k, \mathcal F_{t-1}]`` is the regime-``k`` conditional mean.

[`fitted`](@ref) weights by the **smoothed** probabilities (``\mathcal I = \mathcal F_T``), so `y - fitted(m)` is exactly [`residuals`](@ref). [`predict`](@ref) with `probs=:filtered` weights by ``\mathcal F_t`` instead — the real-time analogue wanted for pseudo-out-of-sample evaluation. Both are computed exactly, over the ``K^{p+1}`` expanded regime state for MS-AR.

```@example nonlinear
(identity_holds = maximum(abs, mh.y .- fitted(mh) .- residuals(mh)),
 filtered_differs = round(maximum(abs, predict(mh; probs=:filtered) .- fitted(mh)); digits=4))
```

The identity ``y - \text{fitted} = \text{residuals}`` holds to 0.0 — exactly, not approximately, because both are built from the same smoothed weights. The filtered mean differs from the smoothed one by as much as 0.954 at some date, roughly the gap between the two regime means. That is precisely why the distinction matters: the filtered series knows only the past and can be badly wrong about a turning point the full sample later resolves.

!!! warning "The filtered mean does not reproduce the residuals"
    `y - predict(m; probs=:filtered)` is **not** `residuals(m)`. The published residuals are the smoothed-weighted ones. Use the filtered series for forecast evaluation, the smoothed one for in-sample diagnostics.

[`forecast`](@ref) propagates the regime probabilities through the transition matrix, ``\xi_{t+h|t} = (P')^h \xi_{t|t}``, and mixes the regime-specific means with those weights. Which signature applies depends on the model:

- **MS-AR** — `forecast(m, h)`. Because ``z_t = y_t - \mu_{s_t}`` follows a *regime-free* AR(``p``), the exact ``h``-step mean needs no expansion of the ``K^{p+1}`` state space and no simulation.
- **Switching regression** — `forecast(m, X_new)`, since ``y_t = x_t'\beta_{s_t} + \varepsilon_t`` cannot be projected without future regressors.

```@example nonlinear
fh = forecast(mh, 8)
(mean_path = round.(fh.forecast; digits=3),
 recession_prob = round.(fh.regime_prob[:, 1]; digits=3))
```

The last observation sits in the expansion state, so the mean path opens at 0.617, peaks at 1.201 by horizon 3 as the AR dynamics play out, then decays toward the ergodic average. The recession probability climbs monotonically from 0.144 to 0.274, converging on the ergodic 0.281 — the chain forgets the current state geometrically, at a rate set by the second eigenvalue of ``P``.

The `forecast` field is the **exact** analytic mean; the `level` bands and `se` come from simulating the Gaussian-mixture predictive density, because a Markov mixture has no convenient closed-form quantile. Two consequences worth knowing: the mean path is reproducible across RNG seeds while the bands are not, and as ``h \to \infty`` the mean converges to the ergodic average ``\sum_k \pi_k \mu_k``.

```@example nonlinear
f_long = forecast(mh, 400; reps=50)
(long_horizon = round(f_long.forecast[end]; digits=6),
 ergodic_mean = round(sum(mh.ergodic .* mh.mu); digits=6))
```

At ``h = 400`` the forecast equals ``\sum_k \pi_k \mu_k`` to all six printed digits, 0.735627 either way. This is a useful invariant to check on a fitted model: any drift between the two would mean the transition matrix and the regime means are mutually inconsistent.

---

## Complete Example

A complete threshold workflow tests linearity first, fits the model only if the test rejects, and reads the threshold together with its confidence interval before forecasting.

```@example nonlinear
# 1. Test linearity — the fixed-regressor bootstrap is the valid p-value here
X = hcat(ones(length(y) - 1), y[1:end-1])
lt = hansen_linearity_test(y[2:end], X, y[1:end-1]; reps=500)

# 2. Fit the SETAR model with automatic delay selection
m = estimate_setar(y, 1, :auto)

# 3. Inspect the regimes, threshold, and its CI
report(m)
```

```@example nonlinear
# 4. Forecast forward, then compare the two regimes' persistence
f = forecast(m, 8; reps=1000)
(reject_linearity = lt.pvalue_lm < 0.05,
 persistence_regime1 = round(m.beta1[2]; digits=3),
 persistence_regime2 = round(m.beta2[2]; digits=3),
 forecast_h1 = round(f.forecast[1]; digits=3))
```

Linearity is rejected, so a threshold model is warranted. The two regimes carry opposite-signed persistence — 0.513 below the threshold against ``-0.504`` above it — and that is the substantive finding: the process is positively autocorrelated in the low state and oscillatory in the high one, so a single linear AR would average the two into something that describes neither. The one-step forecast of ``-1.186`` reflects the last observation lying above the threshold.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example nonlinear
path = joinpath(mktempdir(), "threshold.jld2")
save_model(m, path)
m2 = load_model(path)
typeof(m2)
```

---

## Common Pitfalls

1. **Trusting χ² p-values for linearity.** The threshold is unidentified under the null; only the fixed-regressor bootstrap p-values from `hansen_linearity_test` are valid.
2. **Too aggressive a `trim`.** Trimming must leave enough observations for two OLS fits per regime. If a regime would be rank-deficient the estimator raises an error — lower `trim` or supply more data.
3. **Forecasting a generic threshold model.** `forecast` is defined only for SETAR models (from `estimate_setar`); a generic `estimate_threshold` fit would require future exogenous ``X`` and ``q`` and raises an error. The same restriction applies to STAR: only self-exciting fits (``s_t = y_{t-d}``) can be projected.
4. **Reading the threshold CI as symmetric.** The Hansen (2000) interval inverts a nonstandard LR statistic and is generally asymmetric around ``\hat\gamma``; a very sharp threshold in a large sample can produce a degenerate interval at ``\hat\gamma``.
5. **Confusing the delay `d` with the AR order `p`.** In `estimate_setar(y, p, d)`, `p` is the number of autoregressive lags per regime and `d` is the delay of the threshold variable ``y_{t-d}``.
6. **Requesting an untabulated `ci_level`.** Hansen's (2000) critical values exist only for `0.90`, `0.95`, and `0.99`; any other level raises an `ArgumentError` rather than interpolating a value that has no tabulation behind it.
7. **Comparing STAR ``\hat\gamma`` across series without the scaling in mind.** The reported slope is already divided by ``\hat\sigma_s``, so it is dimension-free — but that also means it is *not* the raw slope of the exponent, and multiplying it back requires the `sigma_s` field.

---

## References

- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is present only under the alternative. *Biometrika* 74(1), 33–43. [doi:10.1093/biomet/74.1.33](https://doi.org/10.1093/biomet/74.1.33)
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2), 357–384. [doi:10.2307/1912559](https://doi.org/10.2307/1912559)
- Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. *The Review of Economics and Statistics* 100(5), 831–843. [doi:10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Hansen, B. E. (1996). Inference when a nuisance parameter is not identified under the null hypothesis. *Econometrica* 64(2), 413–430. [doi:10.2307/2171789](https://doi.org/10.2307/2171789)
- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics* 60(1–2), 1–22. [doi:10.1016/0304-4076(94)90036-1](https://doi.org/10.1016/0304-4076(94)90036-1)
- Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68(3), 575–603. [doi:10.1111/1468-0262.00124](https://doi.org/10.1111/1468-0262.00124)
- Luukkonen, R., Saikkonen, P. & Teräsvirta, T. (1988). Testing linearity against smooth transition autoregressive models. *Biometrika* 75(3), 491–499. [doi:10.1093/biomet/75.3.491](https://doi.org/10.1093/biomet/75.3.491)
- Teräsvirta, T. (1994). Specification, estimation, and evaluation of smooth transition autoregressive models. *Journal of the American Statistical Association* 89(425), 208–218. [doi:10.1080/01621459.1994.10476462](https://doi.org/10.1080/01621459.1994.10476462)
- Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach.* Oxford University Press. ISBN 978-0-19-852300-6.
- van Dijk, D., Teräsvirta, T. & Franses, P. H. (2002). Smooth transition autoregressive models — a survey of recent developments. *Econometric Reviews* 21(1), 1–47. [doi:10.1081/ETC-120008723](https://doi.org/10.1081/ETC-120008723)
