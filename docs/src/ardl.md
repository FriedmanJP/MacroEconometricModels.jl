# [ARDL & Bounds Testing](@id ardl_page)

**MacroEconometricModels.jl** estimates autoregressive distributed-lag (ARDL) models and tests for a long-run (level) relationship with the Pesaran–Shin–Smith (2001) bounds test. The ARDL approach is the workhorse for cointegration analysis when the regressors are a mix of ``I(0)`` and ``I(1)`` variables: it recovers the long-run multipliers, the speed of adjustment, and the short-run dynamics from a single OLS regression, and its bounds test sidesteps the need to pre-test every series for a unit root.

- **ARDL(p, q₁…q_k) by OLS** — `estimate_ardl` fits ``y_t = c + \delta t + \sum_{i=1}^{p}\varphi_i y_{t-i} + \sum_{j=1}^{k}\sum_{\ell=0}^{q_j}\beta_{j\ell} x_{j,t-\ell} + u_t`` on the lagged levels, with optional AIC/BIC lag selection on a common effective sample
- **Long-run coefficients** — `long_run` returns ``\hat\theta_j = (\sum_\ell\hat\beta_{j\ell})/(1-\sum_i\hat\varphi_i)`` with analytic delta-method standard errors
- **Conditional error-correction form** — the speed of adjustment ``\alpha = \sum_i\hat\varphi_i - 1`` and the long-run levels term, recovered without re-fitting
- **Pesaran–Shin–Smith bounds test** — `bounds_test` reports the non-standard ``F``-statistic on the level block and the ``t``-statistic on the lagged dependent level, each compared to the tabulated ``I(0)``/``I(1)`` critical-value bounds
- **Five deterministic cases** — PSS (2001) cases I–V select which deterministics enter and which bounds table applies

All models return an [`ARDLModel`](@ref) and integrate with `report` and `refs`. This module is the scaffold for the nonlinear-ARDL (NARDL) and pooled-mean-group (PMG) panel extensions.

```@setup ardl
using MacroEconometricModels, Random
# A fixed-seed cointegrated pair with a KNOWN long-run multiplier θ = 2 on x.
Random.seed!(20240716)
T = 200
x = cumsum(randn(T))
y = zeros(T)
for t in 2:T
    y[t] = y[t-1] - 0.4 * (y[t-1] - 2.0 * x[t-1]) + 0.5 * (x[t] - x[t-1]) + 0.3 * randn()
end
```

## Quick Start

**Recipe 1: Estimate an ARDL model**

```@example ardl
# ARDL(1, 1) with an unrestricted intercept (PSS case III, the default)
m = estimate_ardl(y, x; p=1, q=1, case=3)
report(m)
```

**Recipe 2: Select the lag orders automatically**

```@example ardl
# Grid-search p ∈ 1:4, q ∈ 0:4 by AIC on a common effective sample
m_auto = estimate_ardl(y, x; p=:auto, q=:auto, max_p=4, max_q=4, ic=:aic)
println("selected ARDL(", m_auto.p, "; ", join(m_auto.q, ", "), ")")
```

**Recipe 3: Recover the long-run multipliers**

```@example ardl
lr = long_run(m)
println("long-run θ = ", round(lr.theta[1], digits=4),
        "  (se = ", round(lr.se[1], digits=4), ")")
```

**Recipe 4: Test for a level relationship**

```@example ardl
bt = bounds_test(m)
report(bt)
```

---

## Estimation

`estimate_ardl(y, X; p, q, case, ...)` fits the ARDL model by OLS on the lagged levels of `y` and the columns of `X` (no intercept column — deterministics are added according to `case`). The effective sample begins at ``t = L+1`` with ``L = \max(p, \max_j q_j)``, so every lag is in-sample.

When `p` or `q` is `:auto`, every candidate ``(p, q_1,\dots,q_k)`` in the grid `1:max_p × (0:max_q)^k` is scored on the **same** effective sample (trimmed to `max(max_p, max_q)` lost observations) so the information criteria are directly comparable; the minimiser is then re-fitted on its own maximal sample.

```@example ardl
# A fixed-lag ARDL(2, 3); q may also be a per-regressor vector for multiple x's
m2 = estimate_ardl(y, x; p=2, q=3, case=3)
println("coefficients: ", length(m2.coef), "  effective n = ", m2.n)
println("AIC = ", round(m2.aic, digits=3), "   BIC = ", round(m2.bic, digits=3))
```

---

## Long-run coefficients and the error-correction form

The long-run multiplier of `x_j` on `y` is ``\hat\theta_j = (\sum_\ell\hat\beta_{j\ell})/(1-\sum_i\hat\varphi_i)``; its standard error follows from the delta method applied analytically to the full OLS variance matrix. The denominator ``1-\sum_i\hat\varphi_i`` is the negative of the error-correction speed of adjustment ``\alpha``, so a value near zero (a near-unit-root `y`) inflates both the multipliers and their standard errors.

```@example ardl
lr = long_run(m)
println("θ = ", round(lr.theta[1], digits=4),
        "   1 − Σφ = ", round(lr.denom, digits=4))
```

The conditional error-correction re-parameterisation writes the same fitted model as ``\Delta y_t = c + \alpha(y_{t-1} - \theta' x_{t-1}) + \text{short-run } \Delta\text{ terms} + u_t``. A negative ``\alpha`` indicates the system corrects deviations from the long-run relationship; `report(m)` prints ``\alpha`` with its ``t``-ratio in the error-correction block.

---

## The Pesaran–Shin–Smith bounds test

`bounds_test(m; case, level, cv_source)` tests the null of **no level relationship**. Two statistics are reported:

- the **``F``-statistic** — a joint Wald/``F`` test that all error-correction level coefficients are zero (the lagged dependent level and every lagged regressor level, plus the restricted intercept/trend in cases II/IV);
- the **``t``-statistic** — the Dickey–Fuller-type ``t``-ratio on the lagged dependent level.

Both distributions are **non-standard** functionals of Brownian motion, so each statistic is compared **only** to the tabulated ``I(0)``/``I(1)`` bounds — **never** to an ``F`` or ``t`` p-value. Above the ``I(1)`` upper bound ⇒ a level relationship exists; below the ``I(0)`` lower bound ⇒ none; in between ⇒ inconclusive.

```@example ardl
bt = bounds_test(m; level=0.05)
println("F = ", round(bt.fstat, digits=3),
        "   5% bounds: I(0) = ", bt.f_lower[2], ", I(1) = ", bt.f_upper[2])
println("decision (F): ", bt.f_decision)
```

The `case` keyword selects the PSS (2001) deterministic specification and its critical-value table:

| Case | Deterministics | Bounds table |
|------|----------------|--------------|
| I    | none | CI(i) / CII(i) |
| II   | restricted intercept | CI(ii) |
| III  | unrestricted intercept (default) | CI(iii) / CII(iii) |
| IV   | unrestricted intercept + restricted trend | CI(iv) |
| V    | unrestricted intercept + trend | CI(v) / CII(v) |

The ``t``-bounds are tabulated only for cases I, III, and V (cases II and IV restrict a deterministic and have no standard ``t``-bounds test).

---

## Asymmetric ARDL (NARDL)

The nonlinear ARDL of Shin, Yu & Greenwood-Nimmo (2014) lets a regressor push ``y`` differently when it rises than when it falls. Each *asymmetric* regressor ``x_j`` is decomposed into positive and negative **partial sums**,

```math
x^{+}_{j,t} = \sum_{s\le t}\max(\Delta x_{j,s},0), \qquad
x^{-}_{j,t} = \sum_{s\le t}\min(\Delta x_{j,s},0),
```

with ``x^{+}_{j,0}=x^{-}_{j,0}=0``, so that ``x_{j,t}=x_{j,1}+x^{+}_{j,t}+x^{-}_{j,t}`` exactly. The two partial sums are cumulated **levels** (``I(1)`` like ``x``) and replace ``x_j`` in the ARDL. `estimate_nardl` builds this enlarged design and hands it to the same estimation, `long_run`, and `bounds_test` machinery — so an asymmetric regressor contributes **two** columns to the bounds-table ``k``.

```@example ardl
using Random
# Asymmetric DGP: y reacts to x⁺ with θ⁺ = 1.5 and to x⁻ with θ⁻ = -0.5
Random.seed!(909)
n = 260
xa = cumsum(randn(n))
dxa = [0.0; diff(xa)]
xap = cumsum(max.(dxa, 0.0)); xan = cumsum(min.(dxa, 0.0))
ya = zeros(n)
for t in 2:n
    ya[t] = ya[t-1] - 0.4 * (ya[t-1] - (1.5*xap[t-1] - 0.5*xan[t-1])) +
            0.25*dxa[t-1] + 0.4*randn()
end

nm = estimate_nardl(ya, xa; asymmetric=:all, p=1, q=1, case=3)
report(nm)
```

The long-run block now carries a separate ``\theta^{+}`` (`…_POS`) and ``\theta^{-}`` (`…_NEG`), and the bounds test is read at the enlarged ``k``:

```@example ardl
lr = long_run(nm)
println("θ⁺ = ", round(lr.theta[1], digits=3), "   θ⁻ = ", round(lr.theta[2], digits=3))
println("bounds k (enlarged) = ", nm.bounds.k)
```

`symmetry_test` runs, per asymmetric regressor, a **long-run** symmetry Wald ``H_0:\theta^{+}=\theta^{-}`` (delta method) and a **short-run** symmetry Wald ``H_0:\sum_\ell\pi^{+}_\ell=\sum_\ell\pi^{-}_\ell`` on the ECM differenced-term coefficients. Each single-restriction statistic is a ``\chi^2(1)`` (equivalently ``F(1,n-K)``):

```@example ardl
st = symmetry_test(nm)
report(st)
```

`dynamic_multipliers(nm, H)` recursively iterates the estimated ARDL to a unit permanent shock in ``x^{+}`` (then ``x^{-}``), giving the cumulative dynamic multipliers ``m^{+}_h`` and ``m^{-}_h``, which converge to ``\theta^{+}`` and ``\theta^{-}``. Pointwise bands come from a recursive-design (condition-on-``x``) residual bootstrap:

```@example ardl
mm = dynamic_multipliers(nm, 24; bootstrap=true, nreps=200, level=0.90,
                         rng=MersenneTwister(1))
println("m⁺(24) = ", round(mm.m_pos[1, end], digits=3),
        "  →  θ⁺ = ", round(mm.theta_pos[1], digits=3))
```

Plot the multipliers (``m^{+}_h``, ``m^{-}_h``, and the asymmetry curve ``m^{+}_h-m^{-}_h`` with their bands):

```julia
plot_result(mm; view=:multipliers)          # or plot_result(nm; view=:multipliers, H=24)
```

---

## Complete Example

```@example ardl
using MacroEconometricModels, Random

# Fixed-seed cointegrated system: y adjusts to a long-run relation with x (θ = 1.5)
Random.seed!(2025)
n = 250
xx = cumsum(randn(n))
yy = zeros(n)
for t in 2:n
    yy[t] = yy[t-1] - 0.5 * (yy[t-1] - 1.5 * xx[t-1]) + 0.4 * (xx[t] - xx[t-1]) + 0.3 * randn()
end

# 1. Select lags and estimate
m = estimate_ardl(yy, xx; p=:auto, q=:auto, max_p=4, max_q=4, ic=:aic, case=3)
report(m)
```

```@example ardl
# 2. Long-run multiplier (should be near the true θ = 1.5)
lr = long_run(m)
println("long-run θ = ", round(lr.theta[1], digits=3))
```

```@example ardl
# 3. Bounds test for a level relationship
bt = bounds_test(m)
report(bt)
```

---

## Common Pitfalls

- **Do not read a p-value off the bounds test.** The PSS ``F`` and ``t`` are non-standard; compare them only to the ``I(0)``/``I(1)`` bounds. `bounds_test` deliberately reports no p-value.
- **Match the `case` to your data.** Cases II/IV place the intercept/trend under the null (they enter the level test); cases III/V leave them unrestricted. A wrong case–regressor mapping is the classic ARDL bug and changes both the statistic and the bounds table.
- **`X` carries no intercept column.** Deterministics are added internally according to `case`; passing a constant column double-counts the intercept.
- **A near-unit-root dependent variable inflates the long-run block.** When ``1-\sum\hat\varphi_i`` is close to zero the multipliers and their delta-method standard errors blow up — check the speed of adjustment ``\alpha`` in the error-correction block.
- **Keep the column order of `X` stable.** `long_run` and the bounds Wald block index regressors by column; reordering `X` reorders the reported long-run coefficients.
- **Finite-sample bounds.** Only the asymptotic PSS (2001) tables (`cv_source=:pss`) are bundled; the Narayan (2005) small-sample bounds are not yet included.

---

## References

```@example ardl
refs(m)
```

- Pesaran, M. H. & Shin, Y. (1999). *An Autoregressive Distributed Lag Modelling Approach to Cointegration Analysis*. In Strøm (ed.), Cambridge University Press.
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds Testing Approaches to the Analysis of Level Relationships. *Journal of Applied Econometrics* 16(3), 289–326.
- Narayan, P. K. (2005). The Saving and Investment Nexus for China: Evidence from Cointegration Tests. *Applied Economics* 37(17), 1979–1990.
- Kripfganz, S. & Schneider, D. C. (2023). ARDL: Estimating Autoregressive Distributed Lag and Equilibrium Correction Models. *Stata Journal* 23(4), 983–1019.
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). *Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework*. In Sickles & Horrace (eds.), Festschrift in Honor of Peter Schmidt, Springer, 281–314.
