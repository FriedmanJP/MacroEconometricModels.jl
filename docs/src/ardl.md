# [ARDL & Bounds Testing](@id ardl_page)

**MacroEconometricModels.jl** estimates autoregressive distributed-lag (ARDL) models and tests for a long-run (level) relationship with the Pesaran–Shin–Smith (2001) bounds test. The ARDL approach is the workhorse for cointegration analysis when the regressors are a mix of ``I(0)`` and ``I(1)`` variables: it recovers the long-run multipliers, the speed of adjustment, and the short-run dynamics from a single OLS regression, and its bounds test sidesteps the need to pre-test every series for a unit root.

- **ARDL(p, q₁…q_k) by OLS** — `estimate_ardl` fits ``y_t = c + \delta t + \sum_{i=1}^{p}\varphi_i y_{t-i} + \sum_{j=1}^{k}\sum_{\ell=0}^{q_j}\beta_{j\ell} x_{j,t-\ell} + u_t`` on the lagged levels, with optional AIC/BIC lag selection on a common effective sample
- **Long-run coefficients** — `long_run` returns ``\hat\theta_j = (\sum_\ell\hat\beta_{j\ell})/(1-\sum_i\hat\varphi_i)`` with analytic delta-method standard errors
- **Conditional error-correction form** — the speed of adjustment ``\alpha = \sum_i\hat\varphi_i - 1`` and the long-run levels term, recovered without re-fitting
- **Pesaran–Shin–Smith bounds test** — `bounds_test` reports the non-standard ``F``-statistic on the level block and the ``t``-statistic on the lagged dependent level, each compared to the tabulated ``I(0)``/``I(1)`` critical-value bounds
- **Asymmetric ARDL (NARDL)** — `estimate_nardl` splits a regressor into positive and negative partial sums, with symmetry tests and cumulative dynamic multipliers
- **Panel ARDL** — `estimate_pmg` fits the Pooled Mean Group, Mean Group, and Dynamic Fixed Effects estimators on a dynamic heterogeneous panel

All single-equation models return an [`ARDLModel`](@ref) and integrate with `report` and `refs`.

ARDL is the **single-equation** route to a long-run relationship: it conditions on ``x_t`` and estimates one cointegrating equation. Two neighbouring pages take the other routes. [Cointegrating Regression (FMOLS / CCR / DOLS)](@ref cointreg_page) also estimates a single cointegrating vector, but requires every regressor to be ``I(1)`` and corrects the OLS-on-levels bias directly rather than through a lag structure. [Vector Error Correction Models](@ref vecm_page) treats all variables as endogenous and estimates the full **system**, allowing more than one cointegrating vector.

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
(p = m_auto.p, q = m_auto.q, criterion = m_auto.ic,
 aic = round(m_auto.aic, digits=2), n = m_auto.n)
```

**Recipe 3: Recover the long-run multipliers**

```@example ardl
lr = long_run(m)
(theta = round.(lr.theta, digits=4), se = round.(lr.se, digits=4),
 denom = round(lr.denom, digits=4))
```

**Recipe 4: Test for a level relationship**

```@example ardl
bt = bounds_test(m)
report(bt)
```

**Recipe 5: Allow asymmetric long-run effects**

```@example ardl
# Split every regressor into x⁺ and x⁻ partial sums (NARDL)
nm = estimate_nardl(y, x; asymmetric=:all, p=1, q=1, case=3)
lr_asym = long_run(nm)
(names = lr_asym.varnames, theta = round.(lr_asym.theta, digits=3))
```

Both partial sums load at ``2.007``, recovering the single ``\theta = 2`` of this symmetric data-generating process — the correct answer when there is no asymmetry to find. The [Asymmetric ARDL (NARDL)](@ref) section below fits the same estimator to a genuinely asymmetric DGP.

---

## Estimation

`estimate_ardl(y, X; p, q, case, ...)` fits the ARDL model by OLS on the lagged levels of `y` and the columns of `X` (no intercept column — deterministics are added according to `case`). The effective sample begins at ``t = L+1`` with ``L = \max(p, \max_j q_j)``, so every lag is in-sample.

When `p` or `q` is `:auto`, every candidate ``(p, q_1,\dots,q_k)`` in the grid ``1..\texttt{max\_p} \times (0..\texttt{max\_q})^k`` is scored on the **same** effective sample (trimmed to ``\max(\texttt{max\_p}, \texttt{max\_q})`` lost observations) so the information criteria are directly comparable; the minimiser is then re-fitted on its own maximal sample.

```@example ardl
# A fixed-lag ARDL(2, 3); q may also be a per-regressor vector for multiple x's
m2 = estimate_ardl(y, x; p=2, q=3, case=3)
(coefficients = m2.K, n = m2.n,
 aic = round(m2.aic, digits=2), bic = round(m2.bic, digits=2))
```

The fixed ARDL(2, 3) spends seven coefficients (intercept, two ``y`` lags, four ``x`` lags) on 197 effective observations and scores AIC 97.40 against the ARDL(1, 1) benchmark of 101.82 — the richer lag structure buys a better in-sample fit. BIC tells the opposite story (123.67 against 118.28) because it penalises the three extra coefficients more heavily. On this DGP the AIC grid search of Recipe 2 selects ARDL(2; 4), which is over-parameterised relative to the true ARDL(1, 1) data-generating process: AIC is not consistent for the lag order, so read the selected orders as an upper bound rather than a point estimate.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `p` | `Symbol` or `Integer` | `:auto` | Autoregressive order, or `:auto` for IC grid search |
| `q` | `Symbol`, `Integer`, or `AbstractVector` | `:auto` | Distributed-lag order — a scalar applies to every regressor, a vector gives one per regressor |
| `max_p` | `Int` | `4` | Largest ``p`` considered when `p=:auto` |
| `max_q` | `Int` | `4` | Largest ``q_j`` considered when `q=:auto` |
| `ic` | `Symbol` | `:aic` | Selection criterion (`:aic` or `:bic`) |
| `case` | `Int` | `3` | PSS (2001) deterministic case ``\in 1..5`` |
| `trend` | `Symbol` | `:none` | Deterministics override (`:none`, `:const`, `:trend`); normally implied by `case` |
| `xnames` | `Vector{String}` | `["x1", …]` | Regressor labels used in the coefficient tables |
| `yname` | `AbstractString` | `"y"` | Dependent-variable label |

The fitted [`ARDLModel`](@ref) carries the OLS block, the index bookkeeping that `long_run` and `bounds_test` reuse, and the cached long-run result:

| Field | Type | Description |
|-------|------|-------------|
| `coef` | `Vector{T}` | OLS coefficients, ordered `[deterministics; y lags; x lags]` |
| `vcov` | `Matrix{T}` | ``\hat\sigma^2 (X'X)^{-1}`` |
| `residuals` / `fitted` | `Vector{T}` | OLS residuals and fitted values |
| `p` / `q` | `Int` / `Vector{Int}` | Autoregressive order and per-regressor distributed-lag orders |
| `case` / `trend` | `Int` / `Symbol` | PSS deterministic case and the deterministics actually in `X` |
| `n` / `K` | `Int` | Effective sample size and number of coefficients |
| `sigma2` / `loglik` | `T` | ``\hat\sigma^2 = \text{SSR}/(n-K)`` and the Gaussian log-likelihood |
| `aic` / `bic` | `T` | Information criteria |
| `ar_idx` / `x_idx` / `det_idx` | `Vector{Int}` / `Vector{Vector{Int}}` / `Vector{Int}` | Columns of `X` holding the ``y`` lags, each regressor's lags, and the deterministics |
| `selected` / `ic` | `Bool` / `Symbol` | Whether `(p, q)` came from the IC grid, and which criterion |
| `longrun` | `ARDLLongRun{T}` | Cached long-run block (see below) |

---

## Long-run coefficients and the error-correction form

The long-run multiplier of ``x_j`` on ``y`` is ``\hat\theta_j = (\sum_\ell\hat\beta_{j\ell})/(1-\sum_i\hat\varphi_i)``; its standard error follows from the delta method applied analytically to the full OLS variance matrix. The denominator ``1-\sum_i\hat\varphi_i`` is the negative of the error-correction speed of adjustment ``\alpha``, so a value near zero (a near-unit-root ``y``) inflates both the multipliers and their standard errors.

```@example ardl
lr = long_run(m)
(theta = round.(lr.theta, digits=4), denom = round(lr.denom, digits=4))
```

The estimated long-run multiplier is ``\hat\theta = 2.0063`` with a delta-method standard error of ``0.0126``, so the true value of 2 sits comfortably inside the 95% interval ``[1.982, 2.031]``. The denominator ``1 - \sum_i\hat\varphi_i = 0.4069`` is well away from zero — the dependent variable is not close to a unit root — which is why the multiplier is estimated this precisely. The same quantity read with the opposite sign is the speed of adjustment ``\alpha = -0.4069``, matching the 0.4 built into the data-generating process.

The conditional error-correction re-parameterisation writes the same fitted model as ``\Delta y_t = c + \alpha(y_{t-1} - \theta' x_{t-1}) + \text{short-run } \Delta\text{ terms} + u_t``. A negative ``\alpha`` indicates the system corrects deviations from the long-run relationship; `report(m)` prints ``\alpha`` with its ``t``-ratio in the error-correction block, where it appears as ``-0.4069`` with ``t = -36.47``. Roughly 41% of any gap between ``y_t`` and its long-run level ``2 x_t`` is closed within one period.

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vector{T}` | Long-run multipliers, one per regressor |
| `se` | `Vector{T}` | Delta-method standard errors |
| `denom` | `T` | The common denominator ``1 - \sum_i\hat\varphi_i`` |
| `varnames` | `Vector{String}` | Regressor labels aligned with `theta` |

---

## The Pesaran–Shin–Smith bounds test

`bounds_test(m; case, level, cv_source)` tests the null of **no level relationship**. Two statistics are reported:

- the **``F``-statistic** — a joint Wald/``F`` test that all error-correction level coefficients are zero (the lagged dependent level and every lagged regressor level, plus the restricted intercept/trend in cases II/IV);
- the **``t``-statistic** — the Dickey–Fuller-type ``t``-ratio on the lagged dependent level.

!!! warning "Never read a p-value off the bounds test"
    Both distributions are **non-standard** functionals of Brownian motion, so each statistic is compared **only** to the tabulated ``I(0)``/``I(1)`` bounds — never to an ``F`` or ``t`` p-value. Above the ``I(1)`` upper bound ⇒ a level relationship exists; below the ``I(0)`` lower bound ⇒ none; in between ⇒ inconclusive. `bounds_test` deliberately reports no p-value.

```@example ardl
bt = bounds_test(m; level=0.05)
(F = round(bt.fstat, digits=2), F_bounds_5pct = (bt.f_lower[2], bt.f_upper[2]),
 t = round(bt.tstat, digits=2), t_bounds_5pct = (bt.t_lower[2], bt.t_upper[2]),
 decision_F = bt.f_decision, decision_t = bt.t_decision)
```

The ``F``-statistic of 695.60 is two orders of magnitude above the 5% ``I(1)`` upper bound of 5.73, so the null of no level relationship is rejected regardless of whether the regressor is treated as ``I(0)`` or ``I(1)`` — the decision is unambiguous rather than inconclusive. The ``t``-statistic of ``-36.47`` tells the same story from the other direction: it lies far below the ``I(1)`` bound of ``-3.22``, confirming that the lagged level of ``y`` carries a genuine error-correction coefficient. Agreement between the two statistics matters: when the ``F`` rejects but the ``t`` does not, the ``F`` can be **degenerate** (driven by the regressor levels alone), and PSS (2001) require both to point the same way before a level relationship is declared.

The `case` keyword selects the PSS (2001) deterministic specification and its critical-value table:

| Case | Deterministics | Bounds table |
|------|----------------|--------------|
| I    | none | CI(i) / CII(i) |
| II   | restricted intercept | CI(ii) |
| III  | unrestricted intercept (default) | CI(iii) / CII(iii) |
| IV   | unrestricted intercept + restricted trend | CI(iv) |
| V    | unrestricted intercept + trend | CI(v) / CII(v) |

The ``t``-bounds are tabulated only for cases I, III, and V; cases II and IV restrict a deterministic under the null and have no standard ``t``-bounds test, so `t_decision` returns `:undefined` there.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `case` | `Int` | `m.case` | PSS deterministic case ``\in 1..5``; defaults to the fitted model's case |
| `level` | `Real` | `0.05` | Significance level for the reported decision (`0.10`, `0.05`, `0.025`, `0.01`) |
| `cv_source` | `Symbol` | `:pss` | Critical-value table; only the asymptotic PSS (2001) bounds are bundled |

The returned [`ARDLBoundsTest`](@ref) carries the full bounds table, not just the decision level:

| Field | Type | Description |
|-------|------|-------------|
| `fstat` / `tstat` | `T` | Bounds ``F``-statistic and ``t``-statistic on the lagged ``y`` level |
| `k` | `Int` | Number of distributed-lag regressors indexing the CV tables |
| `levels` | `Vector{T}` | Tabulated significance levels ``[0.10, 0.05, 0.025, 0.01]`` |
| `f_lower` / `f_upper` | `Vector{T}` | ``I(0)``/``I(1)`` ``F``-bounds at each level |
| `t_lower` / `t_upper` | `Vector{T}` | ``I(0)``/``I(1)`` ``t``-bounds; `NaN` where undefined for the case |
| `f_decision` / `t_decision` | `Symbol` | `:cointegrated`, `:not_cointegrated`, `:inconclusive`, or `:undefined` |
| `level` / `n` | `T` / `Int` | Level used for the reported decision, and the effective sample size |

---

## Asymmetric ARDL (NARDL)

The nonlinear ARDL of Shin, Yu & Greenwood-Nimmo (2014) lets a regressor push ``y`` differently when it rises than when it falls. Each *asymmetric* regressor ``x_j`` is decomposed into positive and negative **partial sums**,

```math
x^{+}_{j,t} = \sum_{s\le t}\max(\Delta x_{j,s},0), \qquad
x^{-}_{j,t} = \sum_{s\le t}\min(\Delta x_{j,s},0),
```

where:
- ``x^{+}_{j,t}`` cumulates only the increases in ``x_j`` up to ``t``
- ``x^{-}_{j,t}`` cumulates only the decreases
- both start at ``x^{+}_{j,0}=x^{-}_{j,0}=0``, so that ``x_{j,t}=x_{j,1}+x^{+}_{j,t}+x^{-}_{j,t}`` exactly

The two partial sums are cumulated **levels** (``I(1)`` like ``x``) and replace ``x_j`` in the ARDL. `estimate_nardl` builds this enlarged design and hands it to the same estimation, `long_run`, and `bounds_test` machinery — so an asymmetric regressor contributes **two** columns to the bounds-table ``k``.

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

The long-run block splits into ``\hat\theta^{+} = 1.4954`` on `x1_POS` and ``\hat\theta^{-} = -0.5043`` on `x1_NEG`, both within one standard error of the ``1.5`` and ``-0.5`` built into the data-generating process. The bounds test is now read at the **enlarged** ``k = 2``, whose 5% ``F``-bounds are ``[3.79, 4.85]`` rather than the ``[4.94, 5.73]`` of the symmetric ``k = 1`` fit; the statistic of 117.59 clears the upper bound either way. Reading a NARDL bounds test at the original ``k`` is a common error that makes the test look more conservative than it is.

```@example ardl
lr_nardl = long_run(nm)
(names = lr_nardl.varnames, theta = round.(lr_nardl.theta, digits=3),
 se = round.(lr_nardl.se, digits=3), bounds_k = nm.bounds.k)
```

`symmetry_test` runs, per asymmetric regressor, a **long-run** symmetry Wald ``H_0:\theta^{+}=\theta^{-}`` (delta method) and a **short-run** symmetry Wald ``H_0:\sum_\ell\pi^{+}_\ell=\sum_\ell\pi^{-}_\ell`` on the ECM differenced-term coefficients. Each single-restriction statistic is a ``\chi^2(1)`` (equivalently ``F(1,n-K)``):

```@example ardl
st = symmetry_test(nm)
report(st)
```

Both restrictions are rejected decisively. The long-run Wald is enormous because ``\hat\theta^{+}`` and ``\hat\theta^{-}`` differ by 2.0 while each is pinned down to within ``0.016`` — the delta-method variance of the difference is tiny. The short-run statistic of 99.83 is far smaller but still rejects at any conventional level, so the asymmetry is present in the transition dynamics as well as in the long-run level. Rejecting only the long-run restriction would indicate a symmetric adjustment path toward asymmetric equilibria.

`dynamic_multipliers(nm, H)` recursively iterates the estimated ARDL to a unit permanent shock in ``x^{+}`` (then ``x^{-}``), giving the cumulative dynamic multipliers ``m^{+}_h`` and ``m^{-}_h``, which converge to ``\theta^{+}`` and ``\theta^{-}``. Pointwise bands come from a recursive-design (condition-on-``x``) residual bootstrap:

```@example ardl
mm = dynamic_multipliers(nm, 24; bootstrap=true, nreps=200, level=0.90,
                         rng=MersenneTwister(1))
report(mm)
```

The multipliers reach their long-run limits quickly: ``m^{+}_h`` climbs from ``-0.008`` on impact to ``1.4954`` by ``h = 24``, and ``m^{-}_h`` falls to ``-0.5043`` — exactly the ``\theta^{+}`` and ``\theta^{-}`` reported above, as the recursion requires. Just over half of each adjustment is complete by ``h = 1`` (``0.7744`` of ``1.4954``, and ``-0.2410`` of ``-0.5043``), consistent with a speed of adjustment near ``0.4``. The asymmetry curve ``m^{+}_h - m^{-}_h`` settles at ``2.00`` with a 90% band of ``[1.996, 2.004]`` that excludes zero from ``h = 1`` onward, so the two responses are statistically distinguishable at every horizon after impact.

```julia
plot_result(mm; view=:multipliers)          # or plot_result(nm; view=:multipliers, H=24)
```

`estimate_nardl` adds one keyword to the `estimate_ardl` list; the rest (`p`, `q`, `max_p`, `max_q`, `ic`, `case`, `xnames`, `yname`) carry over unchanged.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `asymmetric` | `Symbol` or `AbstractVector{<:Integer}` | `:all` | Which columns of `X` to split into partial sums |

`dynamic_multipliers(m, H; ...)` takes its own bootstrap keywords:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `bootstrap` | `Bool` | `true` | Compute pointwise bootstrap bands |
| `nreps` | `Int` | `500` | Bootstrap replications |
| `level` | `Real` | `0.95` | Coverage of the pointwise bands |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source; pass a seeded generator for reproducible bands |

---

## Panel ARDL (PMG / MG / DFE)

For a **dynamic heterogeneous panel** — many cross-sectional units, each an ARDL(``p``, ``q``) in error-correction form — `estimate_pmg` fits three estimators that trade off long-run homogeneity against short-run flexibility:

```math
\Delta y_{it} = \varphi_i\,\bigl(y_{i,t-1} - \theta' x_{i,t-1}\bigr)
              + \sum_{j=1}^{p-1}\xi_{ij}\,\Delta y_{i,t-j}
              + \sum_{j=0}^{q-1}\psi_{ij}'\,\Delta x_{i,t-j} + \mu_i + \varepsilon_{it}.
```

where:
- ``\varphi_i`` is unit ``i``'s speed of adjustment (negative under error correction)
- ``\theta`` is the long-run vector, common across units under PMG
- ``\xi_{ij}``, ``\psi_{ij}`` are the unit-specific short-run dynamics
- ``\mu_i`` is a unit fixed effect and ``\varepsilon_{it}`` the idiosyncratic error

- **`:pmg`** — *Pooled Mean Group* (Pesaran, Shin & Smith 1999): the long-run vector ``\theta`` is **common** across units, while the speed of adjustment ``\varphi_i``, the short-run dynamics, and ``\sigma^2_i`` are heterogeneous. ``\theta`` maximises the concentrated (profile) likelihood.
- **`:mg`** — *Mean Group* (Pesaran & Smith 1995): an unrestricted ARDL per unit, with ``\bar\theta = N^{-1}\sum_i\theta_i`` and the Swamy between-unit standard error ``\operatorname{std}(\theta_i)/\sqrt N``.
- **`:dfe`** — *Dynamic Fixed Effects*: a pooled within-transformed EC regression (common ``\varphi,\theta``, short-run) with unit intercepts and cluster-robust standard errors.

Pass a [`PanelData`](@ref) from `xtset`, the dependent-variable symbol, and one or more long-run regressor symbols. Units with ``\varphi_i \ge 0`` (non-error-correcting) are flagged.

```@setup ardlpanel
using MacroEconometricModels, Random, Statistics, DataFrames

# Fixed-seed heterogeneous panel: common long-run θ = 1.5, heterogeneous φ_i and
# short-run dynamics, an I(1) regressor x, and an error-correction DGP.
Random.seed!(431)
N, Tt = 20, 60
rows = NamedTuple[]
for i in 1:N
    phi = -(0.2 + 0.4 * rand())            # φ_i ∈ (-0.6, -0.2)
    g   = 0.3 * randn()
    x = zeros(Tt); y = zeros(Tt)
    x[1] = randn(); y[1] = 1.5 * x[1] + randn()
    for t in 2:Tt
        x[t] = x[t-1] + randn()
        y[t] = y[t-1] + phi * (y[t-1] - 1.5 * x[t-1]) + g * (x[t] - x[t-1]) + 0.3 * randn()
    end
    for t in 1:Tt
        push!(rows, (id = i, time = t, y = y[t], x = x[t]))
    end
end
pd = xtset(DataFrame(rows), :id, :time)
```

```@example ardlpanel
# Pooled Mean Group: common long-run θ (should be near the true 1.5)
pmg = estimate_pmg(pd, :y, :x; p=1, q=1, method=:pmg)
report(pmg)
```

The pooled long-run coefficient is ``\hat\theta = 1.4998`` with a standard error of ``0.0060``, recovering the common ``1.5`` imposed on all 20 units. The mean speed of adjustment ``\hat\varphi = -0.4568`` says a typical unit closes 46% of its long-run gap each period, and no unit is flagged with ``\varphi_i \ge 0``, so every cross-section error-corrects. The short-run coefficient on ``\Delta x`` is insignificant, as it should be: the DGP draws each unit's short-run slope from a mean-zero normal, so the cross-unit average is zero even though individual units respond.

```@example ardlpanel
# Mean Group and Dynamic Fixed Effects on the same panel
mg  = estimate_pmg(pd, :y, :x; p=1, q=1, method=:mg)
dfe = estimate_pmg(pd, :y, :x; p=1, q=1, method=:dfe)
(pmg = round(pmg.theta[1], digits=4), mg = round(mg.theta[1], digits=4),
 dfe = round(dfe.theta[1], digits=4),
 se = round.([pmg.theta_se[1], mg.theta_se[1], dfe.theta_se[1]], digits=4))
```

All three estimators land within ``0.002`` of the true 1.5, which is what long-run homogeneity guarantees: when the restriction PMG imposes is true, PMG, MG, and DFE are all consistent and differ only in efficiency. The ordering of the standard errors — ``0.0060`` for PMG, ``0.0073`` for MG, ``0.0082`` for DFE — shows PMG buying precision from the pooling restriction. Under long-run *heterogeneity* the ranking would be irrelevant, because PMG and DFE would be inconsistent while MG would not.

The **Hausman test** compares the pooled estimator (efficient under long-run homogeneity) against the always-consistent Mean Group estimator. Failing to reject ``H_0`` supports pooling the long-run relationship:

```@example ardlpanel
h = hausman_test(pmg, mg)   # H0: long-run homogeneity ⇒ PMG efficient
report(h)
```

The statistic is ``\chi^2(1) = 0.054`` with ``p = 0.816``, so the data give no evidence against long-run homogeneity and the efficient PMG estimator is preferred — the correct verdict, since the DGP imposes a common ``\theta``. A rejection would flip the recommendation to Mean Group, which stays consistent under heterogeneous long-run slopes at the cost of the larger standard error seen above. The test compares only the long-run block; heterogeneity in ``\varphi_i`` and the short-run dynamics is permitted under both hypotheses.

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
m_full = estimate_ardl(yy, xx; p=:auto, q=:auto, max_p=4, max_q=4, ic=:aic, case=3)
report(m_full)
```

```@example ardl
# 2. Bounds test for a level relationship
bt_full = bounds_test(m_full)
report(bt_full)
```

AIC selects ARDL(1; 1) on 249 effective observations — here the criterion recovers the true lag structure. The long-run multiplier ``\hat\theta = 1.4939`` (standard error ``0.0055``) sits within one standard error of the true 1.5, and the speed of adjustment ``\alpha = -0.4907`` reproduces the 0.5 in the data-generating process. The bounds ``F`` of 512.33 and ``t`` of ``-31.80`` both clear their 5% ``I(1)`` bounds (``5.73`` and ``-3.22``) by a wide margin, so the level relationship is confirmed without ambiguity.

---

## Common Pitfalls

1. **Do not read a p-value off the bounds test.** The PSS ``F`` and ``t`` are non-standard; compare them only to the ``I(0)``/``I(1)`` bounds. `bounds_test` deliberately reports no p-value.

2. **Match the `case` to your data.** Cases II/IV place the intercept/trend under the null (they enter the level test); cases III/V leave them unrestricted. A wrong case–regressor mapping is the classic ARDL bug and changes both the statistic and the bounds table.

3. **`X` carries no intercept column.** Deterministics are added internally according to `case`; passing a constant column double-counts the intercept.

4. **A near-unit-root dependent variable inflates the long-run block.** When ``1-\sum\hat\varphi_i`` is close to zero the multipliers and their delta-method standard errors blow up — check the speed of adjustment ``\alpha`` in the error-correction block before trusting ``\hat\theta``.

5. **Read a NARDL bounds test at the enlarged ``k``.** Each asymmetric regressor contributes two columns, so a one-regressor NARDL is tested against the ``k = 2`` bounds. `nm.bounds.k` reports the value actually used.

6. **Keep the column order of `X` stable.** `long_run` and the bounds Wald block index regressors by column; reordering `X` reorders the reported long-run coefficients.

7. **Only the asymptotic bounds are bundled.** `cv_source=:pss` supplies the PSS (2001) asymptotic tables; the Narayan (2005) small-sample bounds are not included, and requesting `cv_source=:narayan` raises an error rather than returning transcribed values.

---

## References

```@example ardl
print(refs(m))
```

- Pesaran, M. H., & Shin, Y. (1999). An Autoregressive Distributed Lag Modelling Approach to Cointegration Analysis. In S. Strøm (Ed.), *Econometrics and Economic Theory in the 20th Century*, 371-413. Cambridge University Press. ISBN 978-0-521-63323-9.

- Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds Testing Approaches to the Analysis of Level Relationships.
  *Journal of Applied Econometrics*, 16(3), 289-326. [DOI](https://doi.org/10.1002/jae.616)

- Narayan, P. K. (2005). The Saving and Investment Nexus for China: Evidence from Cointegration Tests.
  *Applied Economics*, 37(17), 1979-1990. [DOI](https://doi.org/10.1080/00036840500278103)

- Kripfganz, S., & Schneider, D. C. (2023). ardl: Estimating Autoregressive Distributed Lag and Equilibrium Correction Models.
  *Stata Journal*, 23(4), 983-1019. [DOI](https://doi.org/10.1177/1536867X231212434)

- Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework. In R. C. Sickles & W. C. Horrace (Eds.), *Festschrift in Honor of Peter Schmidt*, 281-314. Springer. [DOI](https://doi.org/10.1007/978-1-4899-8008-3_9)

- Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled Mean Group Estimation of Dynamic Heterogeneous Panels.
  *Journal of the American Statistical Association*, 94(446), 621-634. [DOI](https://doi.org/10.1080/01621459.1999.10474156)

- Pesaran, M. H., & Smith, R. (1995). Estimating Long-Run Relationships from Dynamic Heterogeneous Panels.
  *Journal of Econometrics*, 68(1), 79-113. [DOI](https://doi.org/10.1016/0304-4076(94)01644-F)

- Blackburne, E. F., & Frank, M. W. (2007). Estimation of Nonstationary Heterogeneous Panels.
  *Stata Journal*, 7(2), 197-208. [DOI](https://doi.org/10.1177/1536867X0700700204)
