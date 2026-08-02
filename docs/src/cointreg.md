# [Cointegrating Regression (FMOLS / CCR / DOLS)](@id cointreg_page)

**MacroEconometricModels.jl** estimates a single cointegrating vector by three asymptotically-efficient, endogeneity-corrected estimators: fully-modified OLS (Phillips–Hansen 1990), canonical cointegrating regression (Park 1992), and dynamic OLS (Saikkonen 1991; Stock & Watson 1993). When ``y_t`` and an ``I(1)`` regressor vector ``x_t`` are cointegrated, plain OLS on the levels is super-consistent but has an asymptotically biased, non-standard distribution because ``x_t`` is correlated with the equation error ``u_t`` and ``u_t`` is serially correlated. All three estimators remove that second-order bias and deliver a mixed-normal long-run coefficient vector on which standard Wald inference is valid.

- **FMOLS** — `estimate_cointreg(y, X; method=:fmols)` corrects the regressand for regressor endogeneity (``y^+_t = y_t - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\Delta x_t``) and applies a serial-correlation bias adjustment built from the one-sided long-run covariance
- **CCR** — `method=:ccr` transforms the *data* (Park's canonical transformation) so that plain OLS on the transformed system is efficient; asymptotically equivalent to FMOLS
- **DOLS** — `method=:dols` augments the levels regression with leads and lags of ``\Delta x_t``; the level-regressor coefficients are the efficient long-run estimates, with automatic AIC/BIC lead/lag selection
- **Reused long-run-variance toolkit** — all three build ``\hat\Omega`` (two-sided), ``\hat\Lambda`` (one-sided) and ``\hat\Sigma = \hat\Gamma_0`` of the stacked ``(u, \Delta x)`` process on the `lrcov`/`lrcov_oneside` HAC estimators
- **Stored covariance pieces** — the fitted [`CointRegModel`](@ref) exposes ``\hat\Omega``, ``\hat\Lambda``, ``\hat\Sigma`` and the conditional long-run variance ``\hat\omega_{u\cdot\Delta x}`` for downstream cointegration stability and panel-cointegration tests
- **Panel extension** — `estimate_xtcointreg(pd, y, xs...)` aggregates the single-unit estimator across the ``N`` units of a [`PanelData`](@ref) into group-mean (Pedroni 2001) or pooled (Pedroni 2000 FMOLS / Kao–Chiang 2000 DOLS) panel FMOLS/DOLS; see [Panel cointegrating regression](@ref cointreg_panel)

All estimators return a [`CointRegModel`](@ref) and integrate with `report` and `refs`.

This page covers the **single-equation** estimators, which condition on ``x_t`` and require every regressor to be ``I(1)``. Two neighbouring pages relax those restrictions. [ARDL & Bounds Testing](@ref ardl_page) is also single-equation but tolerates a mix of ``I(0)`` and ``I(1)`` regressors, recovering the long-run vector from a lag structure rather than a bias correction. [Vector Error Correction Models](@ref vecm_page) treats all variables as endogenous and estimates the full **system**, so it can identify more than one cointegrating vector.

```@setup cointreg
using MacroEconometricModels, Random
# A fixed-seed cointegrated pair y_t = 2 + 1.5 x_t + u_t with endogenous, serially
# correlated errors — exactly the setting where OLS-on-levels is biased.
Random.seed!(20260716)
T = 200
v = randn(T)
e = randn(T)
x = cumsum(v)                       # I(1) regressor (random walk)
u = zeros(T)
for t in 2:T
    u[t] = 0.4 * u[t-1] + e[t] + 0.6 * v[t]   # AR(1) error correlated with Δx
end
y = 2.0 .+ 1.5 .* x .+ u
```

## Quick Start

**Recipe 1: Fully-modified OLS**

```@example cointreg
# FMOLS with an intercept; the true cointegrating vector is (2.0, 1.5)
m = estimate_cointreg(y, x; method=:fmols, trend=:const)
report(m)
```

**Recipe 2: Canonical cointegrating regression**

```@example cointreg
mc = estimate_cointreg(y, x; method=:ccr)
(coef = round.(coef(mc), digits=4), se = round.(stderror(mc), digits=4))
```

**Recipe 3: Dynamic OLS with automatic lead/lag selection**

```@example cointreg
md = estimate_cointreg(y, x; method=:dols, leads=:auto, lags=:auto, ic=:aic)
(leads = md.leads, lags = md.lags, coef = round.(coef(md), digits=4))
```

**Recipe 4: Choose the HAC kernel and bandwidth**

```@example cointreg
# Andrews (1991) plug-in bandwidth with a Parzen kernel
mk = estimate_cointreg(y, x; method=:fmols, kernel=:parzen, bandwidth=:andrews)
(kernel = mk.kernel, bandwidth = round(mk.bandwidth, digits=3),
 coef = round.(coef(mk), digits=4))
```

**Recipe 5: Confidence intervals for the long-run vector**

```@example cointreg
round.(confint(m; level=0.95), digits=4)
```

---

## Fully-modified OLS

`estimate_cointreg(y, X; method=:fmols, ...)` implements Phillips & Hansen (1990). Starting from the OLS-on-levels residual ``\hat u_t``, it forms the stacked process ``\xi_t = (\hat u_t, \Delta x_t')'`` and estimates its long-run covariance ``\hat\Omega`` (two-sided) and one-sided ``\hat\Lambda = \sum_{j\ge 0}\hat\Gamma_j`` on the reused `lrcov`/`lrcov_oneside` toolkit. Partitioning ``\hat\Omega`` into the ``u`` and ``\Delta x`` blocks, the endogeneity-corrected regressand and the serial-correlation bias term give the fully-modified estimator

```math
\hat\theta^{+} = \left(Z'Z\right)^{-1}\left(Z'y^{+} - T\begin{bmatrix}0\\ \hat\Delta^{+}_{\Delta x u}\end{bmatrix}\right),\qquad
y^{+}_t = y_t - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\Delta x_t,
```

where:
- ``Z = [D\ X]`` stacks the deterministics ``D`` and the ``I(1)`` regressors ``X``
- ``y^{+}_t`` is the regressand purged of contemporaneous regressor endogeneity
- ``\hat\Delta^{+}_{\Delta x u}`` is the one-sided serial-correlation bias term built from ``\hat\Lambda``
- ``T`` is the sample size

The coefficient covariance is ``\hat\omega_{u\cdot\Delta x}(Z'Z)^{-1}`` with the conditional long-run variance ``\hat\omega_{u\cdot\Delta x} = \hat\Omega_{uu} - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\hat\Omega_{\Delta x u}``.

!!! warning "Cointegration is assumed, not tested"
    `estimate_cointreg` returns a coefficient vector and Wald-valid standard errors whether or not ``y_t`` and ``x_t`` are cointegrated. On unrelated ``I(1)`` series it estimates a spurious long-run relationship with confident-looking ``t``-statistics. Establish cointegration first with the tests on the [Unit Root & Cointegration](@ref tests_unitroot_page) page.

```@example cointreg
m = estimate_cointreg(y, x; method=:fmols)
b_ols = hcat(ones(length(y)), x) \ y          # uncorrected OLS on the levels
(fmols = round.(coef(m), digits=4), ols = round.(b_ols, digits=4),
 se = round.(stderror(m), digits=4), omega_u_dx = round(m.omega_uv, digits=4))
```

The correction moves the estimate in the direction the theory predicts. Uncorrected OLS on the levels returns a slope of ``1.5431``, biased upward from the true ``1.5`` because ``\Delta x_t`` is positively correlated with the equation error by construction; FMOLS pulls it back to ``1.5328``, removing about a quarter of the ``0.043`` bias. The residual gap of ``0.033`` is two standard errors — the second-order correction is asymptotic, so at ``T = 200`` it removes the leading bias term but not the whole finite-sample discrepancy. The conditional long-run variance ``\hat\omega_{u\cdot\Delta x} = 2.024`` is what scales the standard errors: it exceeds the contemporaneous residual variance because the AR(1) error is positively autocorrelated, so ignoring serial correlation would understate the uncertainty.

---

## Canonical cointegrating regression

Park's (1992) CCR (`method=:ccr`) reaches the same efficient limit by transforming the data rather than the estimator. Using ``\hat\Sigma = \hat\Gamma_0``, ``\hat\Omega`` and ``\hat\Lambda``, it builds transformed regressors ``x^{*}_t`` and regressand ``y^{*}_t`` such that plain OLS of ``y^{*}`` on ``[D\ x^{*}]`` is free of endogeneity and serial-correlation bias.

```@example cointreg
mf = estimate_cointreg(y, x; method=:fmols)
mc = estimate_cointreg(y, x; method=:ccr)
(fmols = round.(coef(mf), digits=4), ccr = round.(coef(mc), digits=4),
 omega_gap = round(abs(mf.omega_uv - mc.omega_uv), sigdigits=2))
```

CCR and FMOLS are asymptotically equivalent, and at ``T = 200`` they agree to four decimal places on the slope (``1.5328`` both ways) and to three on the intercept. Their conditional long-run variances are identical because both are computed from the same ``\hat\Omega``, so the reported standard errors differ only through the design matrix each estimator uses. The choice between them is therefore practical, not statistical: CCR transforms the data once and then runs plain OLS, which makes it easier to feed into downstream diagnostics that expect an ordinary regression.

---

## Dynamic OLS

DOLS (`method=:dols`) augments the levels regression with the contemporaneous value, ``\text{lags}`` lags, and ``\text{leads}`` leads of ``\Delta x_t`` (Saikkonen 1991; Stock & Watson 1993):

```math
y_t = D_t'\delta + x_t'\beta + \sum_{j=-\text{leads}}^{\text{lags}} \Delta x_{t-j}'\gamma_j + u^{*}_t .
```

where:
- ``\beta`` is the long-run (cointegrating) coefficient vector on the levels
- ``\delta`` collects the deterministic coefficients
- ``\gamma_j`` are nuisance coefficients on the leads and lags of ``\Delta x_t``
- ``u^{*}_t`` is orthogonal to the full lead–lag span of ``\Delta x``, which is what removes the endogeneity

Projecting ``u_t`` onto leads and lags of ``\Delta x_t`` is what makes ``(\delta, \beta)`` efficient. Leads and lags may be fixed, or selected automatically over a ``0..k_{\max}`` grid (``k_{\max} = \lfloor 4(T/100)^{1/4}\rfloor``) by AIC (default) or BIC. Standard errors default to the long-run-variance–corrected OLS covariance (`dols_se=:lrv`), where the HAC long-run variance of the DOLS residual replaces ``\sigma^2``; `dols_se=:robust` uses a Newey–West sandwich instead.

```@example cointreg
md22 = estimate_cointreg(y, x; method=:dols, leads=2, lags=2)
ma   = estimate_cointreg(y, x; method=:dols, leads=:auto, lags=:auto, ic=:aic)
(dols_2_2 = round.(coef(md22), digits=4),
 auto = round.(coef(ma), digits=4), auto_leads_lags = (ma.leads, ma.lags))
```

DOLS(2, 2) gives a slope of ``1.5304`` and the AIC-selected DOLS(4, 2) gives ``1.5271`` — both closer to the true ``1.5`` than uncorrected OLS, and in the same neighbourhood as FMOLS. The AIC picks four leads and two lags, an asymmetry that reflects the DGP: the error ``u_t`` loads on the *contemporaneous* innovation ``v_t`` and then decays, so future ``\Delta x`` carries more information about ``u_t`` than the distant past does. The augmentation costs degrees of freedom but also soaks up the serial correlation that inflates the FMOLS long-run variance, so at this sample size the DOLS standard error on the slope (``0.0155``) is marginally *tighter* than the FMOLS one (``0.0164``) rather than wider.

DOLS with zero leads and lags reduces exactly to OLS on the levels — a useful sanity check:

```@example cointreg
m0 = estimate_cointreg(y, x; method=:dols, leads=0, lags=0)
b_ols = hcat(ones(length(y)), x) \ y
round(maximum(abs.(coef(m0) .- b_ols)), sigdigits=2)
```

The two agree to ``4 \times 10^{-15}``, machine precision for a problem of this conditioning. This is the boundary case of the DOLS formula: with no leads or lags the augmentation block is empty and the regression collapses to ``y_t`` on ``[D_t\ x_t]``. It also makes the bias visible — `m0` is the biased estimator, and the distance between `coef(m0)` and `coef(m)` above measures exactly what the fully-modified correction removes.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:fmols` | Estimator (`:fmols`, `:ccr`, `:dols`) |
| `trend` | `Symbol` | `:const` | Deterministics (`:none`, `:const`, `:linear`) |
| `kernel` | `Symbol` | `:bartlett` | HAC kernel for the long-run covariances (`:bartlett`, `:parzen`, `:qs`, `:tukey_hanning`) |
| `bandwidth` | `Symbol` or `Real` | `:andrews` | Plug-in rule (`:andrews`, `:nw94`) or a fixed truncation lag |
| `leads` | `Symbol` or `Integer` | `:auto` | DOLS leads of ``\Delta x``, or `:auto` for IC selection |
| `lags` | `Symbol` or `Integer` | `:auto` | DOLS lags of ``\Delta x``, or `:auto` for IC selection |
| `ic` | `Symbol` | `:aic` | Criterion for automatic lead/lag selection (`:aic` or `:bic`) |
| `dols_se` | `Symbol` | `:lrv` | DOLS standard errors: long-run-variance–corrected (`:lrv`) or Newey–West sandwich (`:robust`) |

---

## Deterministics and stored covariance pieces

The `trend` keyword selects the deterministic block prepended to the ``I(1)`` regressors: `:none`, `:const` (default), or `:linear` (constant plus a linear trend). The fitted model stores the long-run covariance pieces of the stacked ``(u, \Delta x)`` process in the ordering consumed by downstream cointegration stability and panel-cointegration tests — row/column ``1`` is the equation residual, the remaining rows/columns are ``\Delta x``.

```@example cointreg
ml = estimate_cointreg(y, x; method=:fmols, trend=:linear)
(names = ml.varnames, Omega = round.(ml.Omega, digits=4),
 identity_gap = round(maximum(abs.(ml.Omega .- (ml.Lambda .+ ml.Lambda' .- ml.Sigma))),
                      sigdigits=2))
```

Adding a linear trend leaves the slope essentially unchanged at ``1.5537`` and the trend coefficient itself is insignificant (``t = -1.78``), which is the expected outcome on a DGP with no deterministic drift — a reassurance that the deterministic specification is not driving the cointegrating vector. The stored ``\hat\Omega`` has ``\hat\Omega_{uu} = 2.520`` on the diagonal against a contemporaneous ``\hat\Sigma_{uu} = 1.457``, the gap being the accumulated autocovariance of the AR(1) error. The off-diagonal ``\hat\Omega_{u\Delta x} = 0.751`` is the endogeneity that FMOLS corrects for; it would be zero if ``x_t`` were strictly exogenous. The final line verifies the one-sided/two-sided identity ``\Omega = \Lambda + \Lambda' - \Gamma_0`` to ``4 \times 10^{-16}``.

| Field | Type | Description |
|-------|------|-------------|
| `coef` | `Vector{T}` | Corrected long-run coefficients, ordered `[deterministics; stochastic]` |
| `vcov` | `Matrix{T}` | Long-run covariance of `coef`, ``(d+k) \times (d+k)`` |
| `residuals` / `fitted` | `Vector{T}` | ``y - [D\ X]\hat\theta`` and its complement |
| `varnames` | `Vector{String}` | Coefficient labels, length ``d+k`` |
| `method` / `trend` / `kernel` | `Symbol` | Estimator, deterministics, and HAC kernel used |
| `bandwidth` | `T` | The resolved truncation-lag bandwidth actually applied |
| `leads` / `lags` | `Int` | DOLS leads and lags (both `0` for FMOLS and CCR) |
| `Omega` | `Matrix{T}` | Two-sided long-run covariance ``\hat\Omega`` of ``(u, \Delta x)`` |
| `Lambda` | `Matrix{T}` | One-sided long-run covariance ``\hat\Lambda = \sum_{j\ge0}\hat\Gamma_j`` |
| `Sigma` | `Matrix{T}` | Contemporaneous covariance ``\hat\Sigma = \hat\Gamma_0`` |
| `omega_uv` | `T` | Conditional long-run variance ``\hat\omega_{u\cdot\Delta x}`` |
| `nobs` / `d` / `k` | `Int` | Level observations, deterministic columns, stochastic regressors |

---

## [Panel cointegrating regression](@id cointreg_panel)

When the same cointegrating relationship holds across a panel of ``N`` units,
`estimate_xtcointreg` estimates each unit with `estimate_cointreg` and aggregates the
per-unit long-run coefficients. Two poolings are available:

- **Group-mean** (`pooling=:group`, the between-dimension estimator of Pedroni 2001 /
  Mark–Sul 2003): the point estimate is the arithmetic mean of the per-unit coefficient
  vectors, ``\bar\beta = N^{-1}\sum_i \hat\beta_i``, and the reported ``t``-statistic is
  Pedroni's ``N^{-1/2}\sum_i t_i`` (the average of the per-unit ``t``-ratios, **not** the
  ``t``-ratio of ``\bar\beta``). It is robust to cross-unit heterogeneity in the short-run
  dynamics and endogeneity.
- **Pooled** (`pooling=:pooled`, the within-dimension estimator): fixed effects (and, for
  DOLS, unit-specific lead/lag dynamics) are partialled out per unit and the corrected
  moments are pooled into one common slope. Pooled FMOLS (Pedroni 2000) weights each unit by
  its inverse conditional long-run variance ``\hat L_{11i}^{-2}``; pooled DOLS is the
  Kao–Chiang (2000) stacked within-demeaned regression.

```@setup cointreg
using DataFrames
# A fixed-seed heterogeneous cointegrated panel: y_it = a_i + 1.5 x_it + u_it,
# common slope, unit-specific intercepts / dynamics / endogeneity.
let
    global paneldf
    rng = MersenneTwister(20260716)
    Np, Tp = 5, 80
    yv = Float64[]; xv = Float64[]; idv = Int[]; tv = Int[]
    for i in 1:Np
        vv = randn(rng, Tp); ee = randn(rng, Tp); xi = cumsum(vv)
        ui = zeros(Tp)
        for t in 2:Tp
            ui[t] = (0.2 + 0.05i) * ui[t-1] + ee[t] + (0.3 + 0.05i) * vv[t]
        end
        yi = (1.0 + 0.5i) .+ 1.5 .* xi .+ ui
        append!(yv, yi); append!(xv, xi); append!(idv, fill(i, Tp)); append!(tv, 1:Tp)
    end
    paneldf = DataFrame(country=idv, year=tv, ly=yv, lx=xv)
end
```

**Group-mean FMOLS** across the panel:

```@example cointreg
pd = xtset(paneldf, :country, :year)
mg = estimate_xtcointreg(pd, :ly, :lx; method=:fmols, pooling=:group, trend=:const)
report(mg)
```

The group-mean slope is ``1.4870`` with a standard error of ``0.0154``, recovering the common ``1.5`` imposed on all five units. Each unit is fitted separately, so the per-unit slopes in `mg.unit_coefs` scatter between ``1.460`` and ``1.532`` — heterogeneity that the between-dimension estimator averages away without imposing homogeneity on the short-run dynamics or the endogeneity structure, both of which differ by unit in this DGP. The reported ``z`` of ``96.6`` is Pedroni's ``N^{-1/2}\sum_i t_i``, not the ``t``-ratio of the mean, so it aggregates the evidence in each unit's own fit rather than treating ``\bar\beta`` as a single estimate.

**Pooled FMOLS** reports the common slope only, with unit fixed effects removed:

```@example cointreg
mp = estimate_xtcointreg(pd, :ly, :lx; method=:fmols, pooling=:pooled, trend=:const)
mdp = estimate_xtcointreg(pd, :ly, :lx; method=:dols, pooling=:pooled, trend=:const)
(pooled_fmols = round(coef(mp)[1], digits=4), pooled_dols = round(coef(mdp)[1], digits=4),
 group_mean = round(coef(mg)[end], digits=4))
```

Pooled FMOLS returns ``1.4800`` and pooled DOLS (Kao–Chiang, with automatic per-unit lead/lag selection) returns ``1.4764``, both within ``0.024`` of the truth and of the group-mean estimate. The pooled estimators report **only** the slope: the ``N`` unit intercepts are partialled out rather than estimated, which is why `coef(mp)` has one element while the group-mean `coef(mg)` has two. When the slope really is common, pooling is more efficient — the pooled FMOLS standard error of ``0.0140`` beats the group-mean ``0.0154``. When it is not, pooling estimates an uninterpretable average and the group-mean estimator is the safer default.

The group-mean estimate is the exact mean of the per-unit fits — a useful identity for
verification:

```@example cointreg
per = [estimate_cointreg(gd.data[:, 1], gd.data[:, 2]; method=:fmols, trend=:const)
       for gd in (MacroEconometricModels.group_data(pd, g) for g in 1:pd.n_groups)]
mean_of_units = sum(coef.(per)) ./ pd.n_groups
isapprox(coef(mg), mean_of_units; atol=1e-10)
```

Refitting each of the five units by hand and averaging reproduces `coef(mg)` to within ``10^{-10}``, confirming that `pooling=:group` is exactly the arithmetic mean of the per-unit coefficient vectors and nothing more. The aggregation adds no weighting, no shrinkage, and no re-estimation. That transparency is the group-mean estimator's main attraction: any unit whose fit is suspect can be identified in `mg.unit_models` and excluded by hand.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:fmols` | Per-unit estimator (`:fmols` or `:dols`; CCR has no panel form) |
| `pooling` | `Symbol` | `:group` | `:group` (between-dimension) or `:pooled` (within-dimension) |
| `trend` | `Symbol` | `:const` | Per-unit deterministics (`:none`, `:const`, `:linear`) |
| `kernel` | `Symbol` | `:bartlett` | HAC kernel for the per-unit long-run covariances (`:bartlett`, `:parzen`, `:qs`, `:tukey_hanning`) |
| `bandwidth` | `Symbol` or `Real` | `:andrews` | Plug-in rule (`:andrews`, `:nw94`) or a fixed truncation lag |
| `leads` / `lags` | `Symbol` or `Integer` | `:auto` | Per-unit DOLS leads and lags |
| `ic` | `Symbol` | `:aic` | Criterion for automatic per-unit lead/lag selection |
| `dols_se` | `Symbol` | `:lrv` | Per-unit DOLS standard-error flavour |

---

## Complete Example

```@example cointreg
using MacroEconometricModels, Random

# Fixed-seed two-regressor cointegrated system: y = 1 + 0.8 x₁ − 0.5 x₂ + u
Random.seed!(11)
n = 220
x1 = cumsum(randn(n))
x2 = cumsum(randn(n))
uu = zeros(n)
for t in 2:n
    uu[t] = 0.3 * uu[t-1] + randn()
end
yy = 1.0 .+ 0.8 .* x1 .- 0.5 .* x2 .+ uu
X = hcat(x1, x2)

# 1. FMOLS on the multi-regressor system
mm = estimate_cointreg(yy, X; method=:fmols)
report(mm)
```

```@example cointreg
# 2. Compare all three estimators' slope block (true = [0.8, -0.5])
slopes = [round.(coef(estimate_cointreg(yy, X; method=meth))[2:3], digits=4)
          for meth in (:fmols, :ccr, :dols)]
(fmols = slopes[1], ccr = slopes[2], dols = slopes[3])
```

All three estimators recover the true slope vector ``(0.8, -0.5)`` to within ``0.035``, and agree with each other to within ``0.005``: FMOLS gives ``(0.7792, -0.5323)``, CCR ``(0.7793, -0.5322)``, and DOLS ``(0.7752, -0.5337)``. The near-identity of FMOLS and CCR is their asymptotic equivalence showing up at ``T = 220``; DOLS differs a little more because it spends degrees of freedom on the augmentation block. Agreement across three estimators built on different corrections is the practical evidence that the cointegrating vector is well identified — systematic disagreement would point to a failure of the ``I(1)``-and-cointegrated premise rather than to a bad estimator.

---

## Common Pitfalls

1. **The regressors must be ``I(1)`` and cointegrated with ``y``.** FMOLS/CCR/DOLS assume a genuine cointegrating relationship; applied to unrelated ``I(1)`` series they estimate a spurious "long-run" vector. Pre-test for cointegration first — the Johansen test lives on [Unit Root & Cointegration](@ref tests_unitroot_page), the residual-based tests on [Residual-Based Cointegration](@ref tests_cointegration_page), and the Gregory–Hansen test with a break on [Structural Breaks](@ref tests_breaks_page). If some regressors are ``I(0)``, use [ARDL & Bounds Testing](@ref ardl_page) instead.

2. **`X` carries no deterministic column.** The intercept and trend are added internally via `trend`; passing a constant column double-counts the intercept.

3. **The intercept on ``I(1)`` regressors converges slowly.** Only the slope block is superconsistent for the cointegrating vector; the deterministic coefficients converge more slowly, so do not over-interpret a noisy intercept in short samples.

4. **Bandwidth and kernel conventions matter for exact replication.** This package uses the Newey–West normalisation ``1 - j/(b+1)`` (lags ``j = 1..b``); some references (e.g. R's `cointReg`) use ``1 - j/b``. The two coincide when this package's `bandwidth = b` equals the other's ``b+1``.

5. **DOLS needs enough sample for the augmentation.** Large `leads`/`lags` on a short series exhaust the degrees of freedom; automatic selection (`:auto`) caps the grid at ``k_{\max} = \lfloor 4(T/100)^{1/4}\rfloor`` and guards against an over-parameterised fit.

6. **Pooled panel estimates report the slope only.** `pooling=:pooled` partials out the ``N`` unit intercepts rather than estimating them, so `coef` is shorter than under `pooling=:group`. Indexing a pooled `coef` as if it began with a deterministic block reads the wrong element.

---

## References

```@example cointreg
refs(m)
```

- Phillips, P. C. B., & Hansen, B. E. (1990). Statistical Inference in Instrumental Variables Regression with I(1) Processes.
  *Review of Economic Studies*, 57(1), 99-125. [DOI](https://doi.org/10.2307/2297545)

- Park, J. Y. (1992). Canonical Cointegrating Regressions.
  *Econometrica*, 60(1), 119-143. [DOI](https://doi.org/10.2307/2951679)

- Saikkonen, P. (1991). Asymptotically Efficient Estimation of Cointegration Regressions.
  *Econometric Theory*, 7(1), 1-21. [DOI](https://doi.org/10.1017/S0266466600004217)

- Stock, J. H., & Watson, M. W. (1993). A Simple Estimator of Cointegrating Vectors in Higher Order Integrated Systems.
  *Econometrica*, 61(4), 783-820. [DOI](https://doi.org/10.2307/2951763)

- Pedroni, P. (2000). Fully Modified OLS for Heterogeneous Cointegrated Panels.
  *Advances in Econometrics*, 15, 93-130. [DOI](https://doi.org/10.1016/S0731-9053(00)15004-2)

- Pedroni, P. (2001). Purchasing Power Parity Tests in Cointegrated Panels.
  *Review of Economics and Statistics*, 83(4), 727-731. [DOI](https://doi.org/10.1162/003465301753237803)

- Kao, C., & Chiang, M.-H. (2000). On the Estimation and Inference of a Cointegrated Regression in Panel Data.
  *Advances in Econometrics*, 15, 179-222. [DOI](https://doi.org/10.1016/S0731-9053(00)15007-8)

- Mark, N. C., & Sul, D. (2003). Cointegration Vector Estimation by Panel DOLS and Long-Run Money Demand.
  *Oxford Bulletin of Economics and Statistics*, 65(5), 655-680. [DOI](https://doi.org/10.1111/j.1468-0084.2003.00066.x)
