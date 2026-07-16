# [Cointegrating Regression (FMOLS / CCR / DOLS)](@id cointreg_page)

**MacroEconometricModels.jl** estimates a single cointegrating vector by three asymptotically-efficient, endogeneity-corrected estimators: fully-modified OLS (Phillips–Hansen 1990), canonical cointegrating regression (Park 1992), and dynamic OLS (Saikkonen 1991 / Stock–Watson 1993). When ``y_t`` and an ``I(1)`` regressor vector ``x_t`` are cointegrated, plain OLS on the levels is super-consistent but has an asymptotically biased, non-standard distribution because ``x_t`` is correlated with the equation error ``u_t`` and ``u_t`` is serially correlated. All three estimators remove that second-order bias and deliver a mixed-normal long-run coefficient vector on which standard Wald inference is valid.

- **FMOLS** — `estimate_cointreg(y, X; method=:fmols)` corrects the regressand for regressor endogeneity (``y^+_t = y_t - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\Delta x_t``) and applies a serial-correlation bias adjustment built from the one-sided long-run covariance
- **CCR** — `method=:ccr` transforms the *data* (Park's canonical transformation) so that plain OLS on the transformed system is efficient; asymptotically equivalent to FMOLS
- **DOLS** — `method=:dols` augments the levels regression with leads and lags of ``\Delta x_t``; the level-regressor coefficients are the efficient long-run estimates, with automatic AIC/BIC lead/lag selection
- **Reused long-run-variance toolkit** — all three build ``\hat\Omega`` (two-sided), ``\hat\Lambda`` (one-sided) and ``\hat\Sigma = \hat\Gamma_0`` of the stacked ``(u, \Delta x)`` process on the EV-12 `lrcov`/`lrcov_oneside` HAC estimators
- **Stored covariance pieces** — the fitted [`CointRegModel`](@ref) exposes ``\hat\Omega``, ``\hat\Lambda``, ``\hat\Sigma`` and the conditional long-run variance ``\hat\omega_{u\cdot\Delta x}`` for downstream cointegration stability and panel-cointegration tests

- **Panel extension** — `estimate_xtcointreg(pd, y, xs...)` aggregates the single-unit estimator across the ``N`` units of a [`PanelData`](@ref) into group-mean (Pedroni 2001) or pooled (Pedroni 2000 FMOLS / Kao–Chiang 2000 DOLS) panel FMOLS/DOLS; see [Panel cointegrating regression](@ref cointreg_panel).

All estimators return a [`CointRegModel`](@ref) and integrate with `report` and `refs`.

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
println("CCR long-run coefficients: ", round.(coef(mc), digits=4))
```

**Recipe 3: Dynamic OLS with automatic lead/lag selection**

```@example cointreg
md = estimate_cointreg(y, x; method=:dols, leads=:auto, lags=:auto, ic=:aic)
println("selected leads = ", md.leads, ", lags = ", md.lags)
println("DOLS long-run coefficients: ", round.(coef(md), digits=4))
```

**Recipe 4: Choose the HAC kernel and bandwidth**

```@example cointreg
# Andrews (1991) plug-in bandwidth with a Parzen kernel
mk = estimate_cointreg(y, x; method=:fmols, kernel=:parzen, bandwidth=:andrews)
println("resolved bandwidth = ", round(mk.bandwidth, digits=3))
```

---

## Fully-modified OLS

`estimate_cointreg(y, X; method=:fmols, ...)` implements Phillips & Hansen (1990). Starting from the OLS-on-levels residual ``\hat u_t``, it forms the stacked process ``\xi_t = (\hat u_t, \Delta x_t')'`` and estimates its long-run covariance ``\hat\Omega`` (two-sided) and one-sided ``\hat\Lambda = \sum_{j\ge 0}\hat\Gamma_j`` on the reused `lrcov`/`lrcov_oneside` toolkit. Partitioning ``\hat\Omega`` into the ``u`` and ``\Delta x`` blocks, the endogeneity-corrected regressand and the serial-correlation bias term give the fully-modified estimator

```math
\hat\theta^{+} = \left(Z'Z\right)^{-1}\left(Z'y^{+} - T\begin{bmatrix}0\\ \hat\Delta^{+}_{\Delta x u}\end{bmatrix}\right),\qquad
y^{+}_t = y_t - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\Delta x_t,
```

where ``Z = [D\ X]`` stacks the deterministics and the ``I(1)`` regressors. The coefficient covariance is ``\hat\omega_{u\cdot\Delta x}(Z'Z)^{-1}`` with the conditional long-run variance ``\hat\omega_{u\cdot\Delta x} = \hat\Omega_{uu} - \hat\Omega_{u\Delta x}\hat\Omega_{\Delta x\Delta x}^{-1}\hat\Omega_{\Delta x u}``.

```@example cointreg
m = estimate_cointreg(y, x; method=:fmols)
println("θ⁺        = ", round.(coef(m), digits=4))
println("std. err. = ", round.(stderror(m), digits=4))
println("ω_{u·Δx}  = ", round(m.omega_uv, digits=4))
```

---

## Canonical cointegrating regression

Park's (1992) CCR (`method=:ccr`) reaches the same efficient limit by transforming the data rather than the estimator. Using ``\hat\Sigma = \hat\Gamma_0``, ``\hat\Omega`` and ``\hat\Lambda``, it builds transformed regressors ``x^{*}_t`` and regressand ``y^{*}_t`` such that plain OLS of ``y^{*}`` on ``[D\ x^{*}]`` is free of endogeneity and serial-correlation bias. CCR and FMOLS are asymptotically equivalent, so their point estimates and conditional long-run variance agree closely in finite samples.

```@example cointreg
mf = estimate_cointreg(y, x; method=:fmols)
mc = estimate_cointreg(y, x; method=:ccr)
println("FMOLS θ = ", round.(coef(mf), digits=4))
println("CCR   θ = ", round.(coef(mc), digits=4))
println("ω_{u·Δx}: FMOLS = ", round(mf.omega_uv, digits=4),
        ", CCR = ", round(mc.omega_uv, digits=4))
```

---

## Dynamic OLS

DOLS (`method=:dols`) augments the levels regression with the contemporaneous value, ``\text{lags}`` lags, and ``\text{leads}`` leads of ``\Delta x_t`` (Saikkonen 1991; Stock–Watson 1993):

```math
y_t = D_t'\delta + x_t'\beta + \sum_{j=-\text{leads}}^{\text{lags}} \Delta x_{t-j}'\gamma_j + u^{*}_t .
```

The level-regressor coefficients ``(\delta, \beta)`` are the efficient long-run estimates. Leads and lags may be fixed, or selected automatically over a ``0..k_{\max}`` grid (``k_{\max} = \lfloor 4(T/100)^{1/4}\rfloor``) by AIC (default) or BIC. Standard errors default to the long-run-variance–corrected OLS covariance (`dols_se=:lrv`), where the HAC long-run variance of the DOLS residual replaces ``\sigma^2``; `dols_se=:robust` uses a Newey–West sandwich instead.

```@example cointreg
md = estimate_cointreg(y, x; method=:dols, leads=2, lags=2)
println("DOLS(2,2) θ = ", round.(coef(md), digits=4))

# Automatic selection by AIC
ma = estimate_cointreg(y, x; method=:dols, leads=:auto, lags=:auto, ic=:aic)
println("auto leads/lags = (", ma.leads, ", ", ma.lags, ")")
```

DOLS with zero leads and lags reduces exactly to OLS on the levels — a useful sanity check:

```@example cointreg
m0 = estimate_cointreg(y, x; method=:dols, leads=0, lags=0)
b_ols = hcat(ones(length(y)), x) \ y
println("DOLS(0,0) − OLS = ", round(maximum(abs.(coef(m0) .- b_ols)), sigdigits=2))
```

---

## Deterministics and stored covariance pieces

The `trend` keyword selects the deterministic block prepended to the ``I(1)`` regressors: `:none`, `:const` (default), or `:linear` (constant plus a linear trend). The fitted model stores the long-run covariance pieces of the stacked ``(u, \Delta x)`` process in the ordering consumed by downstream cointegration stability and panel-cointegration tests — row/column ``1`` is the equation residual, the remaining rows/columns are ``\Delta x``.

```@example cointreg
m = estimate_cointreg(y, x; method=:fmols, trend=:linear)
println("coefficient names: ", m.varnames)
println("Ω =\n", round.(m.Omega, digits=4))
# One-sided/two-sided identity: Ω = Λ + Λ' − Γ₀
println("‖Ω − (Λ + Λ' − Σ)‖ = ",
        round(maximum(abs.(m.Omega .- (m.Lambda .+ m.Lambda' .- m.Sigma))), sigdigits=2))
```

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
m = estimate_cointreg(yy, X; method=:fmols)
report(m)
```

```@example cointreg
# 2. Compare all three estimators' slope block (true = [0.8, -0.5])
for meth in (:fmols, :ccr, :dols)
    mm = estimate_cointreg(yy, X; method=meth)
    println(rpad(string(meth), 6), " slopes = ", round.(coef(mm)[2:3], digits=4))
end
```

```@example cointreg
# 3. Confidence intervals for the long-run coefficients
ci = confint(m; level=0.95)
println("95% CI:\n", round.(ci, digits=4))
```

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

**Pooled FMOLS** reports the common slope only, with unit fixed effects removed:

```@example cointreg
mp = estimate_xtcointreg(pd, :ly, :lx; method=:fmols, pooling=:pooled, trend=:const)
println("pooled slope = ", round(coef(mp)[1], digits=4), " (true = 1.5)")
```

**Pooled DOLS** (Kao–Chiang), with automatic per-unit lead/lag selection:

```@example cointreg
md = estimate_xtcointreg(pd, :ly, :lx; method=:dols, pooling=:pooled, trend=:const)
println("pooled DOLS slope = ", round(coef(md)[1], digits=4))
```

The group-mean estimate is the exact mean of the per-unit fits — a useful identity for
verification:

```@example cointreg
per = [estimate_cointreg(gd.data[:, 1], gd.data[:, 2]; method=:fmols, trend=:const)
       for gd in (MacroEconometricModels.group_data(pd, g) for g in 1:pd.n_groups)]
mean_of_units = sum(coef.(per)) ./ pd.n_groups
println("β̄ matches mean of per-unit coefs: ",
        isapprox(coef(mg), mean_of_units; atol=1e-10))
```

---

## Common Pitfalls

- **The regressors must be ``I(1)`` and cointegrated with ``y``.** FMOLS/CCR/DOLS assume a genuine cointegrating relationship; applied to unrelated ``I(1)`` series they estimate a spurious "long-run" vector. Pre-test for cointegration (e.g. the Gregory–Hansen or Johansen tests on the [Unit Root & Cointegration](@ref tests_unitroot_page) page) first.
- **`X` carries no deterministic column.** The intercept and trend are added internally via `trend`; passing a constant column double-counts the intercept.
- **The intercept on ``I(1)`` regressors converges slowly.** Only the slope block is ``\sqrt{T}``-superconsistent for the cointegrating vector; the deterministic coefficients converge more slowly, so do not over-interpret a noisy intercept in short samples.
- **Bandwidth and kernel conventions matter for exact replication.** This package uses the Newey–West normalisation ``1 - j/(b+1)`` (lags ``j = 1..b``); some references (e.g. R's `cointReg`) use ``1 - j/b``. The two coincide when this package's `bandwidth = b` equals the other's ``b+1``.
- **DOLS needs enough sample for the augmentation.** Large `leads`/`lags` on a short series exhaust the degrees of freedom; automatic selection (`:auto`) caps the grid at ``k_{\max} = \lfloor 4(T/100)^{1/4}\rfloor`` and guards against an over-parameterised fit.

---

## References

```@example cointreg
refs(m)
```

- Phillips, P. C. B. & Hansen, B. E. (1990). Statistical Inference in Instrumental Variables Regression with I(1) Processes. *Review of Economic Studies* 57(1), 99–125.
- Park, J. Y. (1992). Canonical Cointegrating Regressions. *Econometrica* 60(1), 119–143.
- Saikkonen, P. (1991). Asymptotically Efficient Estimation of Cointegration Regressions. *Econometric Theory* 7(1), 1–21.
- Stock, J. H. & Watson, M. W. (1993). A Simple Estimator of Cointegrating Vectors in Higher Order Integrated Systems. *Econometrica* 61(4), 783–820.
- Pedroni, P. (2000). Fully Modified OLS for Heterogeneous Cointegrated Panels. *Advances in Econometrics* 15, 93–130.
- Pedroni, P. (2001). Purchasing Power Parity Tests in Cointegrated Panels. *Review of Economics and Statistics* 83(4), 727–731.
- Kao, C. & Chiang, M.-H. (2000). On the Estimation and Inference of a Cointegrated Regression in Panel Data. *Advances in Econometrics* 15, 179–222.
- Mark, N. C. & Sul, D. (2003). Cointegration Vector Estimation by Panel DOLS and Long-Run Money Demand. *Oxford Bulletin of Economics and Statistics* 65(5), 655–680.
