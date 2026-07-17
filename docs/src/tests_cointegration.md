# [Residual-Based Cointegration Tests](@id tests_cointegration_page)

**MacroEconometricModels.jl** provides four single-equation cointegration tests that operate on the residuals (or the coefficient path) of a static cointegrating regression ``y_t = D_t'\delta + x_t'\beta + u_t`` with an ``I(1)`` regressor vector ``x_t``. Two are residual-unit-root tests of the null of **no cointegration**, and two are parameter-based tests whose null is **genuine, stable cointegration**.

- **Engle–Granger** — `engle_granger_test(y, X)` runs an augmented Dickey–Fuller regression on the levels residuals; the ``t``-statistic is compared to the MacKinnon (1996/2010) cointegration surface indexed by the number of ``I(1)`` series
- **Phillips–Ouliaris** — `phillips_ouliaris_test(y, X)` forms the semiparametric (Phillips–Perron-style) normalized-bias ``\hat Z_\alpha`` and ``t``-ratio ``\hat Z_t`` on the residual AR(1) root, reusing the EV-12 long-run-variance toolkit
- **Hansen ``L_c``** — `hansen_instability_test(m)` tests a fitted [`CointRegModel`](@ref) for coefficient stability; a large ``L_c`` signals parameter drift (observationally equivalent to no cointegration)
- **Park ``H(p,q)``** — `park_added_test(m)` adds superfluous deterministic trends to the cointegrating regression; under genuine cointegration their coefficients are zero and the statistic is ``\chi^2``, but under a spurious regression it diverges

!!! note "The cointegration p-value trap"
    Residual-based cointegration statistics do **not** follow the univariate Dickey–Fuller distribution. Because the residuals are estimated, their null distribution depends on ``N = k+1`` — the number of ``I(1)`` series in the cointegrating vector. This package indexes the MacKinnon response surface by ``N``; using the plain ADF surface would badly under-reject.

```@setup coint
using MacroEconometricModels, Random
Random.seed!(20260716)
T = 200
v = randn(T); e = randn(T)
x = cumsum(v)                                   # I(1) regressor (random walk)
u = zeros(T)
for t in 2:T
    u[t] = 0.4 * u[t-1] + e[t]                  # I(0) equation error -> cointegration
end
y = 1.0 .+ 2.0 .* x .+ u                        # cointegrated pair, vector (2.0)
# An independent, non-cointegrated pair for contrast.
xn = cumsum(randn(T)); yn = cumsum(randn(T))
```

## Quick Start

**Recipe 1: Engle–Granger two-step test**

```@example coint
# H0: no cointegration. A small p-value rejects in favour of cointegration.
report(engle_granger_test(y, x; trend=:constant))
```

**Recipe 2: Phillips–Ouliaris ``\hat Z_t`` / ``\hat Z_\alpha``**

```@example coint
report(phillips_ouliaris_test(y, x; trend=:constant))
```

**Recipe 3: Contrast with an independent random-walk pair**

```@example coint
r = engle_granger_test(yn, xn)
println("independent pair: ADF = ", round(r.statistic, digits=3),
        ", p = ", round(r.pvalue, digits=3))
```

**Recipe 4: Hansen ``L_c`` stability test on a fitted cointegrating regression**

```@example coint
m = estimate_cointreg(y, x; method=:fmols, trend=:const)
report(hansen_instability_test(m))
```

**Recipe 5: Park ``H(p,q)`` spurious-regression test**

```@example coint
report(park_added_test(m; q_add=2))
```

---

## Engle–Granger two-step test

Engle & Granger (1987) test the null of **no cointegration** in two steps. Stage 1 estimates the static regression by OLS in levels; stage 2 runs an augmented Dickey–Fuller regression with **no deterministic term** on the residuals ``\hat u_t``,

```math
\Delta \hat u_t = \rho\, \hat u_{t-1} + \sum_{j=1}^{p} \gamma_j\, \Delta \hat u_{t-j} + \varepsilon_t,
```

and reports the ``t``-statistic on ``\rho``. Rejection of the residual unit root (``\rho = 0``) is evidence of cointegration.

```@example coint
res = engle_granger_test(y, x; trend=:constant, lags=:aic)
println("ADF statistic = ", round(res.statistic, digits=4))
println("lags selected = ", res.lags, ",  N = k+1 = ", res.N)
println("MacKinnon p-value = ", round(res.pvalue, digits=4))
```

- `trend`: deterministics in the cointegrating regression — `:none`, `:constant`, or `:trend`. This selects the MacKinnon surface (`n`/`c`/`ct`).
- `lags`: augmenting lags `p`, or `:aic`/`:bic` to select over `0:max_lags`.

The p-value comes from the MacKinnon (2010) cointegration response surface indexed by ``N = k+1``, matching the Engle–Granger test in Stata (`egranger`) and statsmodels (`coint`).

---

## Phillips–Ouliaris test

Phillips & Ouliaris (1990) replace the parametric ADF augmentation with a semiparametric (Phillips–Perron-style) correction on the residual AR(1) root ``\hat\rho``. With short-run variance ``s^2`` and long-run variance ``\omega^2`` of the AR(1) innovations,

```math
\hat Z_\alpha = T(\hat\rho - 1) - \tfrac{1}{2}(\omega^2 - s^2)\, \frac{T^2}{\sum \hat u_{t-1}^2},
\qquad
\hat Z_t = \sqrt{\tfrac{s^2}{\omega^2}}\, t_{\hat\rho} - \tfrac{(\omega^2 - s^2)}{2\,\omega\,(T^{-1}\sum \hat u_{t-1}^2)^{1/2}\sqrt{T}}.
```

```@example coint
po = phillips_ouliaris_test(y, x; trend=:constant, kernel=:bartlett)
println("Z_t = ", round(po.statistic, digits=3), " (p = ", round(po.pvalue, digits=3), ")")
println("Z_a = ", round(po.z_alpha, digits=2), " (p = ", round(po.z_alpha_pvalue, digits=3), ")")
```

The ``\hat Z_t`` p-value shares the MacKinnon cointegration surface with Engle–Granger. The normalized-bias ``\hat Z_\alpha`` has no closed-form surface, so its p-value brackets Monte-Carlo critical values (validated against the Phillips–Ouliaris 1990 tables). The long-run variance ``\omega^2`` is built with the EV-12 `lrvar` HAC toolkit (`kernel` and `bandwidth` keywords are forwarded).

---

## Hansen ``L_c`` parameter-instability test

Hansen (1992) tests a fitted cointegrating regression for coefficient **stability**. The null is cointegration with constant coefficients; the alternative is that the loadings follow a martingale (equivalently, no cointegration). With cumulative scores ``\hat S_t = \sum_{i\le t} Z_i \hat u_i`` and the stored conditional long-run variance ``\hat\omega^2_{u\cdot v}``,

```math
L_c = \hat\omega_{u\cdot v}^{-2}\; T^{-1} \sum_{t=1}^T \hat S_t' \left(\sum_i Z_i Z_i'\right)^{-1} \hat S_t.
```

```@example coint
m = estimate_cointreg(y, x; method=:fmols, trend=:const)
lc = hansen_instability_test(m)
println("Lc = ", round(lc.statistic, digits=4), ",  p = ", round(lc.pvalue, digits=3))
```

A **large** ``L_c`` rejects stability. Critical values are Monte-Carlo quantiles indexed by the deterministic case and the number of ``I(1)`` regressors, spot-checked against Hansen (1992) Table 1.

---

## Park ``H(p,q)`` added-variables test

Park (1990) augments the cointegrating regression with ``q_{\mathrm{add}}`` superfluous normalized-time trends ``(t/T)^{p+1}, \dots, (t/T)^{p+q_{\mathrm{add}}}`` and tests that their coefficients are jointly zero, using a long-run-variance-corrected Wald statistic

```math
H(p,q) = \hat\gamma'\left(\hat\omega^2\,[(Z'Z)^{-1}]_{AA}\right)^{-1}\hat\gamma \;\sim\; \chi^2(q_{\mathrm{add}}).
```

Under genuine cointegration the ``I(0)`` errors keep ``H`` small; under a spurious regression the ``I(1)`` errors make it diverge.

```@example coint
pk = park_added_test(m; q_add=2)
println("H(p,q) = ", round(pk.statistic, digits=3), ",  p = ", round(pk.pvalue, digits=3))
```

---

## Complete Example

```@example coint
using MacroEconometricModels, Random
Random.seed!(7)
Tn = 220
xx = cumsum(randn(Tn))
uu = zeros(Tn)
for t in 2:Tn
    uu[t] = 0.5 * uu[t-1] + 0.8 * randn()      # I(0) AR(1) error -> cointegration
end
yy = 3.0 .+ 1.2 .* xx .+ uu

eg = engle_granger_test(yy, xx; trend=:constant)
po = phillips_ouliaris_test(yy, xx; trend=:constant)
cr = estimate_cointreg(yy, xx; method=:fmols, trend=:const)
lc = hansen_instability_test(cr)
pk = park_added_test(cr; q_add=2)

println("Engle-Granger  : ADF = ", round(eg.statistic, digits=2), ", p = ", round(eg.pvalue, digits=3))
println("Phillips-Ouliaris Z_t = ", round(po.statistic, digits=2), ", p = ", round(po.pvalue, digits=3))
println("Hansen Lc      : ", round(lc.statistic, digits=3), ", p = ", round(lc.pvalue, digits=3))
println("Park H(p,q)    : ", round(pk.statistic, digits=3), ", p = ", round(pk.pvalue, digits=3))
```

---

## Common Pitfalls

1. **Do not use the ADF p-value surface.** Residual-based cointegration statistics depend on ``N = k+1``. All four tests here index the correct MacKinnon cointegration surface automatically; passing residuals to `adf_test` instead would badly under-reject.
2. **Order the columns correctly.** The convenience matrix method treats `Y[:,1]` as the dependent variable and `Y[:,2:end]` as the ``I(1)`` regressors.
3. **Null direction differs.** Engle–Granger and Phillips–Ouliaris test ``H_0``: *no* cointegration (small p ⇒ cointegration). Hansen and Park test ``H_0``: *genuine/stable* cointegration (small p ⇒ instability / spurious).
4. **Ensure the inputs are ``I(1)``.** These tests assume each series has a single unit root; pre-test with [`adf_test`](@ref) or [`kpss_test`](@ref).
5. **Kernel/bandwidth affect Phillips–Ouliaris.** The semiparametric ``\hat Z_\alpha``/``\hat Z_t`` depend on the residual long-run variance; the `kernel` and `bandwidth` keywords are forwarded to the EV-12 `lrvar` estimator.

---

## References

- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica* 55(2), 251–276.
- Phillips, P. C. B. & Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. *Econometrica* 58(1), 165–193.
- Hansen, B. E. (1992). Tests for parameter instability in regressions with I(1) processes. *Journal of Business & Economic Statistics* 10(3), 321–335.
- Park, J. Y. (1990). Testing for unit roots and cointegration by adding superfluous regressors. CAE Working Paper, Cornell University.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's University Department of Economics Working Paper 1227.
