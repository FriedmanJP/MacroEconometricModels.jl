# [Volatility Models](@id volatility_page)

**MacroEconometricModels.jl** provides a complete suite of univariate volatility models for capturing time-varying conditional variance in financial and macroeconomic time series. The package covers observation-driven (ARCH/GARCH family) and parameter-driven (stochastic volatility) approaches, with unified diagnostics, forecasting, and visualization.

- **ARCH**: Autoregressive Conditional Heteroskedasticity (Engle 1982) --- conditional variance depends on past squared innovations
- **GARCH**: Generalized ARCH (Bollerslev 1986) --- adds lagged conditional variances for parsimonious volatility persistence
- **EGARCH**: Exponential GARCH (Nelson 1991) --- log-variance specification with asymmetric leverage effects, no positivity constraints
- **GJR-GARCH**: Threshold GARCH (Glosten, Jagannathan & Runkle 1993) --- indicator-based leverage via ``\gamma_i \mathbb{1}(\varepsilon_{t-i} < 0)``
- **GARCH-MIDAS**: Mixed-frequency component GARCH (Engle, Ghysels & Sohn 2013) --- variance splits into a short-run unit-mean GARCH and a long-run MIDAS-filtered macro/realized-variance component
- **FIGARCH / FIEGARCH**: Fractionally integrated (E)GARCH (Baillie, Bollerslev & Mikkelsen 1996; Bollerslev & Mikkelsen 1996) --- hyperbolic (long-memory) volatility persistence via the fractional-difference operator ``(1-L)^d``
- **IGARCH**: Integrated GARCH (Engle & Bollerslev 1986) --- unit-persistence (``\sum\alpha + \sum\beta = 1``) volatility with a divergent unconditional variance (RiskMetrics EWMA is the ``\omega = 0`` case)
- **Component GARCH**: Permanent/transitory decomposition (Engle & Lee 1999) --- a slowly mean-reverting long-run variance trend plus a fast transitory cycle
- **APARCH**: Asymmetric Power ARCH (Ding, Granger & Engle 1993) --- a free power ``\delta`` and Box-Cox leverage term that nests GARCH, GJR-GARCH, and TARCH
- **Stochastic Volatility**: Latent log-variance AR(1) process (Taylor 1986), estimated via Kim-Shephard-Chib (1998) Gibbs sampler with optional leverage and Student-t errors
- **Diagnostics**: ARCH-LM test, Ljung-Box on squared residuals, news impact curves, Engle-Ng sign-bias and Nyblom-Hansen parameter-stability tests
- **Forecasting**: Multi-step ahead variance forecasts with simulation-based confidence intervals (GARCH family) or posterior predictive intervals (SV)

```@setup volatility
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
# FRED-MD "S&P 500", tcode 5 = monthly log returns of the composite index.
spx = filter(isfinite, to_vector(apply_tcode(fred[:, ["S&P 500"]])))
```

Every example below runs on `spx`, the 803 monthly log returns of the S&P 500 composite index in the FRED-MD panel (McCracken & Ng 2016) — a series with a sample standard deviation of 3.6% per month, a worst month of ``-22.8``%, and the volatility clustering these models exist to describe.

## Quick Start

**Recipe 1: ARCH(q)**

```@example volatility
# ARCH(3) — Engle (1982)
arch = estimate_arch(spx, 3)
report(arch)
```

**Recipe 2: GARCH(1,1)**

```@example volatility
# GARCH(1,1) — the workhorse specification
garch = estimate_garch(spx, 1, 1)
report(garch)
```

**Recipe 3: Asymmetric GARCH models**

```@example volatility
# EGARCH captures leverage without positivity constraints
egarch = estimate_egarch(spx, 1, 1)
report(egarch)
```

```@example volatility
# GJR-GARCH captures leverage via an indicator function
gjr = estimate_gjr_garch(spx, 1, 1)
report(gjr)
```

**Recipe 4: Stochastic volatility**

```@example volatility
# SV via Kim-Shephard-Chib (1998) Gibbs sampler
sv = estimate_sv(spx; n_samples=2000, burnin=1000)
report(sv)
```

**Recipe 5: ARCH-LM diagnostics**

```@example volatility
# Raw returns: H₀ of no ARCH effects is rejected
raw = arch_lm_test(spx, 5)

# GARCH standardized residuals: H₀ is no longer rejected
garch = estimate_garch(spx, 1, 1)
resid = arch_lm_test(garch, 5)

(raw_p = round(raw[2], digits=4), residual_p = round(resid[2], digits=4))
```

**Recipe 6: Volatility forecasting**

```@example volatility
garch = estimate_garch(spx, 1, 1)
fc = forecast(garch, 20; conf_level=0.95)
report(fc)
```

```julia
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_volatility.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## ARCH Models

The **Autoregressive Conditional Heteroskedasticity** (ARCH) model of Engle (1982) captures time-varying conditional variance by making the variance a function of past squared innovations. The ARCH(``q``) model is the foundation of the entire GARCH family.

```math
y_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim \mathcal{N}(0, 1)
```

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon^2_{t-i}
```

where:
- ``y_t`` is the observed time series
- ``\mu`` is the conditional mean (intercept)
- ``\varepsilon_t`` is the mean-corrected innovation
- ``\sigma^2_t`` is the conditional variance at time ``t``
- ``\omega > 0`` is the variance intercept
- ``\alpha_i \geq 0`` are the ARCH coefficients
- ``z_t`` is a standardized innovation

The process is covariance stationary when ``\sum_{i=1}^{q} \alpha_i < 1``, with unconditional variance ``\text{Var}(\varepsilon_t) = \omega / (1 - \sum_{i=1}^{q} \alpha_i)``.

!!! note "Technical Note"
    Estimation uses two-stage maximum likelihood. Stage 1 applies Nelder-Mead (derivative-free) to find a good starting region. Stage 2 refines with L-BFGS (gradient-based). Parameters are log-transformed internally to enforce positivity (``\omega > 0``, ``\alpha_i \geq 0``) without constrained optimization. Standard errors use the delta method to transform from optimization space back to the original parameter space.

```@example volatility
# Estimate ARCH(3) model
arch = estimate_arch(spx, 3)
report(arch)
```

All three ARCH coefficients are positive and ``\hat\alpha_1 = 0.176`` and ``\hat\alpha_3 = 0.204`` are significant at 1%: a large return three months ago still raises this month's variance, which is exactly the clustering a homoskedastic model cannot represent. The persistence ``\sum\hat\alpha_i = 0.471`` implies an unconditional variance ``\hat\omega/(1-\sum\hat\alpha_i) = 0.0014``, or 3.7% per month — within a rounding of the sample standard deviation, as it should be for a correctly specified model. That persistence is also the ARCH model's weakness: reproducing the observed decay of volatility with lagged squared returns alone needs many lags, which is what GARCH fixes with a single ``\beta``.

### ARCHModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `q` | `Int` | ARCH order |
| `mu` | `T` | Estimated mean (intercept) |
| `omega` | `T` | Variance intercept ``\omega`` |
| `alpha` | `Vector{T}` | ARCH coefficients ``[\alpha_1, \ldots, \alpha_q]`` |
| `conditional_variance` | `Vector{T}` | Estimated ``\hat{\sigma}^2_t`` at each ``t`` |
| `standardized_residuals` | `Vector{T}` | ``\hat{z}_t = \hat{\varepsilon}_t / \hat{\sigma}_t`` |
| `residuals` | `Vector{T}` | Raw residuals ``\hat{\varepsilon}_t = y_t - \hat{\mu}`` |
| `fitted` | `Vector{T}` | Fitted values (mean) |
| `loglik` | `T` | Maximized log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `method` | `Symbol` | Estimation method (`:mle`) |
| `converged` | `Bool` | Whether optimization converged |
| `iterations` | `Int` | Number of optimizer iterations |

---

## GARCH Models

The **Generalized ARCH** (GARCH) model of Bollerslev (1986) extends ARCH by including lagged conditional variances, producing a parsimonious representation of volatility clustering. Three variants are available: standard GARCH, EGARCH, and GJR-GARCH.

### GARCH(p,q)

The GARCH(``p``,``q``) specification adds ``p`` lagged variance terms to the ARCH equation:

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon^2_{t-i} + \sum_{j=1}^{p} \beta_j \sigma^2_{t-j}
```

where:
- ``\omega > 0`` is the variance intercept
- ``\alpha_i \geq 0`` are the ARCH coefficients (impact of past shocks)
- ``\beta_j \geq 0`` are the GARCH coefficients (variance persistence)
- ``p`` is the GARCH order (lagged variances) and ``q`` is the ARCH order (lagged squared residuals)

The process is covariance stationary when ``\sum_{i=1}^{q} \alpha_i + \sum_{j=1}^{p} \beta_j < 1``. The unconditional variance is ``\sigma^2 = \omega / (1 - \sum \alpha_i - \sum \beta_j)``. The GARCH(1,1) captures the key empirical regularity of volatility clustering with just three variance parameters.

```@example volatility
# Estimate GARCH(1,1) — the workhorse specification
garch = estimate_garch(spx, 1, 1)
report(garch)
```

```@example volatility
# Model-specific summary statistics
(persistence = round(persistence(garch), digits=4),      # α₁ + β₁
 halflife    = round(halflife(garch), digits=2),         # months
 uncond_var  = round(unconditional_variance(garch), digits=6),
 uncond_sd   = round(sqrt(unconditional_variance(garch)), digits=4))
```

!!! note "Robust standard errors by default"
    GARCH, EGARCH, and GJR-GARCH standard errors are **Bollerslev–Wooldridge (1992)
    QMLE-robust** by default (the sandwich ``H^{-1}(S'S)H^{-1}``), consistent even when the
    innovations are fat-tailed and the Gaussian likelihood is only a quasi-likelihood.
    Pass `cov_type=:hessian` to `stderror`/`vcov`/`confint` to recover the classical
    inverse-observed-information errors (valid only under correct Gaussian specification):
    `stderror(garch; cov_type=:hessian)`.

The GARCH(1,1) puts ``\hat\alpha_1 = 0.161`` on last month's squared shock and ``\hat\beta_1 = 0.707`` on last month's variance, both significant against Bollerslev-Wooldridge standard errors. Their sum, the persistence ``\hat\alpha_1 + \hat\beta_1 = 0.869``, is below one, so the process is covariance stationary and reverts to an unconditional variance of 0.0014 — a 3.7% monthly standard deviation. The implied half-life is 4.9 months: a volatility shock has decayed halfway back to normal after roughly a quarter and a half. Two parameters reproduce what took three ARCH lags, and the log-likelihood rises from 1563.7 to 1565.8 while the BIC improves from ``-3094.0`` to ``-3104.9``.

### EGARCH(p,q)

The **Exponential GARCH** (Nelson 1991) models the log of conditional variance, ensuring positivity without parameter constraints and allowing asymmetric responses to positive and negative shocks:

```math
\log(\sigma^2_t) = \omega + \sum_{i=1}^{q} \alpha_i (|z_{t-i}| - \mathbb{E}|z|) + \sum_{i=1}^{q} \gamma_i z_{t-i} + \sum_{j=1}^{p} \beta_j \log(\sigma^2_{t-j})
```

where:
- ``z_t = \varepsilon_t / \sigma_t`` are standardized residuals
- ``\alpha_i`` captures the magnitude (symmetric) effect of shocks
- ``\gamma_i`` captures the sign (asymmetric/leverage) effect --- typically ``\gamma_i < 0`` means negative shocks increase volatility more than positive shocks of equal magnitude
- ``\beta_j`` governs persistence of log-variance
- ``\mathbb{E}|z| = \sqrt{2/\pi}`` for standard normal innovations

The stationarity condition is ``\sum_{j=1}^{p} \beta_j < 1`` (in log-variance), and the unconditional variance is ``\sigma^2 = \exp(\omega / (1 - \sum \beta_j))``.

```@example volatility
egarch = estimate_egarch(spx, 1, 1)
report(egarch)
```

The leverage parameter is ``\hat\gamma_1 = -0.189`` with a standard error of 0.069, so the asymmetry is significant at 1%: a one-standard-deviation *fall* raises next month's log-variance by ``0.189`` more than a rise of the same size. Log-variance persistence is ``\hat\beta_1 = 0.829``. The AIC of ``-3165.0`` is the best of the three symmetric-versus-asymmetric fits here, ahead of GARCH's ``-3123.6`` and GJR's ``-3150.4`` — on monthly equity returns the sign of the shock carries information the symmetric model discards. The log-variance formulation also leaves the parameters unconstrained while guaranteeing ``\sigma^2_t > 0``.

### GJR-GARCH(p,q)

The **GJR-GARCH** (Glosten, Jagannathan & Runkle 1993), also called Threshold GARCH, adds an indicator function for negative shocks:

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} (\alpha_i + \gamma_i \mathbb{1}(\varepsilon_{t-i} < 0)) \varepsilon^2_{t-i} + \sum_{j=1}^{p} \beta_j \sigma^2_{t-j}
```

where:
- ``\gamma_i \geq 0`` are leverage parameters
- ``\mathbb{1}(\varepsilon_{t-i} < 0) = 1`` when past shocks are negative

When ``\gamma_i > 0``, negative shocks have a larger impact on future variance than positive shocks of equal magnitude. This captures the **leverage effect** first documented by Black (1976): stock price declines increase financial leverage, which in turn increases equity volatility. The stationarity condition is ``\sum \alpha_i + \sum \gamma_i / 2 + \sum \beta_j < 1``, and the unconditional variance is ``\sigma^2 = \omega / (1 - \sum \alpha_i - \sum \gamma_i / 2 - \sum \beta_j)``.

```@example volatility
gjr = estimate_gjr_garch(spx, 1, 1)
report(gjr)
```

``\hat\gamma_1 = 0.279`` with a standard error of 0.113 rejects ``\gamma_1 = 0`` at 5%, confirming the leverage effect from the threshold side. The symmetric ARCH coefficient is driven to the boundary ``\hat\alpha_1 = 0`` — a positive shock has *no* measurable effect on next month's variance once the negative-shock indicator is in the model, and the entire news impact runs through ``\gamma_1``. Persistence, ``\hat\alpha_1 + \hat\gamma_1/2 + \hat\beta_1 = 0.803``, again leaves the process stationary. GJR-GARCH nests standard GARCH at ``\gamma_i = 0``.

### News Impact Curve

The **news impact curve** (NIC) shows how a shock ``\varepsilon_{t-1}`` maps to the next-period conditional variance ``\sigma^2_t``, holding all other information constant at the unconditional level. For symmetric models (ARCH, GARCH), the NIC is a parabola centered at zero. For asymmetric models (EGARCH, GJR-GARCH), the NIC is steeper for negative shocks.

```@example volatility
garch = estimate_garch(spx, 1, 1)
egarch = estimate_egarch(spx, 1, 1)
gjr = estimate_gjr_garch(spx, 1, 1)

nic_garch  = news_impact_curve(garch)
nic_egarch = news_impact_curve(egarch; range=(-5.0, 5.0), n_points=400)
nic_gjr    = news_impact_curve(gjr)

# variance implied by a ∓2σ shock, symmetric model versus threshold model
lo, hi = argmin(abs.(nic_garch.shocks .+ 2)), argmin(abs.(nic_garch.shocks .- 2))
(garch_ratio = round(nic_garch.variance[lo] / nic_garch.variance[hi], digits=3),
 gjr_ratio   = round(nic_gjr.variance[lo]   / nic_gjr.variance[hi],   digits=3))
```

The symmetric GARCH assigns a ratio of exactly 1.000 to a ``-2\sigma`` shock against a ``+2\sigma`` shock — its NIC is a parabola centred at zero, so the sign is discarded by construction. The GJR curve puts the same two shocks at 0.0041 and 0.0010, a ratio of 3.9: a two-standard-deviation drop implies almost four times the next-month variance of an identically sized rally. The EGARCH curve reaches a comparable 3.1. When the two ratios come out near 1, the leverage effect is negligible and the simpler symmetric model suffices. `news_impact_curve` returns a named tuple with `shocks` (the grid of ``\varepsilon_{t-1}``) and `variance` (the implied ``\sigma^2_t``); `range` and `n_points` default to `(-3.0, 3.0)` and `200`.

```julia
plot_result(garch; view=:news_impact)
```

```@raw html
<iframe src="../assets/plots/news_impact_curve.html" width="100%" height="450" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### GARCH Diagnostic Visualization

ARCH, GARCH, EGARCH, and GJR-GARCH models produce a three-panel diagnostic figure via `plot_result()`: return series, conditional volatility, and standardized residuals with ``\pm 2`` standard deviation bounds.

```julia
garch = estimate_garch(spx, 1, 1)
plot_result(garch)
```

```@raw html
<iframe src="../assets/plots/model_garch.html" width="100%" height="700" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The top panel shows the raw return series to identify volatility clusters visually. The middle panel plots the estimated conditional standard deviation ``\hat{\sigma}_t``, which spikes during turbulent periods. The bottom panel displays standardized residuals ``\hat{z}_t = \hat{\varepsilon}_t / \hat{\sigma}_t``; these should be approximately i.i.d. standard normal if the model is well-specified.

### GARCH-Family Return Values



**GARCHModel Fields**

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `p` | `Int` | GARCH order (lagged variances) |
| `q` | `Int` | ARCH order (lagged squared residuals) |
| `mu` | `T` | Estimated mean |
| `omega` | `T` | Variance intercept ``\omega`` |
| `alpha` | `Vector{T}` | ARCH coefficients ``[\alpha_1, \ldots, \alpha_q]`` |
| `beta` | `Vector{T}` | GARCH coefficients ``[\beta_1, \ldots, \beta_p]`` |
| `conditional_variance` | `Vector{T}` | Estimated ``\hat{\sigma}^2_t`` |
| `standardized_residuals` | `Vector{T}` | ``\hat{z}_t`` |
| `residuals` | `Vector{T}` | ``\hat{\varepsilon}_t`` |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method |
| `dist` | `Symbol` | Conditional innovation distribution (`:normal`, `:student`, `:ged`) |
| `shape` | `T` | Estimated shape parameter; `NaN` under `:normal` |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Optimizer iterations |

**EGARCHModel Fields**

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `p` | `Int` | Log-variance persistence order |
| `q` | `Int` | Shock order |
| `mu` | `T` | Estimated mean |
| `omega` | `T` | Log-variance intercept |
| `alpha` | `Vector{T}` | Magnitude (symmetric) parameters |
| `gamma` | `Vector{T}` | Leverage (asymmetric) parameters |
| `beta` | `Vector{T}` | Log-variance persistence parameters |
| `conditional_variance` | `Vector{T}` | ``\hat{\sigma}^2_t`` |
| `standardized_residuals` | `Vector{T}` | ``\hat{z}_t`` |
| `residuals` | `Vector{T}` | ``\hat{\varepsilon}_t`` |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method |
| `dist` | `Symbol` | Conditional innovation distribution |
| `shape` | `T` | Estimated shape parameter; `NaN` under `:normal` |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Optimizer iterations |

**GJRGARCHModel Fields**

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `p` | `Int` | GARCH order |
| `q` | `Int` | ARCH order |
| `mu` | `T` | Estimated mean |
| `omega` | `T` | Variance intercept ``\omega`` |
| `alpha` | `Vector{T}` | Symmetric ARCH coefficients |
| `gamma` | `Vector{T}` | Leverage parameters ``[\gamma_1, \ldots, \gamma_q]`` |
| `beta` | `Vector{T}` | GARCH coefficients |
| `conditional_variance` | `Vector{T}` | ``\hat{\sigma}^2_t`` |
| `standardized_residuals` | `Vector{T}` | ``\hat{z}_t`` |
| `residuals` | `Vector{T}` | ``\hat{\varepsilon}_t`` |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method |
| `dist` | `Symbol` | Conditional innovation distribution |
| `shape` | `T` | Estimated shape parameter; `NaN` under `:normal` |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Optimizer iterations |

All three types also carry `param_vcov`, the QML sandwich covariance behind `stderror`, `vcov`, and `confint`.

---

## Fat-Tailed Innovations: Student-t and GED

The GARCH likelihood is Gaussian by default, but financial returns are leptokurtic — the Gaussian conditional density is misspecified, and the misspecification shows up as an understated probability of large moves. Pass `dist` to estimate a fat-tailed conditional distribution instead, with its shape parameter estimated **jointly** with the GARCH parameters:

- `:student` — Bollerslev's (1987) Student-t with ``\nu > 2`` degrees of freedom
- `:ged` — Nelson's (1991) generalized error distribution with shape ``\nu > 0``; ``\nu = 2`` is Gaussian, ``\nu = 1`` Laplace, ``\nu < 2`` fatter-tailed

Both are **standardized to unit variance**, which is what keeps the model identified: the GARCH recursion already owns the scale through ``h_t``, so an innovation distribution carrying its own free scale would be conflated with it. For the t that means dividing by ``\sqrt{\nu/(\nu-2)}``; for the GED it is the constant ``\lambda = \sqrt{\Gamma(1/\nu)/\Gamma(3/\nu)}``.

```math
\log f_t(z) = \log\Gamma\!\left(\tfrac{\nu+1}{2}\right) - \log\Gamma\!\left(\tfrac{\nu}{2}\right)
- \tfrac{1}{2}\log\!\left(\pi(\nu-2)\right) - \tfrac{\nu+1}{2}\log\!\left(1 + \tfrac{z^2}{\nu-2}\right)
```

Note ``\pi(\nu-2)`` rather than ``\pi\nu`` — the Jacobian of the standardization is already folded in.

```@example volatility
mg_n = estimate_garch(spx, 1, 1)
mg_t = estimate_garch(spx, 1, 1; dist=:student)

(loglik_normal = round(mg_n.loglik; digits=2),
 loglik_student = round(mg_t.loglik; digits=2),
 nu = round(mg_t.shape; digits=3),
 aic_prefers_t = mg_t.aic < mg_n.aic)
```

On the S&P 500 returns the estimated shape is ``\hat\nu = 4.77`` — fourth moments only just finite, and far from the ``\nu \to \infty`` Gaussian limit. The log-likelihood rises from 1565.81 to 1612.51 for that one extra parameter, and AIC falls from ``-3123.62`` to ``-3215.02``. The shape parameter is charged to AIC and BIC like any other, so the improvement is not free; here it is overwhelming, which is the usual verdict on monthly equity returns. Note that the GARCH coefficients themselves barely move (``\hat\alpha_1 = 0.148``, ``\hat\beta_1 = 0.736`` against 0.161 and 0.707): the Gaussian QMLE was consistent all along, and what the t buys is the tail probability, not the variance path.

`dist` is available on `estimate_garch`, `estimate_egarch` and `estimate_gjr_garch`. The default `:normal` is unchanged in every respect.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dist` | `Symbol` | `:normal` | Conditional innovation distribution: `:normal`, `:student`, or `:ged` |

!!! note "The shape parameter is clamped"
    Estimation is done in an unconstrained coordinate, ``\nu = 2 + e^{x}`` for the t and ``\nu = e^{x}`` for the GED, and the result is clamped to ``\nu \in [2.01, 500]`` and ``[0.1, 50]`` respectively. The lower clamp is not cosmetic: the t standardization divides by ``\nu - 2``, and ``2 + e^{x}`` rounds to *exactly* 2.0 in Float64 once ``x < -37``, which would make the density infinite and abort the line search.

!!! warning "QMLE versus MLE"
    Under `:normal` the Gaussian likelihood is a *quasi*-likelihood and the Bollerslev-Wooldridge sandwich standard errors remain the right choice. Under a correctly specified `:student` or `:ged` likelihood the inverse Hessian is efficient — but only if the distributional assumption holds. `stderror` continues to default to the robust sandwich, which is valid either way.

---

## GARCH-MIDAS

The **GARCH-MIDAS** model of Engle, Ghysels & Sohn (2013) links high-frequency financial volatility to slowly moving macroeconomic fundamentals by factoring the conditional variance into two multiplicative components:

```math
\sigma^2_{i,t} = \tau_t \cdot g_{i,t}
```

where ``i`` indexes the high-frequency return within low-frequency block ``t``. The **short-run** component ``g`` is a unit-mean GARCH(1,1) on the ``\tau``-standardized return,

```math
g_{i,t} = (1 - \alpha - \beta) + \alpha \frac{(r_{i-1,t} - \mu)^2}{\tau_{i-1}} + \beta \, g_{i-1,t},
```

and the **long-run** component is a MIDAS filter of a low-frequency driver ``X`` (a macro series or realized variance):

```math
\tau_t = \exp\!\left( m + \theta \sum_{k=1}^{K} \varphi_k(w) \, X_{t-k} \right),
```

with Beta weights ``\varphi_k(w)`` (monotone decaying, summing to one). The ``\sqrt{\tau}`` scaling of the short-run innovation keeps ``g`` at unit unconditional mean --- ``\tau`` carries the variance level.

- ``\mu`` --- conditional mean
- ``\alpha, \beta`` --- short-run ARCH/GARCH coefficients (``\alpha + \beta < 1``)
- ``m`` --- long-run intercept
- ``\theta`` --- MIDAS slope on the aggregated low-frequency driver
- ``w`` --- Beta weight shape (``w > 1`` gives decaying weights)

Estimation is by Gaussian QMLE over ``(\mu, \alpha, \beta, m, \theta, w)`` with ``\alpha, \beta`` log-transformed under the stationarity constraint and ``w = 1 + e^{\tilde w}``. The `variance_ratio` field reports ``\mathrm{Var}(\log \tau_t) / \mathrm{Var}(\log \sigma^2_{i,t})``, the share of total variance variation attributable to the long-run macro component (Engle, Ghysels & Sohn 2013).

```@example volatility
# Simulate a daily-frequency return series with a monthly macro driver
Random.seed!(202)
K, m_freq, nblk = 6, 21, 90
phi = midas_weights([1.0, 4.0], K; kind=:beta2)      # decaying Beta weights
Xlf = zeros(nblk); for t in 2:nblk; Xlf[t] = 0.7Xlf[t-1] + randn(); end
tau = [t > K ? exp(-0.5 + 0.3 * sum(phi[k] * Xlf[t-k] for k in 1:K)) : exp(-0.5) for t in 1:nblk]
r = Float64[]; g = 1.0; ep = 0.0; tp = tau[1]
for t in 1:nblk, i in 1:m_freq
    global g, ep, tp
    g = 0.04 + 0.06 * ep^2 / tp + 0.90 * g
    push!(r, 0.02 + sqrt(tau[t] * g) * randn()); ep = r[end] - 0.02; tp = tau[t]
end

# Fit GARCH-MIDAS with the macro driver
gm = estimate_garch_midas(r, Xlf; K=K, m_freq=m_freq, rv=:macro, span=:fixed)
report(gm)
```

The estimator recovers the simulated short-run dynamics closely --- ``\hat\alpha = 0.082`` and ``\hat\beta = 0.881`` against the 0.06 and 0.90 used to generate the data --- and the MIDAS slope ``\hat\theta = 0.225`` against 0.3 keeps the sign and order of magnitude of the macro loading. `variance_ratio` reports 0.255: a quarter of the variation in log conditional variance comes from the low-frequency macro component, the rest from day-to-day clustering. The weight shape ``\hat w`` is only weakly identified at 90 blocks and can drift to large values, which flattens ``\varphi`` onto the first lag; read `weights`, not ``\hat w``, when the decay profile matters.

The realized-variance variant needs no exogenous series --- the long-run driver is the block realized variance of the returns themselves:

```@example volatility
gm_rv = estimate_garch_midas(r; K=K, m_freq=m_freq, rv=:realized)
round(gm_rv.variance_ratio, digits=4)   # long-run share of total variance variation
```

With the macro driver replaced by realized variance the long-run share falls to 0.163, which is the expected direction here: the simulated ``\tau`` was built from `Xlf`, so realized variance recovers it only through the noisy returns.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `K` | `Int` | `12` | Number of low-frequency MIDAS lags |
| `m_freq` | `Int` | — (required) | High-frequency observations per low-frequency block |
| `rv` | `Symbol` | `:macro` | `:macro` for an exogenous driver, `:realized` for block realized variance |
| `span` | `Symbol` | `:fixed` | Rolling-window convention for the long-run component |

Forecasts iterate the short-run ``g`` forward (mean-reverting to 1) while holding the long-run ``\tau`` at its last low-frequency block:

```@example volatility
fc = forecast(gm, 10)
round.(fc.total, digits=4)          # total variance path σ² = τ·g
```

The path rises from 0.271 to 0.328 over ten days as ``g`` reverts to its unit mean from below while ``\tau`` stays fixed; `forecast` also returns `long_run` and `short_run` so the two components can be plotted separately.

The component overlay plots total ``\sqrt{\sigma^2}`` against the long-run ``\sqrt{\tau}``:

```julia
plot_result(gm; view=:components)
```

### GarchMidasModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `mu` | `T` | Conditional mean ``\mu`` |
| `alpha`, `beta` | `T` | Short-run ARCH/GARCH coefficients |
| `m_const` | `T` | Long-run intercept ``m`` |
| `theta` | `T` | MIDAS slope ``\theta`` |
| `w` | `T` | Beta weight shape ``w`` |
| `weights` | `Vector{T}` | Realized weight curve ``\varphi(\hat w)`` (length ``K``) |
| `tau` | `Vector{T}` | Long-run component per retained observation |
| `g` | `Vector{T}` | Short-run unit-mean component |
| `conditional_variance` | `Vector{T}` | Total ``\sigma^2 = \tau g`` |
| `variance_ratio` | `T` | Long-run variance share ``\mathrm{Var}(\log\tau)/\mathrm{Var}(\log\sigma^2)`` |
| `ret_idx` | `Vector{Int}` | Indices of retained (non-ragged) observations |
| `loglik`, `aic`, `bic` | `T` | Fit statistics |
| `converged` | `Bool` | Convergence status |

---

## Fractionally Integrated GARCH (FIGARCH / FIEGARCH)

Standard GARCH persistence decays *geometrically*: a shock to volatility dies out at rate ``(\alpha+\beta)^h``. Many financial and high-frequency macro series instead show *hyperbolic* decay --- the autocorrelation of squared returns falls off far more slowly than any GARCH can reproduce, yet the process is not fully integrated (IGARCH). **FIGARCH** (Baillie, Bollerslev & Mikkelsen 1996) bridges the two by applying the fractional-difference operator ``(1-L)^d`` with ``d \in (0,1)`` to the ARCH polynomial, giving an **ARCH(∞)** conditional variance:

```math
\sigma^2_t = \frac{\omega}{1-\beta(1)} + \Big[1 - \big(1-\beta(L)\big)^{-1}\phi(L)(1-L)^d\Big]\varepsilon^2_t
           = \omega^* + \sum_{i=1}^{K} \lambda_i \, \varepsilon^2_{t-i}.
```

The ``\lambda``-weights are obtained by convolving the ``(1-L)^d`` fractional-difference weights with ``\phi(L)`` and the inverse of ``1-\beta(L)``, then **truncating at ``K`` lags** (`truncation`, default 1000, capped internally at ``T-1``). Because the weights decay only ``\propto i^{-1-d}``, small ``d`` needs the full truncation — the example below uses 250 because the simulated process is itself built from a truncated weight sequence, and cost is ``O(TK)`` per likelihood evaluation. As ``d \to 0`` the model collapses to GARCH(1,1) with ``\alpha = \phi - \beta``; as ``d \to 1`` it approaches IGARCH.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `p`, `q` | `Int` | `1` | GARCH ``\beta(L)`` and ARCH ``\phi(L)`` orders |
| `d0` | `Real` | `0.4` | Starting value for the memory parameter |
| `truncation` | `Int` | `1000` | ARCH(∞) / MA(∞) truncation lag ``K`` |
| `dist` | `Symbol` | `:normal` | Only Gaussian QMLE is supported |

- ``\omega`` --- variance intercept (the ARCH(∞) level is ``\omega/(1-\beta(1))``)
- ``\phi`` --- ARCH polynomial coefficient(s)
- ``\beta`` --- GARCH polynomial coefficient(s)
- ``d`` --- fractional integration order (long-memory parameter), ``d \in (0,1)``

Estimation is by Gaussian QMLE with ``\omega`` log-transformed and ``\phi, \beta, d`` logit-transformed to their unit intervals; standard errors are Bollerslev–Wooldridge QML sandwich SEs with a delta-method back-transform. The pre-sample ``\varepsilon^2`` are set to the sample variance (matching `rugarch`). After fitting, the truncated ``\lambda``-weights are checked against the Baillie–Bollerslev–Mikkelsen non-negativity conditions --- a violation is **warned** (never thrown) and counted in `n_neg_lambda`.

```@example volatility
# Simulate a long-memory FIGARCH(1,d,1) return series (true d = 0.4)
Random.seed!(77)
let
    ω, d, ϕ, β, K = 0.05, 0.4, 0.2, 0.5, 400
    δ = [1.0]; for k in 1:K; push!(δ, δ[end] * (k - 1 - d) / k); end          # (1−L)^d
    g = [δ[k+1] - (k ≥ 1 ? ϕ * δ[k] : 0.0) for k in 0:K]                      # φ(L)(1−L)^d
    c = zeros(K + 1); c[1] = g[1]; for k in 1:K; c[k+1] = g[k+1] + β * c[k]; end
    λ = [-c[i+1] for i in 1:K]                                                # ARCH(∞) weights
    ostar, n, burn = ω / (1 - β), 1500, 800
    e2 = fill(ostar, n + burn); r = zeros(n + burn)
    for t in 1:(n + burn)
        h = ostar
        for i in 1:min(K, t - 1); h += λ[i] * e2[t-i]; end
        r[t] = sqrt(max(h, 1e-12)) * randn(); e2[t] = r[t]^2
    end
    global rets = r[burn+1:end]
end

m = estimate_figarch(rets; truncation=250)
report(m)
```

The fitted ``\hat d = 0.163`` carries a standard error of 0.051, so long memory is significant at the 1% level even though the point estimate sits well below the ``d = 0.4`` used to generate the series: the fractional parameters are only weakly identified at ``T = 1500``, and ``d`` trades off against ``\beta`` in exactly the way Pitfall 7 of the [ARIMA page](@ref arima_page) describes for ARFIMA. All 250 truncated ``\lambda``-weights satisfy the Baillie-Bollerslev-Mikkelsen non-negativity conditions here (`n_neg_lambda` is 0); a nonzero count is a warning, not an error, and means the fitted parameters imply a negative conditional variance somewhere on the grid.

**FIEGARCH** (Bollerslev & Mikkelsen 1996) is the log-variance analogue: long memory enters the *log* conditional variance through ``(1-L)^{-d}`` (note the negative exponent), and the EGARCH news function ``g(z) = \theta z + \gamma(|z| - E|z|)`` captures asymmetry (leverage). Because the log variance is unconstrained there is no positivity restriction:

```math
\ln \sigma^2_t = \omega + \big(1-\beta(L)\big)^{-1}\phi(L)(1-L)^{-d}\, g(z_{t-1}).
```

```@example volatility
fim = estimate_fiegarch(rets; truncation=250)
round(fim.d, digits=4)               # estimated long-memory parameter d
```

FIEGARCH puts the log-variance memory parameter at 0.181 on the same series. Because the log variance is unconstrained the estimate is far less precisely determined than the FIGARCH one, and it is sensitive to the truncation lag — check the estimate at two or three values of `truncation` before reporting it.

Both models support multi-step variance forecasts (simulation-based, feeding the ARCH(∞) / log-variance recursion forward) and a news impact curve:

```@example volatility
fc = forecast(m, 10)
round.(fc.forecast, digits=4)        # 10-step variance path
```

The forecast climbs from 0.418 to 0.471 over ten steps and keeps climbing: under hyperbolic decay the variance forecast approaches its long-run level at a rate ``h^{-d}`` rather than the ``(\alpha+\beta)^h`` of a stationary GARCH, so long-memory models forecast elevated volatility much further out after a turbulent period.

```julia
plot_result(m)                    # returns + fitted conditional volatility
news_impact_curve(m)              # symmetric FIGARCH parabola; FIEGARCH is asymmetric
```

### FIGARCHModel / FIEGARCHModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `mu` | `T` | Conditional mean ``\mu`` |
| `omega` | `T` | Variance intercept ``\omega`` |
| `phi`, `beta` | `Vector{T}` | ARCH ``\phi(L)`` / GARCH ``\beta(L)`` coefficients |
| `d` | `T` | Fractional integration order ``d \in (0,1)`` |
| `lambda` / `psi` | `Vector{T}` | Truncated ARCH(∞) / MA(∞) weights |
| `theta`, `gamma` | `T` | FIEGARCH sign / magnitude news coefficients |
| `conditional_variance` | `Vector{T}` | Fitted ``\sigma^2_t`` |
| `truncation` | `Int` | ARCH(∞) / MA(∞) truncation lag ``K`` |
| `n_neg_lambda` | `Int` | Negative ``\lambda``-weight count (FIGARCH BBM violation) |
| `loglik`, `aic`, `bic` | `T` | Fit statistics |
| `converged` | `Bool` | Convergence status |

---

## IGARCH, Component GARCH, and APARCH

Three further members of the GARCH family target distinct empirical regularities: **unit persistence** (IGARCH), a **long-run/short-run variance decomposition** (Component GARCH), and a **free volatility power with leverage** (APARCH). All three reuse the shared Bollerslev-Wooldridge (1992) QMLE sandwich standard errors and integrate with `report()`, `forecast()`, `news_impact_curve()`, and `plot_result()`.

### IGARCH(p,q)

The **Integrated GARCH** of Engle & Bollerslev (1986) imposes the persistence constraint ``\sum_i \alpha_i + \sum_j \beta_j = 1`` exactly, so a variance shock never dies out:

```math
\sigma^2_t = \omega + \sum_{i=1}^q \alpha_i \varepsilon^2_{t-i} + \sum_{j=1}^p \beta_j \sigma^2_{t-j}, \qquad \sum_i \alpha_i + \sum_j \beta_j = 1
```

Persistence is unity by construction, the unconditional variance diverges, and multi-step variance forecasts grow linearly. The RiskMetrics EWMA is the special case ``\omega = 0``.

```@example volatility
ig = estimate_igarch(spx, 1, 1)
report(ig)
```

The constraint binds: ``\hat\alpha_1 = 0.225`` and ``\hat\beta_1 = 0.775`` sum to one by construction, so only one of them is a free parameter. Imposing that restriction on a series whose unrestricted persistence is 0.869 costs 6.6 log-likelihood points and leaves AIC at ``-3112.5`` against the free GARCH's ``-3123.6`` — the data reject the unit root in variance here, which is the usual outcome for monthly returns and the reason IGARCH is a risk-management convention rather than a description.

```@example volatility
(persistence = persistence(ig),                # exactly 1.0 by construction
 halflife    = halflife(ig),                   # Inf: shocks never decay
 uncond_var  = unconditional_variance(ig))     # Inf: no finite long-run level
```

### Component GARCH(1,1)

The **Component GARCH** of Engle & Lee (1999) splits the conditional variance into a slowly mean-reverting **permanent** trend ``q_t`` and a fast **transitory** cycle:

```math
\sigma^2_t = q_t + \alpha(\varepsilon^2_{t-1} - q_{t-1}) + \beta(\sigma^2_{t-1} - q_{t-1})
```

```math
q_t = \omega + \rho(q_{t-1} - \omega) + \varphi(\varepsilon^2_{t-1} - \sigma^2_{t-1})
```

where:
- ``q_t`` is the permanent component, reverting to the long-run variance ``\omega`` with persistence ``\rho``
- ``\alpha + \beta`` is the transitory persistence; identification requires ``\rho > \alpha + \beta``

That identification condition is not a formality. On the full 803-month sample the MLE drives ``\varphi`` to zero and ``\rho`` onto ``\alpha + \beta``, the two components become indistinguishable, and the fit collapses onto the GARCH(1,1) solution with `converged` reported as `false` and unusable standard errors. Separating a slow trend from a fast cycle needs a window in which the two actually move at different speeds, so fit the last 500 months:

```@example volatility
cg = estimate_cgarch(spx[end-499:end])
report(cg)
```

Here the split identifies: the permanent component reverts with ``\hat\rho = 0.826`` while the transitory cycle carries ``\hat\alpha + \hat\beta = 0.543``, comfortably below it, and the permanent shock loading ``\hat\varphi = 0.234`` is significant at 10%. The long-run variance ``\hat\omega = 0.0015`` matches the sample variance of the window. `component_variances` returns the permanent, transitory, and total conditional-variance series, and the three reconcile exactly:

```@example volatility
comp = component_variances(cg)
(permanent_mean = round(sum(comp.permanent)/length(comp.permanent), digits=6),
 reconstructs_total = maximum(abs.(comp.permanent .+ comp.transitory .- comp.total)))
```

### APARCH(p,q)

The **Asymmetric Power ARCH** of Ding, Granger & Engle (1993) estimates a free power ``\delta > 0`` of the conditional standard deviation together with a Box-Cox leverage term ``\gamma_i \in (-1, 1)``:

```math
\sigma^\delta_t = \omega + \sum_{i=1}^q \alpha_i(|\varepsilon_{t-i}| - \gamma_i \varepsilon_{t-i})^\delta + \sum_{j=1}^p \beta_j \sigma^\delta_{t-j}
```

APARCH nests the standard family exactly: ``(\delta=2, \gamma=0)`` is GARCH, ``(\delta=2, \gamma \ne 0)`` is GJR-GARCH, and ``(\delta=1, \gamma \ne 0)`` is Zakoïan's TARCH. Pin parameters with the `fix_delta` / `fix_gamma` keywords.

```@example volatility
ap = estimate_aparch(spx, 1, 1)
report(ap)
```

The free power comes out at ``\hat\delta = 0.143``, nowhere near the ``\delta = 2`` that GARCH imposes: the S&P returns are better described by a recursion in ``\sigma^{0.14}_t`` than in the variance itself, which is the Ding-Granger-Engle finding that absolute returns raised to a low power show the strongest autocorrelation. The leverage term ``\hat\gamma_1 = 0.863`` is close to its upper bound of one, again pointing at a strongly one-sided news impact. With AIC ``-3176.9`` this is the best-fitting single-regime model on the page, ahead of EGARCH's ``-3165.0``. Pinning both extra parameters reproduces the GARCH likelihood to twelve decimal places, which is the nesting claim made concrete:

```@example volatility
# recover a plain GARCH(1,1) by pinning δ=2, γ=0
apg = estimate_aparch(spx, 1, 1; fix_delta=2.0, fix_gamma=0.0)
(aparch_loglik = round(apg.loglik, digits=6),
 garch_loglik  = round(estimate_garch(spx, 1, 1).loglik, digits=6))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `fix_delta` | `Real` | `nothing` | Pin the power ``\delta`` instead of estimating it |
| `fix_gamma` | `Real` | `nothing` | Pin the leverage ``\gamma`` instead of estimating it |
| `method` | `Symbol` | `:mle` | Estimation method |

### Volatility misspecification tests

The **Engle-Ng (1993) sign-bias test** regresses squared standardized residuals on the lagged shock's sign and size to detect asymmetry a symmetric model has missed; the joint statistic is ``(n-1)R^2 \sim \chi^2(3)``.

```@example volatility
garch = estimate_garch(spx, 1, 1)
sb = sign_bias_test(garch)
(joint = round(sb.joint_statistic, digits=3),
 pvalue = round(sb.joint_pvalue, digits=4),
 sign_bias_t = round(sb.sign_bias_t, digits=3))
```

The joint statistic of 12.75 against ``\chi^2(3)`` gives ``p = 0.005``, so the symmetric GARCH(1,1) leaves detectable asymmetry in its standardized residuals. The sign-bias coefficient itself carries a ``t`` of 2.46, meaning squared residuals are systematically larger after negative shocks — precisely the effect the EGARCH ``\hat\gamma_1 = -0.189`` and the GJR ``\hat\gamma_1 = 0.279`` pick up, and independent confirmation that the asymmetric models earn their extra parameter.

The **Nyblom (1989) / Hansen (1992) test** checks parameter stability against a martingale-parameter alternative, returning per-parameter statistics ``L_k`` and the joint ``L_C`` with the Hansen (1992) critical values:

```@example volatility
ny = nyblom_test(garch)
(joint = round(ny.joint, digits=4),
 cv_5pct = round(ny.cv_joint, digits=3),
 individual = round.(ny.individual, digits=3),
 cv_individual = ny.cv_individual)
```

The joint ``L_C = 0.816`` falls below the 5% critical value of 1.24, and every individual ``L_k`` is well under 0.470, so parameter stability is not rejected: a single GARCH(1,1) parameterization describes all 803 months, structural breaks in mean volatility notwithstanding. A rejection here is the signal to split the sample or to move to a regime-switching specification rather than to add lags.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `omega` | `T` | Variance intercept (IGARCH/APARCH) or long-run variance ``\omega`` (CGARCH) |
| `alpha`, `beta` | `Vector{T}`/`T` | ARCH / GARCH coefficients (IGARCH: ``\sum\alpha+\sum\beta=1``) |
| `rho`, `phi` | `T` | CGARCH permanent persistence ``\rho`` and shock loading ``\varphi`` |
| `gamma`, `delta` | `Vector{T}`, `T` | APARCH leverage ``\gamma`` and power ``\delta`` |
| `permanent`, `transitory` | `Vector{T}` | CGARCH long-run / short-run variance components |
| `conditional_variance` | `Vector{T}` | Fitted ``\sigma^2_t`` |
| `loglik`, `aic`, `bic` | `T` | Fit statistics |
| `converged` | `Bool` | Convergence status |

---

## Multivariate GARCH (CCC / DCC / BEKK)

The univariate models above describe one series at a time. Cross-asset spillovers, exchange-rate co-volatility, and CoVaR inputs instead need the full **conditional covariance matrix** ``H_t``. The multivariate GARCH estimators fit an ``n``-dimensional return matrix ``Y`` (``T \times n``) and return the ``n \times n \times T`` covariance path ``H_t``, the conditional correlations ``R_t``, and the per-series variances --- all extractable with [`covariances`](@ref), [`correlations`](@ref), and [`variances`](@ref). Every estimator **reuses the univariate [`estimate_garch`](@ref)** for its margins, so the marginal volatilities are exactly the standalone GARCH fits.

Three specifications are provided:

- **CCC** ([`estimate_ccc`](@ref), Bollerslev 1990) --- constant conditional correlation ``R``, ``H_t = D_t R D_t`` with ``D_t = \mathrm{diag}(\sigma_{1t}, …, \sigma_{nt})``. Simplest and fastest; the correlation is the sample correlation of the standardized residuals.
- **DCC / cDCC** ([`estimate_dcc`](@ref), Engle 2002 / Aielli 2013) --- dynamic correlation via a scalar recursion in ``(a, b)``.
- **BEKK** ([`estimate_bekk`](@ref), Engle & Kroner 1995) --- scalar or diagonal, modelling the covariance directly with covariance targeting (no separate margins).

```@setup volatility
using LinearAlgebra
# Bivariate DCC(1,1) process with GARCH(1,1) margins (fixed seed for the docs build)
let
    T, a, b, ρ = 400, 0.05, 0.90, 0.3
    rng = MersenneTwister(2024)
    Qbar = [1.0 ρ; ρ 1.0]; Q = copy(Qbar)
    Y = zeros(T, 2); h = [1.0, 1.0]
    for t in 1:T
        d = sqrt.([Q[1,1], Q[2,2]]); R = Q ./ (d * d'); R = (R + R') / 2
        L = cholesky(Symmetric(R)).L
        zt = L * randn(rng, 2)
        e = sqrt.(h) .* zt
        Y[t, :] = e
        Q = (1 - a - b) * Qbar + a * (zt * zt') + b * Q; Q = (Q + Q') / 2
        for i in 1:2; h[i] = 0.02 + 0.08 * e[i]^2 + 0.90 * h[i]; end
    end
    global Yret = Y
end
```

### Constant Conditional Correlation (CCC)

CCC decouples the problem: fit a univariate GARCH to each column, then hold the standardized-residual correlation fixed.

```math
H_t = D_t \, R \, D_t, \qquad R = \operatorname{corr}(z), \qquad z_t = D_t^{-1}\varepsilon_t.
```

```@example volatility
ccc = estimate_ccc(Yret)      # Yret is a 400×2 return matrix
report(ccc)
```

The constant correlation comes out at ``\hat R_{12} = 0.196`` against the 0.3 used to simulate ``\bar Q``, the downward bias one expects when a genuinely time-varying correlation is forced through a single number. It is stored in `ccc.R`; `covariances(ccc)` returns the ``2\times2\times400`` covariance path.

### Dynamic Conditional Correlation (DCC)

DCC lets the correlation evolve while keeping the two-step tractability. Step 1 estimates the same univariate margins; step 2 estimates ``(a, b)`` by maximizing the correlation quasi-likelihood, with the intercept fixed by **correlation targeting** ``\bar{Q} = \tfrac1T\sum_t z_t z_t'``:

```math
Q_t = (1-a-b)\bar{Q} + a\, z_{t-1}z_{t-1}' + b\, Q_{t-1}, \qquad
R_t = \operatorname{diag}(Q_t)^{-1/2} Q_t \operatorname{diag}(Q_t)^{-1/2}.
```

The parameters satisfy ``a, b \ge 0`` and ``a + b < 1`` (enforced by a logit-simplex reparametrization). Setting ``a = b = 0`` reduces DCC exactly to CCC.

```@example volatility
dcc = estimate_dcc(Yret)
report(dcc)
```

```@example volatility
a, b = coef(dcc)              # correlation dynamics
Rt = correlations(dcc)        # 2×2×400 time-varying correlations
(a = round(a, digits=4), b = round(b, digits=4),
 rho12_range = round.(extrema(Rt[1, 2, :]), digits=3))
```

The two-step estimator recovers ``\hat a = 0.059`` and ``\hat b = 0.894`` against the simulated 0.05 and 0.90, and the conditional ``\rho_{12}`` ranges from ``-0.26`` to ``0.57`` around a mean of 0.20 — the swing that the single CCC number averages away. The log-likelihood improves from ``-958.0`` to ``-951.5`` for those two parameters, so AIC prefers DCC (1922.9) to CCC (1933.9).

Pass `correction=:aielli` for the **cDCC** variant (Aielli 2013), which removes the standard-DCC intercept-targeting bias by replacing ``z_t z_t'`` with ``q^*_t q^{*\prime}_t``, ``q^*_t = \operatorname{diag}(Q_t)^{1/2} z_t``.

### BEKK

BEKK models the covariance directly (no separate margins). With **covariance targeting** the intercept is fixed so the unconditional covariance equals the sample covariance ``\bar\Sigma``; only the news/persistence parameters are estimated, which keeps the recursion positive semidefinite and stable.

```math
\text{scalar:}\quad H_t = (1-a-b)\bar\Sigma + a\,\varepsilon_{t-1}\varepsilon_{t-1}' + b\,H_{t-1}.
```

```@example volatility
bekk = estimate_bekk(Yret)               # scalar (default)
report(bekk)
```

Scalar BEKK estimates ``\hat a = 0.092`` and ``\hat b = 0.843`` directly on the covariance and reaches an AIC of 1913.6, the best of the three here — it spends fewer parameters than DCC because covariance targeting fixes the intercept and no separate margins are fitted. The diagonal variant (`kind=:diagonal`) uses ``A, B = \operatorname{diag}(a), \operatorname{diag}(b)`` and ``H_t = \tilde C + A\varepsilon_{t-1}\varepsilon_{t-1}'A + B H_{t-1} B``.

### Accessors, forecasting & plotting

All three models share the same interface:

```@example volatility
H  = covariances(dcc)     # n×n×T conditional covariances
R  = correlations(dcc)    # n×n×T conditional correlations (constant broadcast for CCC/BEKK)
V  = variances(dcc)       # T×n conditional variances (the diagonals)
fc = forecast(dcc, 10)    # 10-step-ahead covariance forecast, n×n×10
size(fc)
```

```julia
plot_result(dcc; view=:correlations)                  # pairwise conditional correlations over time
plot_result(dcc; view=:covariance_heatmap, at=400)    # heatmap of Hₜ at a chosen t
```

### MGARCHModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Data (T×n) |
| `mu` | `Vector{T}` | Per-series conditional mean |
| `margins` | `Vector{GARCHModel{T}}` | Univariate GARCH margin fits (empty for BEKK) |
| `H` | `Array{T,3}` | Conditional covariance path ``H_t`` (n×n×T) |
| `R` | `Matrix{T}` / `Array{T,3}` | Correlation --- constant (CCC/BEKK) or time-varying (DCC) |
| `Rbar` | `Matrix{T}` | Unconditional / targeting correlation |
| `params` | `Vector{T}` | Second-stage parameters (`[a,b]` for DCC/scalar BEKK; empty for CCC) |
| `param_vcov` | `Matrix{T}` | QML sandwich covariance of `params` |
| `loglik`, `aic`, `bic` | `T` | Joint Gaussian (quasi) log-likelihood and information criteria |
| `kind` | `Symbol` | `:ccc`, `:dcc`, or `:bekk` |
| `correction` | `Symbol` | `:none` or `:aielli` (DCC) |
| `bekk_kind` | `Symbol` | `:scalar` or `:diagonal` (BEKK) |
| `converged` | `Bool` | Second-stage convergence status |

---

## Stochastic Volatility

The **stochastic volatility** (SV) model of Taylor (1986) treats the log-variance as a latent autoregressive process with its own source of randomness, making it fundamentally different from the observation-driven GARCH family. The SV model is a state-space model with a non-Gaussian observation equation, providing greater flexibility in capturing empirical volatility dynamics at the cost of requiring simulation-based estimation.

```math
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
```

```math
h_t = \mu + \varphi (h_{t-1} - \mu) + \sigma_\eta \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
```

where:
- ``h_t`` is the log-variance at time ``t``
- ``\mu`` is the log-variance level (unconditional mean of ``h_t``)
- ``\varphi \in (-1, 1)`` is the persistence parameter
- ``\sigma_\eta > 0`` is the volatility of volatility
- ``\varepsilon_t`` and ``\eta_t`` are independent standard normal innovations

### SV Variants

Three variants are available, selected via keyword arguments:

**Basic SV** (`leverage=false`, `dist=:normal`): The standard specification above.

**SV with Leverage** (`leverage=true`): Allows correlation between return and volatility innovations:

```math
\begin{pmatrix} \varepsilon_t \\ \eta_t \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)
```

where:
- ``\rho`` is the correlation between return and volatility shocks

When ``\rho < 0`` (the typical case for equities), negative returns are associated with increases in volatility, analogous to the leverage effect in EGARCH and GJR-GARCH models.

**SV with Student-t Errors** (`dist=:studentt`): Replaces the Gaussian observation equation with Student-t innovations to accommodate heavier tails:

```math
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim t_\nu
```

where:
- ``\nu > 2`` is the degrees of freedom parameter (ensuring finite variance)

### Priors and Estimation

The SV model is estimated via the Kim-Shephard-Chib (1998) Gibbs sampler with the Omori et al. (2007) 10-component mixture approximation. The default priors are:

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| ``\mu`` | ``\mathcal{N}(0, 10)`` | Weakly informative for log-variance level |
| ``\varphi`` | ``\text{Beta}(20, 1.5) \to (-1, 1)`` | Concentrates mass near 1 (high persistence), ensures stationarity |
| ``\sigma_\eta`` | ``\text{HalfNormal}(1)`` | Positive, moderately informative for vol-of-vol |
| ``\rho`` (leverage) | ``\text{Uniform}(-1, 1)`` | Uninformative over correlation range |
| ``\nu`` (Student-t) | ``\text{Exponential}(0.1) + 2`` | Ensures ``\nu > 2`` (finite variance) |

!!! note "Technical Note"
    The Kim-Shephard-Chib (1998) Gibbs sampler approximates the non-Gaussian observation equation ``\log y_t^2 = h_t + \log \varepsilon_t^2`` using a 10-component Gaussian mixture (Omori et al. 2007). Each Gibbs iteration: (1) samples the mixture indicators conditional on ``h``, (2) samples ``h_{1:T}`` via the simulation smoother conditional on parameters and indicators, and (3) samples ``(\mu, \varphi, \sigma_\eta)`` from their conditional posteriors. Typical run times are under 30 seconds for ``T = 500`` with 2000 posterior draws.

```@example volatility
# Basic SV model
sv = estimate_sv(spx; n_samples=2000, burnin=1000)
report(sv)
```

```@example volatility
# SV with leverage effect
sv_lev = estimate_sv(spx; leverage=true, n_samples=2000, burnin=1000)
report(sv_lev)
```

```@example volatility
# SV with Student-t errors
sv_t = estimate_sv(spx; dist=:studentt, n_samples=2000, burnin=1000)
report(sv_t)
```

The posterior mean of ``\varphi`` is ``\approx 0.92`` with a 95% credible interval that excludes one, so log-volatility is highly persistent but stationary — the same conclusion the GARCH persistence of 0.869 reaches by a different route. The volatility of volatility ``\sigma_\eta`` posterior sits near 0.25, and ``\mu \approx -6.9`` implies a long-run variance ``e^{\mu} \approx 0.001``, again matching the GARCH unconditional variance. Adding leverage or Student-t errors moves ``\varphi`` by less than 0.01 on this series: the extra flexibility is absorbed by the observation equation, not the volatility dynamics.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_samples` | `Int` | `2000` | Number of posterior draws (after burnin) |
| `burnin` | `Int` | `1000` | Number of initial draws to discard |
| `dist` | `Symbol` | `:normal` | Error distribution (`:normal` or `:studentt`) |
| `leverage` | `Bool` | `false` | Whether to estimate leverage correlation ``\rho`` |
| `quantile_levels` | `Vector{Real}` | `[0.025, 0.5, 0.975]` | Quantile levels for posterior volatility bands |

### SV Posterior Visualization

The SV model visualization shows posterior volatility with quantile credible bands:

```julia
sv = estimate_sv(spx; n_samples=2000, burnin=1000)
plot_result(sv)
```

```@raw html
<iframe src="../assets/plots/model_sv.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The plot displays the posterior mean of ``\exp(h_t)`` (the conditional standard deviation) with 95% credible bands. Wider bands during turbulent periods reflect greater posterior uncertainty about the latent volatility state.

### SVModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `h_draws` | `Matrix{T}` | Latent log-volatility draws (n_samples ``\times`` T) |
| `mu_post` | `Vector{T}` | Posterior draws of ``\mu`` |
| `phi_post` | `Vector{T}` | Posterior draws of ``\varphi`` |
| `sigma_eta_post` | `Vector{T}` | Posterior draws of ``\sigma_\eta`` |
| `volatility_mean` | `Vector{T}` | Posterior mean of ``\exp(h_t)`` at each ``t`` |
| `volatility_quantiles` | `Matrix{T}` | ``T \times n_q`` quantiles of ``\exp(h_t)`` |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., ``[0.025, 0.5, 0.975]``) |
| `dist` | `Symbol` | Error distribution (`:normal` or `:studentt`) |
| `leverage` | `Bool` | Whether leverage effect was estimated |
| `n_samples` | `Int` | Number of posterior samples |

---

## Diagnostics

Two diagnostic tests verify whether ARCH effects are present in raw data or have been adequately captured by a fitted model.

### ARCH-LM Test

The **ARCH-LM test** (Engle 1982) regresses squared residuals on ``q`` of their own lags and computes the test statistic:

```math
\text{LM} = T \cdot R^2 \sim \chi^2(q)
```

where:
- ``T`` is the sample size
- ``R^2`` is the coefficient of determination from the auxiliary regression of ``\hat{\varepsilon}^2_t`` on ``(\hat{\varepsilon}^2_{t-1}, \ldots, \hat{\varepsilon}^2_{t-q})``
- ``q`` is the number of lags

Under the null hypothesis of no ARCH effects, ``\text{LM} \sim \chi^2(q)``. Rejection indicates ARCH effects are present (or remain after fitting).

```@example volatility
# Test raw data for ARCH effects (H₀: no ARCH effects)
stat, pval, q = arch_lm_test(spx, 5)

# Test standardized residuals after fitting (should fail to reject)
garch = estimate_garch(spx, 1, 1)
stat_r, pval_r, q_r = arch_lm_test(garch, 5)
```

A significant test on raw data (small p-value) confirms the need for volatility modeling. After fitting, the test on standardized residuals should fail to reject, confirming the model has adequately captured the variance dynamics.

### Ljung-Box Test on Squared Residuals

The **Ljung-Box test** applied to squared standardized residuals tests for remaining serial correlation in the variance:

```math
Q = n(n+2) \sum_{k=1}^{K} \frac{\hat{\rho}^2_k}{n - k} \sim \chi^2(K)
```

where:
- ``n`` is the number of observations
- ``\hat{\rho}_k`` is the sample autocorrelation of squared standardized residuals at lag ``k``
- ``K`` is the maximum lag order

```@example volatility
garch = estimate_garch(spx, 1, 1)
stat, pval, K = ljung_box_squared(garch, 10)
```

Failure to reject indicates the model has adequately captured the variance dynamics. A significant result suggests the need for higher ARCH/GARCH orders or an alternative specification.

---

## Volatility Forecasting

All volatility models support multi-step ahead forecasting via `forecast()`. ARCH and GARCH-family models use simulation-based confidence intervals; SV models use posterior predictive simulation from MCMC draws.

### GARCH-Family Forecasts

For stationary GARCH processes, multi-step forecasts converge geometrically to the unconditional variance at rate equal to the persistence parameter. The speed of convergence is measured by the half-life:

```math
\text{halflife} = \frac{\log(0.5)}{\log(\text{persistence})}
```

where:
- ``\text{persistence} = \sum \alpha_i + \sum \beta_j`` for GARCH (adjusted for EGARCH and GJR-GARCH)

Confidence intervals are constructed by simulating ``n`` paths forward from the last observed state, generating the empirical distribution of future conditional variances.

```@example volatility
garch = estimate_garch(spx, 1, 1)
fc = forecast(garch, 20; conf_level=0.95, n_sim=10000)
report(fc)
```

The forecast report displays a table of point forecasts, standard errors, and confidence interval bounds at each horizon. The point forecast at horizon 1 reflects the current volatility state, while long-horizon forecasts converge to `unconditional_variance(garch)`. For ARCH models, forecasts beyond horizon ``q`` equal the unconditional variance exactly (no lagged variance terms to propagate).

### SV Forecasts

For SV models, each posterior draw provides a full parameter vector ``(\mu, \varphi, \sigma_\eta)`` and the terminal log-volatility ``h_T``. The forecast simulates the log-volatility process forward from the last state for each draw, yielding a posterior predictive distribution of future volatility. The reported intervals are posterior predictive quantiles, not frequentist confidence intervals.

```@example volatility
sv = estimate_sv(spx; n_samples=2000, burnin=1000)
fc_sv = forecast(sv, 20; conf_level=0.95)
report(fc_sv)
```

### Volatility Forecast Visualization

```julia
garch = estimate_garch(spx, 1, 1)
fc = forecast(garch, 10)
plot_result(fc; history=garch.conditional_variance)
```

```@raw html
<iframe src="../assets/plots/forecast_volatility.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The plot shows the conditional variance forecast (blue line) with confidence bands, optionally preceded by the in-sample conditional variance history. Forecasts fan out as the horizon increases, reflecting growing uncertainty, and converge toward the unconditional variance level.

### VolatilityForecast Return Values

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Vector{T}` | Point forecasts of conditional variance ``\hat{\sigma}^2_{T+h}`` |
| `ci_lower` | `Vector{T}` | Lower confidence/credible interval bound |
| `ci_upper` | `Vector{T}` | Upper confidence/credible interval bound |
| `se` | `Vector{T}` | Standard errors of forecasts |
| `horizon` | `Int` | Forecast horizon |
| `conf_level` | `T` | Confidence level (e.g., 0.95) |
| `model_type` | `Symbol` | Source model (`:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`) |

---

## Type Accessors

The following accessor functions provide model-specific summary statistics. The formulas differ across model types:

| Function | ARCH | GARCH | EGARCH | GJR-GARCH | SV |
|----------|------|-------|--------|-----------|-----|
| `persistence(m)` | ``\sum \alpha_i`` | ``\sum \alpha_i + \sum \beta_j`` | ``\sum \beta_j`` | ``\sum \alpha_i + \sum \gamma_i/2 + \sum \beta_j`` | ``\mathbb{E}[\varphi]`` |
| `halflife(m)` | ``\log(0.5)/\log(p)`` | ``\log(0.5)/\log(p)`` | ``\log(0.5)/\log(p)`` | ``\log(0.5)/\log(p)`` | ``\log(0.5)/\log(p)`` |
| `unconditional_variance(m)` | ``\frac{\omega}{1 - \sum \alpha_i}`` | ``\frac{\omega}{1 - \sum \alpha_i - \sum \beta_j}`` | ``\exp\!\left(\frac{\omega}{1 - \sum \beta_j}\right)`` | ``\frac{\omega}{1 - \sum \alpha_i - \sum \gamma_i/2 - \sum \beta_j}`` | ``\exp(\mathbb{E}[\mu])`` |
| `arch_order(m)` | ``q`` | ``q`` | ``q`` | ``q`` | --- |
| `garch_order(m)` | --- | ``p`` | ``p`` | ``p`` | --- |

In the table, ``p`` denotes `persistence(m)`. The half-life returns `Inf` if the process is non-stationary (persistence ``\geq 1``).

```@example volatility
garch = estimate_garch(spx, 1, 1)
persistence(garch)              # α₁ + β₁
halflife(garch)                 # Half-life in periods
unconditional_variance(garch)   # Long-run variance
arch_order(garch)               # q
garch_order(garch)              # p
```

---

## StatsAPI Interface

All volatility models implement the standard StatsAPI interface:

| Function | Description |
|----------|-------------|
| `nobs(m)` | Number of observations |
| `coef(m)` | Coefficient vector |
| `residuals(m)` | Raw residuals ``\hat{\varepsilon}_t`` |
| `predict(m)` | Conditional variance series ``\hat{\sigma}^2_t`` (or posterior mean for SV) |
| `loglikelihood(m)` | Maximized log-likelihood (ARCH/GARCH) |
| `aic(m)` | Akaike Information Criterion |
| `bic(m)` | Bayesian Information Criterion |
| `dof(m)` | Number of estimated parameters |
| `islinear(m)` | `false` (all volatility models are nonlinear) |
| `stderror(m; cov_type=:robust)` | Standard errors: Bollerslev–Wooldridge QMLE-robust sandwich (default) or `:hessian` (inverse observed information) |
| `confint(m)` | Confidence intervals for parameters |
| `vcov(m)` | Variance-covariance matrix of parameter estimates |

```@example volatility
garch = estimate_garch(spx, 1, 1)
nobs(garch)          # Number of observations
loglikelihood(garch) # Maximized log-likelihood
aic(garch)           # AIC for model comparison
bic(garch)           # BIC for model comparison
coef(garch)          # [μ, ω, α₁, ..., αq, β₁, ..., βp]
```

---

## Complete Example

This example estimates all four GARCH-family models on monthly industrial production growth (FRED-MD INDPRO, tcode-transformed to log-growth), runs diagnostics, compares specifications, and estimates an SV model for comparison.

```@example volatility
# === Step 1: Test for ARCH effects ===
stat, pval, q = arch_lm_test(spx, 5)

# === Step 2: Estimate competing GARCH-family models ===
garch  = estimate_garch(spx, 1, 1)
egarch = estimate_egarch(spx, 1, 1)
gjr    = estimate_gjr_garch(spx, 1, 1)

# Display each model's coefficient table and fit statistics
report(garch)
```

```@example volatility
report(egarch)
```

```@example volatility
report(gjr)
```

```@example volatility
# === Step 3: Compare information criteria and persistence ===
round(aic(garch), digits=1)
round(aic(egarch), digits=1)
round(aic(gjr), digits=1)
round(persistence(garch), digits=4)
round(persistence(egarch), digits=4)
round(persistence(gjr), digits=4)
```

```@example volatility
# === Step 4: Check residual diagnostics ===
_, p_g, _ = arch_lm_test(garch, 5)
_, p_e, _ = arch_lm_test(egarch, 5)
_, p_j, _ = arch_lm_test(gjr, 5)
nothing  # hide
```

```@example volatility
# === Step 5: Forecast volatility ===
fc = forecast(garch, 20; conf_level=0.95)
report(fc)
```

```julia
plot_result(fc; history=garch.conditional_variance)
```

```@raw html
<iframe src="../assets/plots/forecast_volatility.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example volatility
# === Step 6: Stochastic volatility for comparison ===
sv = estimate_sv(spx; n_samples=2000, burnin=1000)
report(sv)
```

```julia
plot_result(sv)
```

```@raw html
<iframe src="../assets/plots/model_sv.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example volatility
# SV forecast
fc_sv = forecast(sv, 20)
report(fc_sv)
```

The industrial production growth series exhibits ARCH effects, confirming the need for volatility modeling. The EGARCH and GJR-GARCH models capture asymmetric leverage effects --- negative shocks increase volatility more than positive shocks of equal magnitude --- which the symmetric GARCH specification misses. After fitting, all models' standardized residuals pass the ARCH-LM test, confirming adequate capture of variance dynamics. The SV model provides an independent Bayesian assessment via the Kim-Shephard-Chib (1998) Gibbs sampler, with posterior credible bands quantifying parameter uncertainty.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example volatility
path = joinpath(mktempdir(), "garch.jld2")
save_model(garch, path)
garch2 = load_model(path)
typeof(garch2)
```

---

## Common Pitfalls

1. **Non-stationarity when persistence ``\geq 1``**: If ``\sum \alpha_i + \sum \beta_j \geq 1`` (GARCH), the process is non-stationary and the unconditional variance is infinite. The `unconditional_variance()` accessor returns `Inf` and `halflife()` returns `Inf`. This typically indicates the model is overparameterized or the data contains a structural break in volatility. Consider a lower-order specification or splitting the sample.

2. **EGARCH parameter sign conventions**: In the EGARCH specification, ``\gamma_i`` captures the leverage (sign) effect and ``\alpha_i`` captures the magnitude (symmetric) effect. A negative ``\gamma_i`` means negative shocks increase volatility more than positive shocks. Do not confuse ``\gamma_i`` in EGARCH with ``\gamma_i`` in GJR-GARCH --- in GJR-GARCH, ``\gamma_i \geq 0`` with positive values indicating leverage.

3. **SV burnin too short**: The default burnin of 1000 draws is adequate for most applications, but highly persistent series (``\varphi > 0.99``) or heavy-tailed data may require longer burnin (2000--5000 draws) for the Gibbs sampler to reach the stationary distribution. Monitor the posterior traces of ``\mu``, ``\varphi``, and ``\sigma_\eta`` for convergence.

4. **NelderMead convergence issues**: The two-stage optimizer occasionally fails to converge, particularly for high-order models or short samples. Check `m.converged` after estimation. If `false`, try different starting values by re-estimating on a slightly different sample or reducing the model order.

5. **ARCH-LM test interpretation**: Rejecting the null on raw data means ARCH effects are present (good --- proceed with volatility modeling). Rejecting the null on standardized residuals from a fitted model means the model has not adequately captured the variance dynamics (bad --- try a higher order or different specification). The test is one-sided: failure to reject does not prove the absence of ARCH effects, only that the test lacks power to detect them at the chosen lag order.

6. **ARCH order ``q`` versus GARCH notation**: In `estimate_garch(y, p, q)`, the first argument `p` is the GARCH order (lagged variances) and the second `q` is the ARCH order (lagged squared residuals). This follows the Bollerslev (1986) convention. The standard workhorse is `estimate_garch(y, 1, 1)`.

---

## References

- Aielli, G. P. (2013). Dynamic Conditional Correlation: On Properties and Estimation.
  *Journal of Business & Economic Statistics*, 31(3), 282--299. [DOI](https://doi.org/10.1080/07350015.2013.771027)

- Black, F. (1976). Studies of Stock Price Volatility Changes.
  *Proceedings of the 1976 Meetings of the American Statistical Association*, 171--177.

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
  *Journal of Econometrics*, 31(3), 307--327. [DOI](https://doi.org/10.1016/0304-4076(86)90063-1)

- Bollerslev, T. (1990). Modelling the Coherence in Short-Run Nominal Exchange Rates: A Multivariate Generalized ARCH Model.
  *Review of Economics and Statistics*, 72(3), 498--505. [DOI](https://doi.org/10.2307/2109358)

- Engle, R. F. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models.
  *Journal of Business & Economic Statistics*, 20(3), 339--350. [DOI](https://doi.org/10.1198/073500102288618487)

- Engle, R. F., & Kroner, K. F. (1995). Multivariate Simultaneous Generalized ARCH.
  *Econometric Theory*, 11(1), 122--150. [DOI](https://doi.org/10.1017/S0266466600009063)

- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.
  *Econometrica*, 50(4), 987--1007. [DOI](https://doi.org/10.2307/1912773)

- Engle, R. F., Ghysels, E., & Sohn, B. (2013). Stock Market Volatility and Macroeconomic Fundamentals.
  *Review of Economics and Statistics*, 95(3), 776--797. [DOI](https://doi.org/10.1162/REST_a_00300)

- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks.
  *Journal of Finance*, 48(5), 1779--1801. [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)

- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models.
  *Review of Economic Studies*, 65(3), 361--393. [DOI](https://doi.org/10.1111/1467-937X.00050)

- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
  *Econometrica*, 59(2), 347--370. [DOI](https://doi.org/10.2307/2938260)

- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference.
  *Journal of Econometrics*, 140(2), 425--449. [DOI](https://doi.org/10.1016/j.jeconom.2006.07.008)

- Taylor, S. J. (1986). *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90975-7.
