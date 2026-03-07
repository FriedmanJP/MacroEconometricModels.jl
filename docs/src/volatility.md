# [Volatility Models](@id volatility_page)

**MacroEconometricModels.jl** provides a complete suite of univariate volatility models for capturing time-varying conditional variance in financial and macroeconomic time series. The package covers observation-driven (ARCH/GARCH family) and parameter-driven (stochastic volatility) approaches, with unified diagnostics, forecasting, and visualization.

- **ARCH**: Autoregressive Conditional Heteroskedasticity (Engle 1982) --- conditional variance depends on past squared innovations
- **GARCH**: Generalized ARCH (Bollerslev 1986) --- adds lagged conditional variances for parsimonious volatility persistence
- **EGARCH**: Exponential GARCH (Nelson 1991) --- log-variance specification with asymmetric leverage effects, no positivity constraints
- **GJR-GARCH**: Threshold GARCH (Glosten, Jagannathan & Runkle 1993) --- indicator-based leverage via ``\gamma_i \mathbb{1}(\varepsilon_{t-i} < 0)``
- **Stochastic Volatility**: Latent log-variance AR(1) process (Taylor 1986), estimated via Kim-Shephard-Chib (1998) Gibbs sampler with optional leverage and Student-t errors
- **Diagnostics**: ARCH-LM test, Ljung-Box on squared residuals, news impact curves
- **Forecasting**: Multi-step ahead variance forecasts with simulation-based confidence intervals (GARCH family) or posterior predictive intervals (SV)

```@setup volatility
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
ip = filter(isfinite, to_vector(apply_tcode(fred[:, "INDPRO"])))
ip = ip[end-99:end]
```

## Quick Start

**Recipe 1: ARCH(q)**

```@example volatility
# ARCH(5) — Engle (1982)
arch = estimate_arch(ip, 5)
report(arch)
```

**Recipe 2: GARCH(1,1)**

```@example volatility
# GARCH(1,1) — the workhorse specification
garch = estimate_garch(ip, 1, 1)
report(garch)
```

**Recipe 3: Asymmetric GARCH models**

```@example volatility
# EGARCH captures leverage without positivity constraints
egarch = estimate_egarch(ip, 1, 1)
report(egarch)
```

```@example volatility
# GJR-GARCH captures leverage via an indicator function
gjr = estimate_gjr_garch(ip, 1, 1)
report(gjr)
```

**Recipe 4: Stochastic volatility**

```@example volatility
# SV via Kim-Shephard-Chib (1998) Gibbs sampler
sv = estimate_sv(ip; n_samples=2000, burnin=1000)
report(sv)
```

**Recipe 5: ARCH-LM diagnostics**

```@example volatility
# Test raw data for ARCH effects (should reject)
stat, pval, q = arch_lm_test(ip, 5)

# Fit GARCH and test residuals (should fail to reject)
garch = estimate_garch(ip, 1, 1)
stat_r, pval_r, q_r = arch_lm_test(garch, 5)
```

**Recipe 6: Volatility forecasting**

```@example volatility
garch = estimate_garch(ip, 1, 1)
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
# Estimate ARCH(5) model
arch = estimate_arch(ip, 5)
report(arch)
```

The `report()` output displays a publication-quality coefficient table with standard errors, z-statistics, p-values, and 95% confidence intervals, followed by fit statistics including persistence, unconditional variance, and information criteria. Persistence close to 1 indicates highly persistent volatility clustering. The AIC and BIC enable comparison with GARCH-family alternatives.

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
garch = estimate_garch(ip, 1, 1)
report(garch)
```

```@example volatility
# Model-specific summary statistics
persistence(garch)              # α₁ + β₁ (close to 1 = slow reversion)
halflife(garch)                 # Half-life in periods
unconditional_variance(garch)   # Long-run variance level
```

The persistence ``\alpha_1 + \beta_1`` for equity returns typically falls between 0.9 and 0.99, implying that volatility shocks are highly persistent. A half-life of 13 periods means that half the impact of a volatility shock dissipates after 13 time units. The unconditional variance provides the long-run level to which the conditional variance mean-reverts.

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
egarch = estimate_egarch(ip, 1, 1)
report(egarch)
```

A negative ``\gamma_1`` confirms the leverage effect: bad news increases volatility more than good news. The EGARCH specification is particularly useful when the symmetric GARCH assumption is restrictive, as the log-variance formulation accommodates unconstrained parameters while guaranteeing ``\sigma^2_t > 0``.

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
gjr = estimate_gjr_garch(ip, 1, 1)
report(gjr)
```

A statistically significant ``\gamma_1 > 0`` confirms the leverage effect. The GJR-GARCH nests the standard GARCH as a special case when ``\gamma_i = 0``.

### News Impact Curve

The **news impact curve** (NIC) shows how a shock ``\varepsilon_{t-1}`` maps to the next-period conditional variance ``\sigma^2_t``, holding all other information constant at the unconditional level. For symmetric models (ARCH, GARCH), the NIC is a parabola centered at zero. For asymmetric models (EGARCH, GJR-GARCH), the NIC is steeper for negative shocks.

```@example volatility
garch = estimate_garch(ip, 1, 1)
egarch = estimate_egarch(ip, 1, 1)
gjr = estimate_gjr_garch(ip, 1, 1)

nic_garch  = news_impact_curve(garch)
nic_egarch = news_impact_curve(egarch; range=(-3.0, 3.0), n_points=200)
nic_gjr    = news_impact_curve(gjr)
```

Comparing news impact curves across models reveals whether asymmetric specifications capture economically important leverage effects that symmetric GARCH misses. If the NIC from GARCH and GJR-GARCH are nearly identical, the leverage effect is negligible and the simpler symmetric model suffices. The NIC returns a named tuple with fields `shocks` (grid of ``\varepsilon_{t-1}`` values) and `variance` (corresponding ``\sigma^2_t`` values).

### GARCH Diagnostic Visualization

ARCH, GARCH, EGARCH, and GJR-GARCH models produce a three-panel diagnostic figure via `plot_result()`: return series, conditional volatility, and standardized residuals with ``\pm 2`` standard deviation bounds.

```julia
garch = estimate_garch(ip, 1, 1)
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
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Optimizer iterations |

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
sv = estimate_sv(ip; n_samples=2000, burnin=1000)
report(sv)
```

```@example volatility
# SV with leverage effect
sv_lev = estimate_sv(ip; leverage=true, n_samples=2000, burnin=1000)
report(sv_lev)
```

```@example volatility
# SV with Student-t errors
sv_t = estimate_sv(ip; dist=:studentt, n_samples=2000, burnin=1000)
report(sv_t)
```

The `report()` output displays a posterior summary table with mean, standard deviation, and 95% credible interval for each parameter, followed by model specification details. A posterior mean ``\varphi`` near 0.97 indicates that log-volatility is highly persistent. The ``\sigma_\eta`` parameter governs the variability of the volatility process itself --- larger values produce more volatile volatility paths.

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
sv = estimate_sv(ip; n_samples=2000, burnin=1000)
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
stat, pval, q = arch_lm_test(ip, 5)

# Test standardized residuals after fitting (should fail to reject)
garch = estimate_garch(ip, 1, 1)
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
garch = estimate_garch(ip, 1, 1)
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
garch = estimate_garch(ip, 1, 1)
fc = forecast(garch, 20; conf_level=0.95, n_sim=10000)
report(fc)
```

The forecast report displays a table of point forecasts, standard errors, and confidence interval bounds at each horizon. The point forecast at horizon 1 reflects the current volatility state, while long-horizon forecasts converge to `unconditional_variance(garch)`. For ARCH models, forecasts beyond horizon ``q`` equal the unconditional variance exactly (no lagged variance terms to propagate).

### SV Forecasts

For SV models, each posterior draw provides a full parameter vector ``(\mu, \varphi, \sigma_\eta)`` and the terminal log-volatility ``h_T``. The forecast simulates the log-volatility process forward from the last state for each draw, yielding a posterior predictive distribution of future volatility. The reported intervals are posterior predictive quantiles, not frequentist confidence intervals.

```@example volatility
sv = estimate_sv(ip; n_samples=2000, burnin=1000)
fc_sv = forecast(sv, 20; conf_level=0.95)
report(fc_sv)
```

### Volatility Forecast Visualization

```julia
garch = estimate_garch(ip, 1, 1)
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
garch = estimate_garch(ip, 1, 1)
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
| `stderror(m)` | Standard errors via numerical Hessian and delta method |
| `confint(m)` | Confidence intervals for parameters |
| `vcov(m)` | Variance-covariance matrix of parameter estimates |

```@example volatility
garch = estimate_garch(ip, 1, 1)
nobs(garch)          # Number of observations
loglikelihood(garch) # Maximized log-likelihood
aic(garch)           # AIC for model comparison
bic(garch)           # BIC for model comparison
coef(garch)          # [μ, ω, α₁, ..., αq, β₁, ..., βp]
```

---

## Complete Example

This example estimates all four GARCH-family models on S&P 500 returns, runs diagnostics, compares specifications, and estimates an SV model for comparison.

```@example volatility
# === Step 1: Test for ARCH effects ===
stat, pval, q = arch_lm_test(ip, 5)

# === Step 2: Estimate competing GARCH-family models ===
garch  = estimate_garch(ip, 1, 1)
egarch = estimate_egarch(ip, 1, 1)
gjr    = estimate_gjr_garch(ip, 1, 1)

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
sv = estimate_sv(ip; n_samples=2000, burnin=1000)
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

The S&P 500 returns exhibit strong ARCH effects, confirming the need for volatility modeling. The GARCH(1,1) persistence is typically close to 0.95 for monthly equity returns, meaning volatility shocks take roughly 13 periods to dissipate to half their initial impact. The EGARCH and GJR-GARCH models capture asymmetric leverage effects --- negative shocks increase volatility more than positive shocks of equal magnitude --- which the symmetric GARCH specification misses. After fitting, all models' standardized residuals pass the ARCH-LM test, confirming adequate capture of variance dynamics. The SV model provides an independent Bayesian assessment via the Kim-Shephard-Chib (1998) Gibbs sampler, with posterior credible bands quantifying parameter uncertainty.

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

- Black, F. (1976). Studies of Stock Price Volatility Changes.
  *Proceedings of the 1976 Meetings of the American Statistical Association*, 171--177.

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
  *Journal of Econometrics*, 31(3), 307--327. [DOI](https://doi.org/10.1016/0304-4076(86)90063-1)

- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.
  *Econometrica*, 50(4), 987--1007. [DOI](https://doi.org/10.2307/1912773)

- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks.
  *Journal of Finance*, 48(5), 1779--1801. [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)

- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models.
  *Review of Economic Studies*, 65(3), 361--393. [DOI](https://doi.org/10.1111/1467-937X.00050)

- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
  *Econometrica*, 59(2), 347--370. [DOI](https://doi.org/10.2307/2938260)

- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference.
  *Journal of Econometrics*, 140(2), 425--449. [DOI](https://doi.org/10.1016/j.jeconom.2006.07.008)

- Taylor, S. J. (1986). *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90975-7.
