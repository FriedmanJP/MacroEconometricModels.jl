# Volatility Models

This page covers univariate volatility modeling: ARCH, GARCH (including EGARCH and GJR-GARCH), and Stochastic Volatility (SV) models. These models capture time-varying conditional variance — a pervasive feature of financial and macroeconomic time series.

## Quick Start

```julia
using MacroEconometricModels
using Random

# S&P 500 monthly returns (FRED-MD)
fred = load_example(:fred_md)
sp_idx = findfirst(v -> occursin("S&P", v) && occursin("500", v), varnames(fred))
y = filter(isfinite, apply_tcode(fred[:, varnames(fred)[sp_idx]], 5))

# ARCH(5) — Engle (1982)
arch = estimate_arch(y, 5)

# GARCH(1,1) — Bollerslev (1986)
garch = estimate_garch(y, 1, 1)

# EGARCH(1,1) — Nelson (1991)
egarch = estimate_egarch(y, 1, 1)

# GJR-GARCH(1,1) — Glosten, Jagannathan & Runkle (1993)
gjr = estimate_gjr_garch(y, 1, 1)

# Stochastic Volatility — Taylor (1986), Kim-Shephard-Chib (1998)
Random.seed!(42)
sv = estimate_sv(y; n_samples=2000, burnin=1000)

# Diagnostics
arch_lm_test(y, 5)         # ARCH-LM test
ljung_box_squared(y, 10)   # Ljung-Box on squared residuals
nic = news_impact_curve(garch)  # News impact curve

# Forecast 20 steps ahead
fc = forecast(garch, 20; conf_level=0.95)
```

---

## ARCH Models

### Theory

The Autoregressive Conditional Heteroskedasticity (ARCH) model of Engle (1982) allows the conditional variance to depend on past squared innovations. The ARCH(``q``) specification is:

```math
y_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim \mathcal{N}(0, 1)
```

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon^2_{t-i}
```

where
- ``y_t`` is the observed time series
- ``\mu`` is the conditional mean (intercept)
- ``\varepsilon_t`` is the mean-corrected innovation
- ``\sigma^2_t`` is the conditional variance at time ``t``
- ``\omega > 0`` is the variance intercept
- ``\alpha_i \geq 0`` are the ARCH coefficients
- ``z_t`` is a standardized innovation

**Stationarity condition**: The ARCH(``q``) process is covariance stationary if ``\sum_{i=1}^{q} \alpha_i < 1``.

**Unconditional variance**: Under stationarity, ``\text{Var}(\varepsilon_t) = \omega / (1 - \sum_{i=1}^{q} \alpha_i)``.

### Estimation

ARCH models are estimated by maximum likelihood (MLE) using two-stage optimization:

1. **Stage 1** (NelderMead): Derivative-free search to find a good starting region.
2. **Stage 2** (L-BFGS): Gradient-based refinement from the Stage 1 solution.

Parameters are log-transformed internally to enforce positivity constraints (``\omega > 0``, ``\alpha_i \geq 0``) without constrained optimization.

```julia
# Estimate ARCH(5) model
arch = estimate_arch(y, 5)

# Access estimated parameters
arch.omega      # Variance intercept
arch.alpha      # ARCH coefficients [α₁, ..., α₅]
arch.mu         # Mean
arch.loglik     # Log-likelihood
```

### Diagnostics

Two diagnostic tests check whether ARCH effects have been adequately captured:

**ARCH-LM Test** (Engle 1982): Tests for remaining ARCH effects in model residuals or raw data.

```julia
# Test raw data for ARCH effects (H₀: no ARCH effects)
stat, pval, q = arch_lm_test(y, 5)

# Test standardized residuals after fitting (should fail to reject)
stat, pval, q = arch_lm_test(arch, 5)
```

The test regresses squared residuals on ``q`` of their own lags and computes ``TR^2 \sim \chi^2(q)``. Rejection of ``H_0`` indicates ARCH effects are present (or remain after fitting).

**Ljung-Box Test on Squared Residuals**: Tests for serial correlation in ``z_t^2``.

```julia
stat, pval, K = ljung_box_squared(arch, 10)
```

The test statistic is ``Q = n(n+2) \sum_{k=1}^{K} \hat{\rho}^2_k / (n - k) \sim \chi^2(K)``, where ``\hat{\rho}_k`` is the sample autocorrelation of squared standardized residuals at lag ``k``. Failure to reject indicates the model has adequately captured the variance dynamics.

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

### GARCH(p,q) — Bollerslev (1986)

The Generalized ARCH model extends ARCH by including lagged conditional variances:

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon^2_{t-i} + \sum_{j=1}^{p} \beta_j \sigma^2_{t-j}
```

where
- ``\omega > 0`` is the variance intercept
- ``\alpha_i \geq 0`` are the ARCH coefficients (impact of past shocks)
- ``\beta_j \geq 0`` are the GARCH coefficients (variance persistence)
- ``p`` is the GARCH order (lagged variances) and ``q`` is the ARCH order (lagged squared residuals)

**Stationarity condition**: ``\sum_{i=1}^{q} \alpha_i + \sum_{j=1}^{p} \beta_j < 1``.

**Unconditional variance**: ``\sigma^2 = \omega / (1 - \sum \alpha_i - \sum \beta_j)``.

The GARCH(1,1) is the most widely used specification in practice, capturing the key empirical regularity of volatility clustering with just three parameters.

```julia
# Estimate GARCH(1,1) — the workhorse specification
garch = estimate_garch(y, 1, 1)

# Persistence: how quickly volatility reverts to its long-run level
persistence(garch)              # α₁ + β₁ (close to 1 = slow reversion)
halflife(garch)                 # Half-life in periods
unconditional_variance(garch)   # Long-run variance level
```

### EGARCH(p,q) — Nelson (1991)

The Exponential GARCH models the log of conditional variance, ensuring positivity without parameter constraints and allowing asymmetric responses to positive and negative shocks:

```math
\log(\sigma^2_t) = \omega + \sum_{i=1}^{q} \alpha_i (|z_{t-i}| - \mathbb{E}|z|) + \sum_{i=1}^{q} \gamma_i z_{t-i} + \sum_{j=1}^{p} \beta_j \log(\sigma^2_{t-j})
```

where
- ``z_t = \varepsilon_t / \sigma_t`` are standardized residuals
- ``\alpha_i`` captures the magnitude (symmetric) effect of shocks
- ``\gamma_i`` captures the sign (asymmetric/leverage) effect — typically ``\gamma_i < 0`` means negative shocks increase volatility more than positive shocks of equal magnitude
- ``\beta_j`` governs persistence of log-variance
- ``\mathbb{E}|z| = \sqrt{2/\pi}`` for standard normal innovations

**Stationarity condition**: ``\sum_{j=1}^{p} \beta_j < 1`` (in log-variance, unconditional parameters).

**Unconditional variance**: ``\sigma^2 = \exp(\omega / (1 - \sum \beta_j))``.

```julia
# Estimate EGARCH(1,1)
egarch = estimate_egarch(y, 1, 1)

# Leverage parameters (γ < 0 → "leverage effect")
egarch.gamma    # Leverage coefficients
```

### GJR-GARCH(p,q) — Glosten, Jagannathan & Runkle (1993)

The GJR-GARCH (also called Threshold GARCH) adds an indicator function for negative shocks:

```math
\sigma^2_t = \omega + \sum_{i=1}^{q} (\alpha_i + \gamma_i \mathbb{1}(\varepsilon_{t-i} < 0)) \varepsilon^2_{t-i} + \sum_{j=1}^{p} \beta_j \sigma^2_{t-j}
```

where
- ``\gamma_i \geq 0`` are leverage parameters
- ``\mathbb{1}(\varepsilon_{t-i} < 0) = 1`` when past shocks are negative

When ``\gamma_i > 0``, negative shocks have a larger impact on future variance than positive shocks of equal magnitude. This captures the empirical "leverage effect" first documented by Black (1976): stock price declines increase financial leverage, which in turn increases equity volatility.

**Stationarity condition**: ``\sum \alpha_i + \sum \gamma_i / 2 + \sum \beta_j < 1``.

**Unconditional variance**: ``\sigma^2 = \omega / (1 - \sum \alpha_i - \sum \gamma_i / 2 - \sum \beta_j)``.

```julia
# Estimate GJR-GARCH(1,1)
gjr = estimate_gjr_garch(y, 1, 1)

# Leverage effect: γ > 0 means negative shocks increase variance more
gjr.gamma    # Leverage coefficients
```

### News Impact Curve

The news impact curve (NIC) shows how a shock ``\varepsilon_{t-1}`` maps to the next-period conditional variance ``\sigma^2_t``, holding all other information constant at the unconditional level. For symmetric models (ARCH, GARCH), the NIC is a parabola centered at zero. For asymmetric models (EGARCH, GJR-GARCH), the NIC is steeper for negative shocks.

```julia
# Compute news impact curves
nic_garch  = news_impact_curve(garch)
nic_egarch = news_impact_curve(egarch; range=(-3.0, 3.0), n_points=200)
nic_gjr    = news_impact_curve(gjr)

# Returns named tuple: (shocks=Vector, variance=Vector)
nic_garch.shocks     # Grid of εₜ₋₁ values
nic_garch.variance   # Corresponding σ²ₜ values
```

Comparing news impact curves across models reveals whether asymmetric specifications (EGARCH, GJR-GARCH) capture economically important leverage effects that symmetric GARCH misses. If the NIC from GARCH and GJR-GARCH are nearly identical, the leverage effect is negligible and the simpler symmetric model suffices.

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

### Theory

The stochastic volatility (SV) model of Taylor (1986) treats the log-variance as a latent autoregressive process:

```math
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)
```

```math
h_t = \mu + \varphi (h_{t-1} - \mu) + \sigma_\eta \eta_t, \qquad \eta_t \sim \mathcal{N}(0, 1)
```

where
- ``h_t`` is the log-variance at time ``t``
- ``\mu`` is the log-variance level (unconditional mean of ``h_t``)
- ``\varphi \in (-1, 1)`` is the persistence parameter
- ``\sigma_\eta > 0`` is the volatility of volatility
- ``\varepsilon_t`` and ``\eta_t`` are independent standard normal innovations

The SV model differs fundamentally from GARCH in that volatility has its own independent source of randomness (``\eta_t``), making it a state-space model with a non-Gaussian observation equation. This provides greater flexibility in capturing empirical volatility dynamics, but precludes closed-form likelihood evaluation.

### Variants

Three SV variants are available:

**Basic SV** (`leverage=false`, `dist=:normal`): The standard specification above.

**SV with Leverage** (`leverage=true`): Allows correlation between return and volatility innovations:

```math
\begin{pmatrix} \varepsilon_t \\ \eta_t \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)
```

When ``\rho < 0`` (the typical case for equities), negative returns are associated with increases in volatility, analogous to the leverage effect in EGARCH and GJR-GARCH models.

**SV with Student-t Errors** (`dist=:studentt`): Replaces the Gaussian observation equation with Student-t innovations to accommodate heavier tails:

```math
y_t = \exp(h_t / 2) \, \varepsilon_t, \qquad \varepsilon_t \sim t_\nu
```

where ``\nu > 2`` is the degrees of freedom parameter.

### Priors

The SV model is estimated via the Kim-Shephard-Chib (1998) Gibbs sampler with the Omori et al. (2007) 10-component mixture approximation. The default priors are:

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| ``\mu`` | ``\mathcal{N}(0, 10)`` | Weakly informative for log-variance level |
| ``\varphi`` | ``\text{Beta}(20, 1.5) \to (-1, 1)`` | Concentrates mass near 1 (high persistence), ensures stationarity |
| ``\sigma_\eta`` | ``\text{HalfNormal}(1)`` | Positive, moderately informative for vol-of-vol |
| ``\rho`` (leverage) | ``\text{Uniform}(-1, 1)`` | Uninformative over correlation range |
| ``\nu`` (Student-t) | ``\text{Exponential}(0.1) + 2`` | Ensures ``\nu > 2`` (finite variance) |

### Estimation

```julia
# Basic SV model
sv = estimate_sv(y; n_samples=2000, burnin=1000)

# SV with leverage effect
sv_lev = estimate_sv(y; leverage=true, n_samples=2000, burnin=1000)

# SV with Student-t errors
sv_t = estimate_sv(y; dist=:studentt, n_samples=2000, burnin=1000)

# Access posterior summaries
mean(sv.mu_post)          # Posterior mean of μ
mean(sv.phi_post)         # Posterior mean of φ
mean(sv.sigma_eta_post)   # Posterior mean of σ_η

# Posterior volatility (time series)
sv.volatility_mean        # Posterior mean of exp(hₜ) at each t
sv.volatility_quantiles   # Quantiles (T × n_quantiles matrix)
sv.quantile_levels        # Default: [0.025, 0.5, 0.975]

# Latent log-volatility draws
sv.h_draws                # n_samples × T matrix of posterior hₜ draws
```

!!! note "Technical Note"
    The Kim-Shephard-Chib (1998) Gibbs sampler approximates the non-Gaussian observation equation ``\log y_t^2 = h_t + \log \varepsilon_t^2`` using a 10-component Gaussian mixture (Omori et al., 2007). Each Gibbs iteration: (1) samples the mixture indicators conditional on ``h``, (2) samples ``h_{1:T}`` via the simulation smoother conditional on parameters and indicators, and (3) samples ``(\mu, \varphi, \sigma_\eta)`` from their conditional posteriors. Typical run times are under 30 seconds for ``T = 500`` with 2000 posterior draws.

### SVModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original data |
| `h_draws` | `Matrix{T}` | Latent log-volatility draws (n_samples × T) |
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

## Volatility Forecasting

All volatility models support multi-step ahead forecasting via `forecast()`. ARCH and GARCH-family models use simulation-based confidence intervals; SV models use posterior predictive simulation from MCMC draws.

### ARCH/GARCH Forecasts

```julia
# Forecast 20 steps ahead
fc = forecast(garch, 20; conf_level=0.95, n_sim=10000)

# Point forecasts converge to unconditional variance
fc.forecast     # Vector of length 20
fc.ci_lower     # Lower CI bound
fc.ci_upper     # Upper CI bound
fc.se           # Standard errors

# Compare unconditional variance with long-horizon forecast
unconditional_variance(garch)
fc.forecast[end]  # Should be close for large h
```

For stationary GARCH processes, multi-step forecasts converge geometrically to the unconditional variance at rate equal to the persistence parameter. The speed of convergence is measured by the half-life: ``\text{halflife} = \log(0.5) / \log(\text{persistence})``.

Confidence intervals are constructed by simulating ``n`` paths forward from the last observed state, generating the distribution of future conditional variances. For ARCH models, forecasts beyond horizon ``q`` equal the unconditional variance exactly (no lagged variance terms to propagate).

### SV Forecasts

```julia
# Posterior predictive forecast from SV model
fc_sv = forecast(sv, 20; conf_level=0.95)

fc_sv.forecast     # Posterior mean forecast
fc_sv.ci_lower     # 2.5th percentile
fc_sv.ci_upper     # 97.5th percentile
```

For SV models, each posterior draw provides a full parameter vector ``(\mu, \varphi, \sigma_\eta)`` and the terminal log-volatility ``h_T``. The forecast simulates the log-volatility process forward from the last state for each draw, yielding a posterior predictive distribution of future volatility. The reported intervals are posterior predictive quantiles, not frequentist confidence intervals.

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
| `arch_order(m)` | ``q`` | ``q`` | ``q`` | ``q`` | — |
| `garch_order(m)` | — | ``p`` | ``p`` | ``p`` | — |

In the table, ``p`` denotes `persistence(m)`. The half-life returns `Inf` if the process is non-stationary (persistence ``\geq 1``).

```julia
persistence(garch)              # 0.95 → high persistence
halflife(garch)                 # ≈ 13.5 periods
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

```julia
nobs(garch)          # Number of observations
loglikelihood(garch) # Maximized log-likelihood
aic(garch)           # AIC for model comparison
bic(garch)           # BIC for model comparison
coef(garch)          # [μ, ω, α₁, ..., αq, β₁, ..., βp]
```

---

## Complete Example

This example estimates all four GARCH-family models on the same data, compares their news impact curves and forecasts, runs diagnostics, and estimates an SV model for comparison.

```julia
using MacroEconometricModels
using Random
using Statistics

# S&P 500 monthly returns (FRED-MD)
fred = load_example(:fred_md)
sp_idx = findfirst(v -> occursin("S&P", v) && occursin("500", v), varnames(fred))
y = filter(isfinite, apply_tcode(fred[:, varnames(fred)[sp_idx]], 5))

println("S&P 500 returns: T=$(length(y))")

# === Step 1: Test for ARCH effects ===
stat, pval, q = arch_lm_test(y, 5)
println("\nARCH-LM test (q=5):")
println("  Statistic: ", round(stat, digits=2))
println("  P-value: ", round(pval, digits=6))

stat2, pval2, K = ljung_box_squared(y, 10)
println("\nLjung-Box squared (K=10):")
println("  Statistic: ", round(stat2, digits=2))
println("  P-value: ", round(pval2, digits=6))

# === Step 2: Estimate competing models ===
garch   = estimate_garch(y, 1, 1)
egarch  = estimate_egarch(y, 1, 1)
gjr     = estimate_gjr_garch(y, 1, 1)

println("\n" * "="^60)
println("Model Comparison")
println("="^60)
println("              AIC         BIC     Persistence")
println("  GARCH:   ", round(aic(garch), digits=1),
        "    ", round(bic(garch), digits=1),
        "    ", round(persistence(garch), digits=4))
println("  EGARCH:  ", round(aic(egarch), digits=1),
        "    ", round(bic(egarch), digits=1),
        "    ", round(persistence(egarch), digits=4))
println("  GJR:     ", round(aic(gjr), digits=1),
        "    ", round(bic(gjr), digits=1),
        "    ", round(persistence(gjr), digits=4))

# === Step 3: Compare news impact curves ===
nic_g  = news_impact_curve(garch)
nic_e  = news_impact_curve(egarch)
nic_j  = news_impact_curve(gjr)

println("\nNews Impact at ε = -2 vs ε = +2:")
idx_neg = findfirst(x -> x >= -2.0, nic_g.shocks)
idx_pos = findfirst(x -> x >= 2.0, nic_g.shocks)

println("  GARCH:  σ²(-2) = ", round(nic_g.variance[idx_neg], digits=4),
        "   σ²(+2) = ", round(nic_g.variance[idx_pos], digits=4))
println("  EGARCH: σ²(-2) = ", round(nic_e.variance[idx_neg], digits=4),
        "   σ²(+2) = ", round(nic_e.variance[idx_pos], digits=4))
println("  GJR:    σ²(-2) = ", round(nic_j.variance[idx_neg], digits=4),
        "   σ²(+2) = ", round(nic_j.variance[idx_pos], digits=4))

# === Step 4: Check residual diagnostics ===
println("\nResidual ARCH-LM test (q=5):")
for (name, m) in [("GARCH", garch), ("EGARCH", egarch), ("GJR", gjr)]
    _, p, _ = arch_lm_test(m, 5)
    status = p > 0.05 ? "Pass" : "FAIL"
    println("  $name: p=$(round(p, digits=4))  $status")
end

# === Step 5: Forecast volatility ===
H = 20
fc_g = forecast(garch, H)
fc_e = forecast(egarch, H)
fc_j = forecast(gjr, H)

println("\nVolatility forecasts (conditional variance):")
println("  h    GARCH    EGARCH   GJR      Uncond")
for h_idx in [1, 5, 10, 20]
    println("  $h_idx    ",
            round(fc_g.forecast[h_idx], digits=4), "  ",
            round(fc_e.forecast[h_idx], digits=4), "  ",
            round(fc_j.forecast[h_idx], digits=4), "  ",
            round(unconditional_variance(garch), digits=4))
end

# === Step 6: Stochastic volatility for comparison ===
println("\nEstimating SV model via KSC Gibbs sampler...")
Random.seed!(42)
sv = estimate_sv(y; n_samples=2000, burnin=1000)

println("SV posterior summary:")
println("  μ (log-vol level):   ", round(mean(sv.mu_post), digits=3),
        " [", round(quantile(sv.mu_post, 0.025), digits=3),
        ", ", round(quantile(sv.mu_post, 0.975), digits=3), "]")
println("  φ (persistence):    ", round(mean(sv.phi_post), digits=3),
        " [", round(quantile(sv.phi_post, 0.025), digits=3),
        ", ", round(quantile(sv.phi_post, 0.975), digits=3), "]")
println("  σ_η (vol of vol):   ", round(mean(sv.sigma_eta_post), digits=3),
        " [", round(quantile(sv.sigma_eta_post, 0.025), digits=3),
        ", ", round(quantile(sv.sigma_eta_post, 0.975), digits=3), "]")

# SV forecast
fc_sv = forecast(sv, H)
println("\nSV forecast at h=1: ", round(fc_sv.forecast[1], digits=4))
println("SV forecast at h=20: ", round(fc_sv.forecast[end], digits=4))
```

The S&P 500 returns exhibit strong ARCH effects, confirming the need for volatility modeling. The GARCH(1,1) persistence is typically close to 0.95 for equity returns, meaning volatility shocks are highly persistent. The news impact curves reveal asymmetry: for EGARCH and GJR-GARCH, a negative leverage parameter confirms that bad news increases volatility more than good news of equal magnitude — for symmetric GARCH, the NIC is symmetric by construction. All models' standardized residuals should pass the ARCH-LM test after fitting, confirming that the conditional variance dynamics are adequately captured. The SV model provides an independent, Bayesian assessment of the volatility dynamics via the Kim-Shephard-Chib (1998) Gibbs sampler.

---

### See Also

- [Hypothesis Tests](hypothesis_tests.md) -- ARCH-LM test and diagnostic testing
- [ARIMA Models](arima.md) -- Mean equation modeling before volatility estimation
- [API Reference](api_functions.md) -- Complete function signatures

## References

- Black, Fischer. 1976. "Studies of Stock Price Volatility Changes." *Proceedings of the 1976 Meetings of the American Statistical Association*, 171–177.
- Bollerslev, Tim. 1986. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31 (3): 307–327. [https://doi.org/10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- Engle, Robert F. 1982. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica* 50 (4): 987–1007. [https://doi.org/10.2307/1912773](https://doi.org/10.2307/1912773)
- Glosten, Lawrence R., Ravi Jagannathan, and David E. Runkle. 1993. "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48 (5): 1779–1801. [https://doi.org/10.1111/j.1540-6261.1993.tb05128.x](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)
- Nelson, Daniel B. 1991. "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59 (2): 347–370. [https://doi.org/10.2307/2938260](https://doi.org/10.2307/2938260)
- Kim, Sangjoon, Neil Shephard, and Siddhartha Chib. 1998. "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies* 65 (3): 361–393. [https://doi.org/10.1111/1467-937X.00050](https://doi.org/10.1111/1467-937X.00050)
- Omori, Yasuhiro, Siddhartha Chib, Neil Shephard, and Jouchi Nakajima. 2007. "Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference." *Journal of Econometrics* 140 (2): 425–449. [https://doi.org/10.1016/j.jeconom.2006.07.008](https://doi.org/10.1016/j.jeconom.2006.07.008)
- Taylor, Stephen J. 1986. *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90975-7.
