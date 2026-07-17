# [State-Space Models](@id statespace_page)

The state-space module provides a general, user-specified linear-Gaussian state-space object with maximum-likelihood estimation of its hyper-parameters, Kalman filtering, RTS smoothing, and forecasting. It is a thin, documented front end over the package's consolidated Kalman kernel — no separate filter is introduced — and covers the central-bank staples: **unobserved-components** trend/cycle models, **structural time series** (Harvey 1989), and **time-varying-parameter (TVP) regression** (Durbin & Koopman 2012).

Part of the [Univariate Models](@ref) family.

- **General system** — any single-block observation/state pair via [`StateSpaceModel`](@ref) and [`estimate_statespace`](@ref).
- **Convenience wrappers** — [`local_level`](@ref), [`local_linear_trend`](@ref), [`estimate_tvp_reg`](@ref).
- **Diagnostics & forecasting** — standardized one-step residuals, filtered/smoothed states with variances, and multi-step [`forecast`](@ref).

## Quick Start

```@setup ss
using MacroEconometricModels, Random, Statistics
Random.seed!(42)
nile = load_example(:nile)
```

**Recipe 1: Local-level (random walk + noise) on the Nile**

```@example ss
m = local_level(nile)
report(m)
```

**Recipe 2: Local linear trend**

```@example ss
mt = local_linear_trend(nile)
round.(mt.theta, digits=2)   # σ²_ε, σ²_ξ (level), σ²_ζ (slope)
```

**Recipe 3: Multi-step forecast with predictive standard errors**

```@example ss
fc = forecast(m, 5)
[fc.mean fc.se]
```

**Recipe 4: Time-varying-parameter regression**

```@example ss
Random.seed!(7)
T = 200
x = randn(T)
β = cumsum(0.05 .* randn(T)) .+ 1.0        # random-walk slope
y = 0.5 .+ β .* x .+ 0.3 .* randn(T)
tvp = estimate_tvp_reg(y, reshape(x, :, 1))
round(cor(tvp.smoothed_state[:, 2], β), digits=3)   # recovered slope path
```

---

## The linear-Gaussian state-space form

Every model in this module is a single-block linear-Gaussian system:

```math
y_t = Z \, \alpha_t + d + \varepsilon_t, \qquad \varepsilon_t \sim N(0, H)
```
```math
\alpha_{t+1} = T \, \alpha_t + c + R \, \eta_t, \qquad \eta_t \sim N(0, Q)
```

where:
- ``y_t`` is the ``n_\text{obs} \times 1`` observation vector
- ``\alpha_t`` is the ``n_x \times 1`` latent state
- ``Z`` (``n_\text{obs} \times n_x``) is the observation matrix, ``d`` the observation intercept, ``H`` the observation-noise covariance
- ``T`` (``n_x \times n_x``) is the transition matrix, ``c`` the state intercept, ``R`` the state-noise loading, ``Q`` the state-noise covariance

Filtering and smoothing run through the shared kernel `_kalman_filter!` / `_rts_smoother`; the estimator maximizes the prediction-error-decomposition log-likelihood ``\sum_t \log p(y_t \mid y_{1:t-1})``.

!!! note "Initialization"
    Nonstationary states are seeded with a large-``\kappa`` approximate-diffuse prior (``P_1 = \kappa I``, ``\kappa = 10^6``), matching statsmodels' `UnobservedComponents` default. Pass `init_mode=:stationary` for a fully stationary state (Lyapunov seed) or `init_mode=:diffuse` for the kernel's exact unit-root/stationary split.

---

## Local level

The **local-level** (random-walk-plus-noise) model decomposes a series into a stochastic level plus irregular noise:

```math
y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_\varepsilon); \qquad
\mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)
```

[`local_level`](@ref) estimates ``(\sigma^2_\varepsilon, \sigma^2_\eta)`` by MLE. On the classic Nile river-flow series it recovers the Durbin & Koopman (2012, §2) values ``\hat\sigma^2_\varepsilon \approx 15099`` and ``\hat\sigma^2_\eta \approx 1469``:

```@example ss
m = local_level(nile)
round.(m.theta, digits=1)
```

The **signal-to-noise ratio** ``\hat\sigma^2_\eta / \hat\sigma^2_\varepsilon`` is small (≈0.1), so the smoothed level ``\mu_{t|T}`` is a heavily damped version of the data — capturing the well-known level shift around 1899 without chasing high-frequency noise.

```julia
plot_result(m)   # observed vs filtered vs smoothed level with 95% band
```

### Return values

| Field | Type | Description |
|---|---|---|
| `theta` | `Vector{T}` | Estimated hyper-parameters (natural variances for the wrappers) |
| `loglik` | `T` | Maximized prediction-error-decomposition log-likelihood |
| `filtered_state` / `filtered_cov` | `Matrix` / `Array{T,3}` | ``a_{t|t}``, ``P_{t|t}`` |
| `smoothed_state` / `smoothed_cov` | `Matrix` / `Array{T,3}` | ``a_{t|T}``, ``P_{t|T}`` |
| `innovations` / `std_residuals` | `Matrix{T}` | One-step errors ``v_t`` and standardized ``v_t/\sqrt{F_t}`` |

---

## Local linear trend

The **local-linear-trend** model adds a stochastic slope ``\beta_t`` to the level, so the trend can drift in both level and growth rate:

```math
y_t = \mu_t + \varepsilon_t; \qquad
\mu_{t+1} = \mu_t + \beta_t + \xi_t; \qquad
\beta_{t+1} = \beta_t + \zeta_t
```

with ``\varepsilon_t \sim N(0,\sigma^2_\varepsilon)``, ``\xi_t \sim N(0,\sigma^2_\xi)``, ``\zeta_t \sim N(0,\sigma^2_\zeta)``. The state is ``[\mu_t, \beta_t]``.

```@example ss
mt = local_linear_trend(nile)
report(mt)
```

Setting ``\sigma^2_\zeta = 0`` collapses this to a smooth deterministic-slope trend; setting ``\sigma^2_\xi = 0`` gives the integrated-random-walk (smooth-trend) model.

---

## Time-varying-parameter regression

[`estimate_tvp_reg`](@ref) fits a regression whose coefficients follow a random walk:

```math
y_t = X_t \, \beta_t + \varepsilon_t, \quad \varepsilon_t \sim N(0,\sigma^2_\varepsilon); \qquad
\beta_{t+1} = \beta_t + \eta_t, \quad \eta_t \sim N(0, \operatorname{diag}(\sigma^2_\eta))
```

The observation matrix ``Z_t = X_t`` is time-varying, so the coefficient path ``\beta_t`` is recovered as the smoothed state. A leading intercept column is added by default (`intercept=true`).

```@example ss
Random.seed!(7)
T = 200
x = randn(T)
β = cumsum(0.05 .* randn(T)) .+ 1.0
y = 0.5 .+ β .* x .+ 0.3 .* randn(T)
tvp = estimate_tvp_reg(y, reshape(x, :, 1))
β_hat = tvp.smoothed_state[:, 2]           # column 2 = slope (column 1 = intercept)
round(cor(β_hat, β), digits=3)
```

The estimated hyper-parameters are the irregular variance ``\sigma^2_\varepsilon`` and the per-coefficient random-walk variances; larger ``\sigma^2_\eta`` lets a coefficient move more freely over the sample.

---

## Custom systems

For an arbitrary model, pass a **builder** closure `build(θ) -> NamedTuple(Z, H, T, Q, ...)` and a starting vector to [`estimate_statespace`](@ref). The builder is responsible for the positivity of variance blocks (typically by exponentiating log-variance parameters). Here is an AR(1)-plus-noise unobserved-components model:

```@example ss
Random.seed!(11)
T = 300
φ = 0.7
α = zeros(T)
for t in 2:T; α[t] = φ * α[t-1] + randn(); end
z = α .+ 0.5 .* randn(T)
build = θ -> (Z = reshape([1.0], 1, 1), H = reshape([exp(θ[3])], 1, 1),
              T = reshape([tanh(θ[1])], 1, 1), Q = reshape([exp(θ[2])], 1, 1))
m_ar = estimate_statespace(build, [0.3, 0.0, 0.0], z;
                           param_names = ["atanh(φ)", "logσ²_η", "logσ²_ε"])
round(tanh(m_ar.theta[1]), digits=2)   # recovered persistence φ
```

A fixed-matrix `StateSpaceModel(Z, H, T, Q; ...)` builds an unfitted spec that `estimate_statespace(ss, y)` filters and smooths without optimization.

---

## Complete Example

Estimate the Nile local level, inspect residual diagnostics, and forecast five years ahead:

```@example ss
m = local_level(nile)

# One-step standardized residuals should be white noise on a well-specified model.
r = vec(m.std_residuals); r = r[.!isnan.(r)]
lb = ljung_box_test(r; lags=10)
round(lb.pvalue, digits=3)
```

```@example ss
fc = forecast(m, 5)
[fc.mean fc.se]        # flat point forecast (random walk), widening bands
```

```@example ss
refs(m)
```

---

## Common Pitfalls

1. **Log-likelihood is not directly comparable to exact-diffuse software.** The large-``\kappa`` prior keeps a ``-\tfrac12\log\kappa`` offset for the diffuse observation, whereas exact-diffuse implementations (KFAS, statsmodels with `initialization='diffuse'`) drop it. The MLE variance estimates and filtered states still match to within a fraction of a percent.
2. **Variance components hitting zero.** When a variance is truly zero (e.g. a fixed intercept in a TVP regression), its log-parameter drifts to ``-\infty`` and the optimizer reports `converged=false` even though the fit is correct. Check the recovered paths, not just the flag.
3. **Smoothed states do not exactly satisfy the observation equation.** Use a relaxed tolerance if you compare ``Z\,\alpha_{t|T} + d`` against ``y_t``.
4. **Custom builders must keep `H` and `Q` positive.** Parameterize variances as `exp(θ)` and persistences as `tanh(θ)`; a raw `clamp` flattens the gradient at the boundary and stalls `LBFGS`.

---

## References

- Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press. [doi:10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press. [doi:10.1017/CBO9781107049994](https://doi.org/10.1017/CBO9781107049994)
