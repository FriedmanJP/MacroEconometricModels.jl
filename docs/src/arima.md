# [ARIMA Models](@id arima_page)

**MacroEconometricModels.jl** provides a complete suite for estimating, diagnosing, and forecasting with univariate ARIMA-class models. The implementation covers the full Box-Jenkins (1976) workflow from model identification through order selection and out-of-sample forecasting.

- **AR(p)**: Autoregressive models estimated via OLS or exact MLE
- **MA(q)**: Moving average models estimated via CSS, exact MLE, or CSS-MLE
- **ARMA(p,q)**: Combined autoregressive-moving average with three estimation methods
- **ARIMA(p,d,q)**: Integrated ARMA for non-stationary series via ``d``-fold differencing
- **Forecasting**: Multi-step point forecasts with ``\psi``-weight confidence intervals
- **Order Selection**: Grid search over information criteria and automatic `auto_arima`
- **StatsAPI Interface**: Full `coef`, `nobs`, `predict`, `fit`, `residuals`, `aic`, `bic` compatibility

## Quick Start

**Recipe 1: Estimate an AR(2) on industrial production growth**

```julia
using MacroEconometricModels

# Industrial production growth (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

ar = estimate_ar(y, 2)
report(ar)
```

**Recipe 2: Fit an ARMA(1,1) and forecast 12 months ahead**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

arma = estimate_arma(y, 1, 1)
fc = forecast(arma, 12; conf_level=0.95)
report(fc)
```

**Recipe 3: ARIMA(1,1,0) on a non-stationary level series**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))

arima = estimate_arima(y_level, 1, 1, 0)
report(arima)
```

**Recipe 4: Automatic order selection via grid search**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

sel = select_arima_order(y, 4, 4)
report(sel)
```

**Recipe 5: Fully automatic model selection with `auto_arima`**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))

best = auto_arima(y_level; max_p=5, max_q=5, max_d=2, criterion=:bic)
report(best)
```

**Recipe 6: Forecast and visualize**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

ar = estimate_ar(y, 2)
fc = forecast(ar, 20)
p = plot_result(fc; history=y, n_history=30)
```

---

## The AR(p) Model

An **autoregressive model** of order ``p`` expresses the current observation as a linear combination of its own past values plus a white noise innovation. AR models are the workhorse of univariate time series analysis and serve as building blocks for VAR, BVAR, and local projection methods.

```math
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t
```

where:
- ``y_t`` is the observed value at time ``t``
- ``c`` is the intercept (constant term)
- ``\phi_1, \ldots, \phi_p`` are the autoregressive coefficients
- ``\varepsilon_t \sim \text{WN}(0, \sigma^2)`` is white noise
- ``p`` is the lag order

In lag-operator notation: ``\phi(L) y_t = c + \varepsilon_t`` where ``\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p``.

### Stationarity

The process is **covariance stationary** if all roots of the characteristic polynomial ``\phi(z) = 0`` lie outside the unit circle. Equivalently, all eigenvalues of the companion matrix

```math
F = \begin{bmatrix}
\phi_1 & \phi_2 & \cdots & \phi_{p-1} & \phi_p \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{bmatrix}
```

where:
- ``F`` is the ``p \times p`` companion matrix
- ``\phi_i`` are the AR coefficients placed in the first row

satisfy ``|\lambda_i(F)| < 1`` for all ``i``. The estimator checks this condition and truncates coefficients toward stationarity when initializing optimization.

### Estimation

AR models support two estimation methods. **OLS** (`:ols`, default) constructs the lagged regressor matrix and applies ordinary least squares --- consistent and asymptotically efficient for stationary processes (Hamilton 1994, Section 5.2). **MLE** (`:mle`) maximizes the exact Gaussian log-likelihood via the Kalman filter (see [Exact MLE via Kalman Filter](@ref kalman_mle) below).

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

# OLS estimation (default)
ar_ols = estimate_ar(y, 2)
report(ar_ols)

# MLE estimation
ar_mle = estimate_ar(y, 2; method=:mle)
report(ar_mle)
```

The AR(2) model on IP growth captures the short-run momentum (positive ``\phi_1``) and mean-reversion (negative ``\phi_2``) that characterize industrial production dynamics. OLS and MLE produce nearly identical estimates for this large sample, but MLE provides exact inference through proper likelihood treatment.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:ols` | Estimation method (`:ols` or `:mle`) |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |

### ARModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original time series |
| `p` | `Int` | AR order |
| `c` | `T` | Intercept (constant term) |
| `phi` | `Vector{T}` | AR coefficients ``[\phi_1, \ldots, \phi_p]`` |
| `sigma2` | `T` | Innovation variance ``\hat{\sigma}^2`` |
| `residuals` | `Vector{T}` | Estimated residuals |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `method` | `Symbol` | Estimation method (`:ols` or `:mle`) |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of optimization iterations (0 for OLS) |

---

## The MA(q) Model

A **moving average model** of order ``q`` expresses the current observation as a linear function of current and past white noise innovations. MA models naturally arise as the Wold representation of any covariance-stationary process (Hamilton 1994, Chapter 4).

```math
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
```

where:
- ``y_t`` is the observed value at time ``t``
- ``c`` is the intercept
- ``\theta_1, \ldots, \theta_q`` are the moving average coefficients
- ``\varepsilon_t \sim \text{WN}(0, \sigma^2)`` is white noise
- ``q`` is the MA order

In lag-operator notation: ``y_t = c + \theta(L) \varepsilon_t`` where ``\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q``.

### Invertibility

The MA process is **invertible** if all roots of ``\theta(z) = 0`` lie outside the unit circle. Invertibility guarantees a unique MA representation and permits expressing the process in autoregressive form. The estimator enforces invertibility by truncating initial MA coefficients when roots approach the unit circle.

### Estimation

MA parameters cannot be estimated by OLS because the innovations ``\varepsilon_t`` are unobserved. Three methods are available:

- **CSS** (`:css`): Conditional Sum of Squares --- fast, approximate; conditions on initial residuals being zero
- **MLE** (`:mle`): Exact MLE via Kalman filter --- efficient but sensitive to starting values
- **CSS-MLE** (`:css_mle`, default): CSS initialization followed by MLE refinement, combining robustness with efficiency

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

ma = estimate_ma(y, 1; method=:css_mle)
report(ma)
```

The MA(1) coefficient ``\theta_1`` captures one-period serial correlation in shocks to industrial production growth. A positive ``\theta_1`` indicates that a positive surprise this month raises the forecast for next month beyond the unconditional mean.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:css_mle` | Estimation method (`:css`, `:mle`, or `:css_mle`) |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |
| `max_iter` | `Int` | `500` | Maximum optimization iterations |

### MAModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original time series |
| `q` | `Int` | MA order |
| `c` | `T` | Intercept |
| `theta` | `Vector{T}` | MA coefficients ``[\theta_1, \ldots, \theta_q]`` |
| `sigma2` | `T` | Innovation variance |
| `residuals` | `Vector{T}` | Estimated residuals |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `method` | `Symbol` | Estimation method (`:css`, `:mle`, `:css_mle`) |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of optimization iterations |

---

## The ARMA(p,q) Model

The **ARMA(p,q) model** combines autoregressive and moving average components, providing a parsimonious representation of both persistent dynamics and transient shock propagation. The ARMA class nests AR and MA as special cases and forms the stationary core of the ARIMA framework.

```math
\phi(L) \, y_t = c + \theta(L) \, \varepsilon_t
```

where:
- ``\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p`` is the autoregressive lag polynomial
- ``\theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q`` is the moving average lag polynomial
- ``c`` is the intercept
- ``\varepsilon_t \sim \text{WN}(0, \sigma^2)`` is white noise

The process is stationary when all roots of ``\phi(z) = 0`` lie outside the unit circle, and invertible when all roots of ``\theta(z) = 0`` lie outside the unit circle.

!!! note "Technical Note"
    CSS (Conditional Sum of Squares) conditions on initial residuals being zero, introducing bias in small samples. MLE via the Kalman filter provides exact inference by properly handling initialization but is computationally more expensive and can be sensitive to starting values. The default `:css_mle` combines both: CSS provides robust starting values, then MLE refines to the exact optimum. For pure AR models, OLS is equivalent to CSS and is preferred for speed.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

arma = estimate_arma(y, 1, 1; method=:css_mle)
report(arma)
```

The ARMA(1,1) model captures both the autoregressive persistence in IP growth (through ``\phi_1``) and the one-period shock amplification (through ``\theta_1``). An ARMA(1,1) often achieves a lower BIC than a pure AR model of comparable fit because the MA component absorbs short-run dynamics that would otherwise require additional AR lags.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:css_mle` | Estimation method (`:css`, `:mle`, or `:css_mle`) |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |
| `max_iter` | `Int` | `500` | Maximum optimization iterations |

### ARMAModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original time series |
| `p` | `Int` | AR order |
| `q` | `Int` | MA order |
| `c` | `T` | Intercept |
| `phi` | `Vector{T}` | AR coefficients ``[\phi_1, \ldots, \phi_p]`` |
| `theta` | `Vector{T}` | MA coefficients ``[\theta_1, \ldots, \theta_q]`` |
| `sigma2` | `T` | Innovation variance |
| `residuals` | `Vector{T}` | Estimated residuals |
| `fitted` | `Vector{T}` | Fitted values |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `method` | `Symbol` | Estimation method |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of iterations |

---

## The ARIMA(p,d,q) Model

The **ARIMA(p,d,q) model** extends ARMA to non-stationary series by applying ``d``-fold differencing before fitting an ARMA(p,q). Many macroeconomic variables --- real GDP, industrial production, price levels --- exhibit unit roots and require differencing to achieve stationarity (Nelson & Plosser 1982).

```math
\phi(L) \, (1-L)^d \, y_t = c + \theta(L) \, \varepsilon_t
```

where:
- ``(1-L)^d y_t`` is the ``d``-th difference of ``y_t``
- ``\phi(L)`` and ``\theta(L)`` are the AR and MA lag polynomials applied to the differenced series
- ``d`` is the integration order

Common cases:
- ``d = 1``: ``\Delta y_t = y_t - y_{t-1}`` (first difference, for I(1) series)
- ``d = 2``: ``\Delta^2 y_t`` (second difference, for I(2) series)

The implementation differences the series ``d`` times, estimates ARMA(p,q) on the differenced series using the unified estimation pipeline, and stores both the original and differenced data.

```julia
using MacroEconometricModels

# Log industrial production — an I(1) series
fred = load_example(:fred_md)
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))

model = estimate_arima(y_level, 1, 1, 0)
report(model)
```

The ARIMA(1,1,0) on log IP first-differences the level series (removing the stochastic trend), then fits an AR(1) to the growth rate. The AR coefficient on the differenced series captures month-to-month momentum in industrial production growth.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:css_mle` | Estimation method (`:css`, `:mle`, or `:css_mle`) |
| `include_intercept` | `Bool` | `true` | Include constant on differenced series |
| `max_iter` | `Int` | `500` | Maximum optimization iterations |

### ARIMAModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original (undifferenced) time series |
| `y_diff` | `Vector{T}` | ``d``-fold differenced series |
| `p` | `Int` | AR order |
| `d` | `Int` | Integration order |
| `q` | `Int` | MA order |
| `c` | `T` | Intercept (on differenced series) |
| `phi` | `Vector{T}` | AR coefficients |
| `theta` | `Vector{T}` | MA coefficients |
| `sigma2` | `T` | Innovation variance |
| `residuals` | `Vector{T}` | Estimated residuals |
| `fitted` | `Vector{T}` | Fitted values (on differenced scale) |
| `loglik` | `T` | Log-likelihood |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `method` | `Symbol` | Estimation method |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of iterations |

---

## [Exact MLE via Kalman Filter](@id kalman_mle)

For exact maximum likelihood estimation, the ARMA(p,q) model is cast into the state-space form of Harvey (1993). This avoids the conditioning bias of CSS and provides asymptotically efficient estimates with correctly computed standard errors.

### State-Space Representation

```math
y_t = c + Z \, \alpha_t
```

```math
\alpha_{t+1} = T \, \alpha_t + R \, \eta_t, \quad \eta_t \sim N(0, Q)
```

where:
- ``\alpha_t = [a_t, a_{t-1}, \ldots, a_{t-r+1}]'`` is the ``r \times 1`` state vector with ``r = \max(p, q+1)``
- ``Z = [1, \theta_1, \ldots, \theta_{r-1}]`` is the ``1 \times r`` observation vector
- ``T`` is the ``r \times r`` companion matrix with AR coefficients in the first row
- ``R = [1, 0, \ldots, 0]'`` is the ``r \times 1`` selection vector
- ``Q = [\sigma^2]`` is the scalar innovation variance

### Prediction Error Decomposition

The Kalman filter computes the exact log-likelihood via the prediction error decomposition (Durbin & Koopman 2012):

```math
\ell(\Theta) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \sum_{t=1}^{n} \left( \log f_t + \frac{v_t^2}{f_t} \right)
```

where:
- ``v_t = y_t - \hat{y}_{t|t-1}`` is the one-step prediction error
- ``f_t = Z P_{t|t-1} Z' + H`` is the prediction error variance
- ``n`` is the number of observations
- ``\Theta = (\phi_1, \ldots, \phi_p, \theta_1, \ldots, \theta_q, \sigma^2)`` is the full parameter vector

!!! note "Technical Note"
    Initialization uses the unconditional (stationary) distribution ``P_0 = \text{dlyap}(T, RQR')`` when the system is stable. For non-stationary parameters the filter falls back to diffuse initialization (``P_0 = 10^6 I``). The variance parameter ``\sigma^2`` is optimized on the log scale for unconstrained optimization via L-BFGS.

---

## Forecasting

The `forecast` function computes optimal multi-step-ahead predictions with confidence intervals for all ARIMA-class models. Forecast uncertainty grows with the horizon, reflecting the accumulation of future unknown shocks.

### Point Forecasts

The optimal ``h``-step ahead forecast minimizes mean squared error. For an ARMA(p,q) process, forecasts are computed recursively (Hamilton 1994, Section 4.2):

```math
\hat{y}_{T+h|T} = c + \sum_{i=1}^{p} \phi_i \hat{y}_{T+h-i|T} + \sum_{j=1}^{q} \theta_j \hat{\varepsilon}_{T+h-j}
```

where:
- ``\hat{y}_{T+k|T} = y_{T+k}`` for ``k \leq 0`` (known past values)
- ``\hat{\varepsilon}_{T+k} = 0`` for ``k \geq 1`` (future residuals set to their expectation)
- ``\hat{\varepsilon}_{T+k} = \varepsilon_{T+k}`` for ``k \leq 0`` (estimated past residuals)

### Forecast Uncertainty

Forecast standard errors derive from the MA(``\infty``) representation. The ``\psi``-weights satisfy the recursion:

```math
\psi_j = \sum_{i=1}^{\min(p,j)} \phi_i \, \psi_{j-i} + \theta_j \, \mathbb{1}(j \leq q), \quad \psi_0 = 1
```

where:
- ``\psi_j`` is the ``j``-th coefficient in ``y_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}``
- ``\phi_i`` are the AR coefficients (zero for ``i > p``)
- ``\theta_j`` are the MA coefficients (zero for ``j > q``)

The ``h``-step ahead forecast variance is:

```math
\text{Var}(e_{T+h|T}) = \sigma^2 \left(1 + \psi_1^2 + \psi_2^2 + \cdots + \psi_{h-1}^2 \right)
```

where:
- ``e_{T+h|T} = y_{T+h} - \hat{y}_{T+h|T}`` is the forecast error
- ``\sigma^2`` is the innovation variance

Confidence intervals are symmetric Gaussian: ``\hat{y}_{T+h|T} \pm z_{\alpha/2} \cdot \text{se}_h``.

### ARIMA Forecasting

For ARIMA(p,d,q) models, forecasts are computed on the differenced series and integrated back to the original scale. For ``d = 1``:

```math
\hat{y}_{T+h} = y_T + \sum_{j=1}^{h} \widehat{\Delta y}_{T+j|T}
```

where:
- ``\hat{y}_{T+h}`` is the level forecast
- ``\widehat{\Delta y}_{T+j|T}`` is the forecast of the differenced series
- ``y_T`` is the last observed level

Standard errors are adjusted for the integration via cumulative variance accumulation.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

arma = estimate_arma(y, 1, 1)
fc = forecast(arma, 12; conf_level=0.95)
report(fc)

# Visualize with recent history
p = plot_result(fc; history=y, n_history=30)
```

```@raw html
<iframe src="../assets/plots/forecast_arima.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The forecast fan widens with the horizon as cumulative ``\psi``-weight variance grows. For the ARMA(1,1) model, the one-step-ahead standard error equals ``\sigma`` (the innovation standard deviation), while longer-horizon forecasts converge toward the unconditional mean with uncertainty approaching ``\sigma / \sqrt{1 - \phi_1^2}``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `conf_level` | `Real` | `0.95` | Confidence level for interval construction |

### ARIMAForecast Return Values

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Vector{T}` | Point forecasts ``\hat{y}_{T+1}, \ldots, \hat{y}_{T+h}`` |
| `ci_lower` | `Vector{T}` | Lower confidence bound |
| `ci_upper` | `Vector{T}` | Upper confidence bound |
| `se` | `Vector{T}` | Forecast standard errors (from ``\psi``-weights) |
| `horizon` | `Int` | Forecast horizon ``h`` |
| `conf_level` | `T` | Confidence level (e.g., 0.95) |

---

## Order Selection

Choosing the AR and MA orders is a central step in the Box-Jenkins methodology. The package provides both manual grid search and fully automatic selection, using the Akaike Information Criterion (Akaike 1974) and the Bayesian Information Criterion (Schwarz 1978).

### Grid Search

`select_arima_order` evaluates all ARMA(p,q) combinations up to specified maxima and selects the best model by AIC or BIC:

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

# Search over p in {0,...,4}, q in {0,...,4}
sel = select_arima_order(y, 4, 4)
report(sel)
```

The BIC-optimal order typically selects a more parsimonious model than AIC because BIC penalizes free parameters more heavily (penalty ``k \log n`` vs. ``2k``). For forecasting applications, BIC-selected models often outperform AIC-selected models at longer horizons due to reduced parameter estimation uncertainty.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `criterion` | `Symbol` | `:bic` | Selection criterion (`:aic` or `:bic`) |
| `d` | `Int` | `0` | Integration order (0 = ARMA search) |
| `method` | `Symbol` | `:css_mle` | Estimation method for each candidate model |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |

### ARIMAOrderSelection Return Values

| Field | Type | Description |
|-------|------|-------------|
| `best_p_aic` | `Int` | Optimal AR order by AIC |
| `best_q_aic` | `Int` | Optimal MA order by AIC |
| `best_p_bic` | `Int` | Optimal AR order by BIC |
| `best_q_bic` | `Int` | Optimal MA order by BIC |
| `aic_matrix` | `Matrix{T}` | ``(p_{\max}+1) \times (q_{\max}+1)`` matrix of AIC values |
| `bic_matrix` | `Matrix{T}` | ``(p_{\max}+1) \times (q_{\max}+1)`` matrix of BIC values |
| `best_model_aic` | `AbstractARIMAModel` | Fitted model with best AIC |
| `best_model_bic` | `AbstractARIMAModel` | Fitted model with best BIC |

### Automatic Selection

`auto_arima` implements a fully automatic model selection procedure. It first determines the integration order ``d`` via a variance-reduction heuristic (differencing until variance stops decreasing), then performs a grid search over ``p`` and ``q``:

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))

best = auto_arima(y_level; max_p=5, max_q=5, max_d=2, criterion=:bic)
report(best)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_p` | `Int` | `5` | Maximum AR order to consider |
| `max_q` | `Int` | `5` | Maximum MA order to consider |
| `max_d` | `Int` | `2` | Maximum integration order to consider |
| `criterion` | `Symbol` | `:bic` | Selection criterion (`:aic` or `:bic`) |
| `method` | `Symbol` | `:css_mle` | Estimation method for each candidate |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |

---

## StatsAPI Interface

All ARIMA-class models implement the Julia `StatsAPI.RegressionModel` interface, providing interoperability with the broader Julia statistics ecosystem.

```julia
using MacroEconometricModels, StatsAPI

fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

model = estimate_arma(y, 1, 1)

# Standard accessors
coef(model)          # Coefficient vector [c, phi_1, theta_1]
nobs(model)          # Number of observations
dof(model)           # Degrees of freedom (number of parameters)
dof_residual(model)  # Residual degrees of freedom
loglikelihood(model) # Log-likelihood
aic(model)           # Akaike Information Criterion
bic(model)           # Bayesian Information Criterion
residuals(model)     # Residual vector
fitted(model)        # Fitted values
r2(model)            # R-squared

# fit interface
model = fit(ARModel, y, 2)           # AR(2)
model = fit(MAModel, y, 1)           # MA(1)
model = fit(ARMAModel, y, 1, 1)      # ARMA(1,1)

# Prediction
yhat = predict(model, 12)  # 12-step point forecasts
```

The `fit` interface provides a standard constructor pattern consistent with other Julia statistical packages. The `predict` method with an integer argument returns point forecasts (without confidence intervals); use `forecast` for the full `ARIMAForecast` object with standard errors and confidence bands.

---

## Complete Example

This example demonstrates the full Box-Jenkins workflow: unit root testing, order selection, estimation, diagnostics, and forecasting on FRED-MD industrial production data.

```julia
using MacroEconometricModels

# Industrial production growth (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

# Step 1: Check for unit root — IP growth should be stationary
adf_result = adf_test(y; lags=:aic, regression=:constant)
report(adf_result)

# Step 2: Select ARMA order via BIC grid search
sel = select_arima_order(y, 4, 4)
report(sel)

# Step 3: Estimate the BIC-optimal model
model = sel.best_model_bic
report(model)

# Step 4: Forecast IP growth 12 months ahead
fc = forecast(model, 12; conf_level=0.95)
report(fc)

# Step 5: Visualize forecast with recent history
p = plot_result(fc; history=y, n_history=50)
```

The ADF test rejects the unit root null at the 1% level, confirming that IP growth is stationary and no differencing is required. The BIC grid search identifies the optimal ARMA order, balancing fit against parsimony. The 12-month forecast shows IP growth reverting toward its unconditional mean, with widening confidence bands that reflect increasing uncertainty at longer horizons. The one-step standard error provides the minimal forecast uncertainty, while the 12-step band is substantially wider due to the accumulation of ``\psi``-weight variance.

---

## Common Pitfalls

1. **Fitting ARMA to a non-stationary series**: Estimating ARMA(p,q) on an I(1) level series produces spurious coefficient estimates and unreliable forecasts. Always test for unit roots with `adf_test` or `kpss_test` before estimation, and use `estimate_arima` with ``d \geq 1`` for integrated processes.

2. **Over-differencing**: Applying ``d = 2`` to an I(1) series introduces an artificial MA unit root, inflating MA coefficient estimates toward ``-1`` and degrading forecast accuracy. Let `auto_arima` choose ``d`` via variance reduction, or determine ``d`` from unit root tests applied sequentially.

3. **CSS vs. MLE convergence**: CSS conditions on initial residuals being zero, which biases estimates in small samples (``n < 100``). MLE via Kalman filter is exact but can converge to local optima when started from poor initial values. The default `:css_mle` mitigates both problems --- use it unless there is a specific reason to prefer one method.

4. **`auto_arima` criteria selection**: AIC tends to select larger models that fit in-sample noise, while BIC selects more parsimonious models that often forecast better out of sample. For forecasting applications, prefer `criterion=:bic`. For structural analysis where capturing all dynamics matters, consider `criterion=:aic`.

5. **ARMA order identifiability**: An ARMA(p,q) model with common roots in ``\phi(z)`` and ``\theta(z)`` is not identified --- the common factor cancels. If `select_arima_order` returns similar IC values for ARMA(1,1) and AR(1), the MA component may not be contributing meaningfully. Inspect coefficient significance via `report()` before choosing the larger model.

6. **Forecast integration for ARIMA**: Forecasts from `forecast(::ARIMAModel, h)` are automatically integrated back to the original level scale. The returned `forecast` field contains level forecasts, not differenced forecasts. Standard errors account for the cumulative variance from integration.

---

## References

- Akaike, H. (1974). A New Look at the Statistical Model Identification.
  *IEEE Transactions on Automatic Control*, 19(6), 716-723. [DOI](https://doi.org/10.1109/TAC.1974.1100705)

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*.
  San Francisco: Holden-Day. ISBN 978-0-816-21104-3.

- Brockwell, P. J., & Davis, R. A. (1991). *Time Series: Theory and Methods*. 2nd ed.
  New York: Springer. ISBN 978-1-4419-0319-8.

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. 2nd ed.
  Oxford: Oxford University Press. [DOI](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)

- Hamilton, J. D. (1994). *Time Series Analysis*.
  Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Harvey, A. C. (1993). *Time Series Models*. 2nd ed.
  Cambridge, MA: MIT Press. ISBN 978-0-262-08224-2.

- Nelson, C. R., & Plosser, C. I. (1982). Trends and Random Walks in Macroeconomic Time Series.
  *Journal of Monetary Economics*, 10(2), 139-162. [DOI](https://doi.org/10.1016/0304-3932(82)90012-5)

- Schwarz, G. (1978). Estimating the Dimension of a Model.
  *The Annals of Statistics*, 6(2), 461-464. [DOI](https://doi.org/10.1214/aos/1176344136)
