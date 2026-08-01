# [ARIMA Models](@id arima_page)

**MacroEconometricModels.jl** provides a complete suite for estimating, diagnosing, and forecasting with univariate ARIMA-class models. The implementation covers the full Box-Jenkins (1976) workflow from model identification through order selection and out-of-sample forecasting.

- **AR(p)**: Autoregressive models estimated via OLS or exact MLE
- **MA(q)**: Moving average models estimated via CSS, exact MLE, or CSS-MLE
- **ARMA(p,q)**: Combined autoregressive-moving average with three estimation methods
- **ARIMA(p,d,q)**: Integrated ARMA for non-stationary series via ``d``-fold differencing
- **SARIMA(p,d,q)(P,D,Q)ₛ**: Multiplicative seasonal ARIMA with seasonal differencing and automatic order search
- **Forecasting**: Multi-step point forecasts with ``\psi``-weight confidence intervals
- **Order Selection**: Grid search over information criteria and automatic `auto_arima`
- **StatsAPI Interface**: Full `coef`, `nobs`, `predict`, `fit`, `residuals`, `aic`, `bic` compatibility

```@setup arima
using MacroEconometricModels
fred = load_example(:fred_md)
# One log difference of the CPI level = monthly inflation. Deliberately NOT
# apply_tcode, whose tcode 6 for CPIAUCSL is the SECOND log difference.
cpi_raw = fred[:, "CPIAUCSL"]
y = filter(isfinite, diff(log.(cpi_raw)))
y = y[end-99:end]

# FRED-MD ships seasonally adjusted series, so the seasonal examples use a simulated
# airline process (0,1,1)(0,1,1)₁₂ with known θ = -0.4, Θ = -0.6.
using Random
let rng = Random.MersenneTwister(7), n = 240, s = 12, th = -0.4, TH = -0.6
    resid = randn(rng, n + 50)
    global y_seasonal = zeros(n + 50)
    for t in (s+2):(n+50)
        w = resid[t] + th*resid[t-1] + TH*resid[t-s] + th*TH*resid[t-s-1]
        y_seasonal[t] = w + y_seasonal[t-1] + y_seasonal[t-s] - y_seasonal[t-s-1]
    end
    global y_seasonal = y_seasonal[51:end]
end
```

## Quick Start

**Recipe 1: Estimate an AR(2) on CPI inflation**

```@example arima
ar = estimate_ar(y, 2)
report(ar)
```

**Recipe 2: Fit an ARMA(1,1) and forecast 12 months ahead**

```@example arima
arma = estimate_arma(y, 1, 1)
fc = forecast(arma, 12; conf_level=0.95)
report(fc)
```

**Recipe 3: ARIMA(1,1,0) on a non-stationary level series**

```@example arima
y_level = cumsum(y)  # synthetic I(1) series
arima = estimate_arima(y_level, 1, 1, 0)
report(arima)
```

**Recipe 4: Automatic order selection via grid search**

```@example arima
sel = select_arima_order(y, 4, 4)
report(sel)
```

**Recipe 5: Fully automatic model selection with `auto_arima`**

```@example arima
best = auto_arima(y_level; max_p=5, max_q=5, max_d=2, criterion=:bic)
report(best)
```

**Recipe 6: Forecast and visualize**

```@example arima
ar = estimate_ar(y, 2)
fc = forecast(ar, 20)
```

```julia
p = plot_result(fc; history=y, n_history=30)
```

```@raw html
<iframe src="../assets/plots/forecast_ar.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
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

```@example arima
# OLS estimation (default)
ar_ols = estimate_ar(y, 2)
report(ar_ols)
```

```@example arima
# MLE estimation
ar_mle = estimate_ar(y, 2; method=:mle)
report(ar_mle)
```

The AR(2) on monthly CPI inflation captures short-run momentum (``\hat\phi_1 = 0.581`` by OLS) and mild mean reversion (``\hat\phi_2 = -0.079``); MLE returns ``0.573`` and ``-0.079``, so the two estimators agree to the second decimal. The information criteria do not: OLS reports AIC ``-898.5`` against MLE's ``-916.9``, because the conditional OLS likelihood is evaluated over the ``n - p`` usable rows while the Kalman likelihood uses all ``n``. Compare AIC or BIC across models only when they were computed under the same estimation method.

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

```@example arima
ma = estimate_ma(y, 1; method=:css_mle)
report(ma)
```

The estimated MA(1) coefficient ``\hat\theta_1 = 0.599`` captures one-period serial correlation in shocks to CPI inflation. Its positive sign means a positive inflation surprise this month raises next month's forecast above the unconditional mean ``\hat c = 0.0029`` (0.29% per month). With a BIC of ``-911.6`` this two-parameter model is the most parsimonious fit on the page — the grid search below selects exactly this specification.

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

```@example arima
arma = estimate_arma(y, 1, 1; method=:css_mle)
report(arma)
```

The ARMA(1,1) splits the persistence of CPI inflation between an autoregressive root (``\hat\phi_1 = 0.226``) and a one-period moving-average term (``\hat\theta_1 = 0.409``). Its BIC of ``-908.0`` beats the MLE-estimated AR(2) (``-906.5``) by a hair, and both lose to the plain MA(1) at ``-911.6``: the AR and MA roots here are close enough that the pair is nearly a common factor, so the extra parameter buys almost no fit. This near-cancellation is the identification problem in Pitfall 5, visible in a single comparison.

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

```@example arima
model = estimate_arima(y_level, 1, 1, 0)
report(model)
```

The ARIMA(1,1,0) runs on `y_level = cumsum(y)`, the accumulated inflation series — an I(1) process that is, up to a constant, the log price level. The estimator first-differences it back to inflation and fits an AR(1), giving ``\hat\phi_1 = 0.534``: month-to-month momentum, and close to the AR(2)'s ``\hat\phi_1 + \hat\phi_2 = 0.502`` total persistence. The reported `fitted` and `residuals` live on the differenced scale, while `forecast` returns levels.

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

## The SARIMA(p,d,q)(P,D,Q)ₛ Model

Monthly and quarterly macro series carry seasonal dependence that a non-seasonal ARIMA cannot represent parsimoniously: a December effect in monthly data would need ``\phi_{12}`` and every lag below it. The **multiplicative seasonal ARIMA** adds a second pair of polynomials in ``L^s``:

```math
\Phi_P(L^s)\,\phi_p(L)\,(1-L)^d (1-L^s)^D y_t = \Theta_Q(L^s)\,\theta_q(L)\,\varepsilon_t
```

where:
- ``\phi_p(L) = 1 - \phi_1 L - \cdots - \phi_p L^p`` is the non-seasonal AR polynomial
- ``\Phi_P(L^s) = 1 - \Phi_1 L^s - \cdots - \Phi_P L^{Ps}`` is the seasonal AR polynomial
- ``\theta_q(L)``, ``\Theta_Q(L^s)`` are the corresponding MA polynomials
- ``s`` is the seasonal period (12 monthly, 4 quarterly)
- ``d``, ``D`` are the regular and seasonal differencing orders

!!! note "Technical Note"
    The multiplicative structure is handled by **expanding** the two polynomial pairs into a single long ARMA — ``\phi_p(L)\Phi_P(L^s)`` has degree ``p + Ps`` — which the existing Kalman likelihood then evaluates unchanged. The expansion is what makes the model parsimonious: SARIMA(0,1,1)(0,1,1)₁₂ has two parameters but an MA of degree 13, with the lag-13 coefficient pinned to the product ``\theta_1\Theta_1`` rather than estimated freely.

### Estimation

`estimate_sarima` takes both order triples and the seasonal period, and offers the same three methods as `estimate_arima`: `:css`, `:mle`, and `:css_mle` (the default). The canonical case is Box and Jenkins' **airline model**, SARIMA(0,1,1)(0,1,1)₁₂:

```@example arima
airline = estimate_sarima(y_seasonal, 0, 1, 1, 0, 1, 1, 12; include_intercept=false)
report(airline)
```

The two estimated coefficients sit within one standard error of the data-generating values ``\theta_1 = -0.4`` and ``\Theta_1 = -0.6``. The expanded MA polynomial shows the multiplicative restriction directly — a coefficient at lag 1, one at lag 12, and their product at lag 13, with zeros in between:

```@example arima
round.(airline.theta_expanded[[1, 12, 13]], digits=4)
```

With `P = D = Q = 0` the model reduces exactly to `estimate_arima(y, p, d, q)` — identical coefficients, likelihood, and forecasts.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:css_mle` | `:css`, `:mle`, or `:css_mle` |
| `include_intercept` | `Bool` | `true` | Constant on the differenced series |
| `max_iter` | `Int` | `500` | Optimizer iteration cap |

### SARIMAModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Original (undifferenced) series |
| `y_diff` | `Vector{T}` | Series after ``(1-L)^d(1-L^s)^D`` |
| `p`, `d`, `q` | `Int` | Non-seasonal orders |
| `P`, `D`, `Q` | `Int` | Seasonal orders |
| `s` | `Int` | Seasonal period |
| `c` | `T` | Intercept (on the differenced series) |
| `phi`, `theta` | `Vector{T}` | Non-seasonal AR and MA coefficients |
| `Phi`, `Theta` | `Vector{T}` | Seasonal AR and MA coefficients |
| `phi_expanded` | `Vector{T}` | AR coefficients of ``\phi_p(L)\Phi_P(L^s)``, length ``p + Ps`` |
| `theta_expanded` | `Vector{T}` | MA coefficients of ``\theta_q(L)\Theta_Q(L^s)``, length ``q + Qs`` |
| `sigma2` | `T` | Innovation variance |
| `residuals`, `fitted` | `Vector{T}` | Residuals and fitted values (differenced scale) |
| `loglik`, `aic`, `bic` | `T` | Log-likelihood and information criteria |
| `method` | `Symbol` | Estimation method |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of iterations |

### Seasonal Forecasting

`forecast` projects the expanded ARMA on the differenced scale, then un-differences through the full operator ``(1-L)^d(1-L^s)^D``. Prediction intervals come from the ``\psi``-weights of the **non-differenced** operator ``\phi(L)\Phi(L^s)(1-L)^d(1-L^s)^D``, so the bands widen at the rate the doubly integrated process implies rather than at the stationary ARMA rate:

```@example arima
fc = forecast(airline, 24)
report(fc)
```

The point forecasts reproduce the seasonal profile of the last observed year, and the standard errors grow monotonically with the horizon.

```julia
plot_result(fc; history=y_seasonal)
```

### Automatic Seasonal Order Selection

`auto_sarima` searches the ARMA orders at a given seasonal period. Differencing orders are chosen first and held fixed — information criteria are not comparable across differencing orders — with ``D`` selected by the [HEGY seasonal unit-root test](@ref tests_unitroot_advanced_page) and ``d`` by the KPSS test on the seasonally differenced series. The order search itself runs with `:css` for speed and the winner is refit with `method`, following Hyndman & Khandakar (2008).

```@example arima
best = auto_sarima(y_seasonal, 12; max_p=1, max_q=1, max_P=1, max_Q=1)
(p=best.p, d=best.d, q=best.q, P=best.P, D=best.D, Q=best.Q)
```

The search recovers the airline orders (0,1,1)(0,1,1)₁₂.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `d`, `D` | `Int` | `nothing` | Fix the differencing orders instead of testing for them |
| `max_p`, `max_q` | `Int` | `2` | Non-seasonal search bounds |
| `max_P`, `max_Q` | `Int` | `1` | Seasonal search bounds |
| `criterion` | `Symbol` | `:aic` | `:aic` or `:bic` |
| `method` | `Symbol` | `:css_mle` | Estimation method for the final refit |
| `include_intercept` | `Bool` | `true` | Constant on the differenced series |

!!! note "Seasonal period support in `D` selection"
    The HEGY test is tabulated for quarterly and monthly data only, so automatic ``D`` selection applies when `s ∈ {4, 12}` and the sample has at least ``3s + 5`` observations. At any other period `D` defaults to `0` and should be supplied explicitly.

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

```@example arima
arma = estimate_arma(y, 1, 1)
fc = forecast(arma, 12; conf_level=0.95)
report(fc)
```

```julia
# Visualize with recent history
p = plot_result(fc; history=y, n_history=30)
```

```@raw html
<iframe src="../assets/plots/forecast_arma.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The forecast fan widens with the horizon as cumulative ``\psi``-weight variance grows. For this ARMA(1,1) the one-step standard error is 0.00235, exactly ``\hat\sigma``, since ``\psi_0 = 1`` and no future shocks have accumulated. By ``h = 12`` it has risen to 0.00281 and is within a rounding of its limit, the unconditional standard deviation ``\hat\sigma\sqrt{(1 + 2\hat\phi_1\hat\theta_1 + \hat\theta_1^2)/(1 - \hat\phi_1^2)} = 0.00281``. Point forecasts converge to ``\hat c/(1-\hat\phi_1)`` over the same horizon, so the band is effectively at its stationary width after a year.

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

```@example arima
# Search over p in {0,...,4}, q in {0,...,4}
sel = select_arima_order(y, 4, 4)
report(sel)
```

The AIC selects an ARMA(1,2) at ``-922.4`` while the BIC selects the more parsimonious ARMA(0,1) at ``-911.6``, because BIC penalizes free parameters more heavily (``k \log n`` against ``2k``, and ``\log 100 = 4.6``). The disagreement is the usual one over two extra parameters that improve in-sample fit by less than the BIC charges for them. For forecasting, BIC-selected models often win at longer horizons because they carry less parameter-estimation uncertainty.

!!! note "CSS order comparability"
    Under `:css` estimation the conditional likelihood is evaluated over ``n - \max(p,q)`` observations, a window that varies with the candidate order. `select_arima_order` therefore rescores every `:css` candidate's AIC/BIC on a **common conditioning window** ``n - \max(\text{max\_p}, \text{max\_q})`` so the criteria are comparable across orders; MLE / CSS-MLE candidates already use the full sample.

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

`auto_arima` implements a fully automatic model selection procedure. It first determines the integration order ``d`` by iterating an **Augmented Dickey-Fuller unit-root test** --- differencing while ADF fails to reject a unit root (``p > 0.05``), stopping at ``max_d`` or as soon as a differenced series falls below 20 observations, and falling back to a variance-reduction rule only when the ADF test itself errors on a degenerate series. It then searches ``p`` and ``q`` by the **Hyndman-Khandakar (2008) stepwise** procedure: fit the seeds (0,0), (1,0), (0,1), (2,2), then walk to neighbours differing by ``\pm 1`` in ``p`` or ``q`` until no move improves the criterion. Pass `stepwise=false` for an exhaustive grid instead.

```@example arima
best = auto_arima(y_level; max_p=5, max_q=5, max_d=2, criterion=:bic)
report(best)
```

The search differences once and settles on ARIMA(0,1,1) with BIC ``-902.0`` --- the same MA(1) structure the grid search found on the already-differenced series, which is the consistency check worth running whenever `auto_arima` is handed a level series.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_p` | `Int` | `5` | Maximum AR order to consider |
| `max_q` | `Int` | `5` | Maximum MA order to consider |
| `max_d` | `Int` | `2` | Maximum integration order to consider |
| `criterion` | `Symbol` | `:bic` | Selection criterion (`:aic` or `:bic`) |
| `method` | `Symbol` | `:css_mle` | Estimation method for each candidate |
| `include_intercept` | `Bool` | `true` | Whether to include constant term |
| `stepwise` | `Bool` | `true` | Hyndman-Khandakar stepwise search; `false` for an exhaustive ``(p,q)`` grid |

---

## StatsAPI Interface

All ARIMA-class models implement the Julia `StatsAPI.RegressionModel` interface, providing interoperability with the broader Julia statistics ecosystem.

```@example arima
model = estimate_arma(y, 1, 1)

(coef        = round.(coef(model), digits=4),   # [c, φ₁, θ₁]
 nobs        = nobs(model),
 dof         = dof(model),                      # estimated parameters
 dof_residual = dof_residual(model),
 loglik      = round(loglikelihood(model), digits=2),
 aic         = round(aic(model), digits=2),
 bic         = round(bic(model), digits=2),
 r2          = round(r2(model), digits=4))
```

`residuals(model)` and `fitted(model)` return the corresponding vectors. The `fit` constructor pattern matches the rest of the Julia statistics ecosystem, and `predict` with an integer horizon returns point forecasts only:

```@example arima
m_ar = fit(ARModel, y, 2)          # equivalently estimate_ar(y, 2)
m_ma = fit(MAModel, y, 1)
m_arma = fit(ARMAModel, y, 1, 1)
round.(predict(m_arma, 12), digits=5)
```

Use `forecast` instead of `predict` when standard errors and confidence bands are needed --- it returns the full `ARIMAForecast` object documented above.

---

## Complete Example

This example demonstrates the full Box-Jenkins workflow: unit root testing, order selection, estimation, diagnostics, and forecasting on FRED-MD CPI (CPIAUCSL) inflation data.

```@example arima
# Step 1: Check for unit root — CPI inflation should be stationary
adf_result = adf_test(y; lags=:aic, regression=:constant)
report(adf_result)
```

```@example arima
# Step 2: Select ARMA order via BIC grid search
sel = select_arima_order(y, 4, 4)
report(sel)
```

```@example arima
# Step 3: Estimate the BIC-optimal model
model = sel.best_model_bic
report(model)
```

```@example arima
# Step 4: Forecast CPI inflation 12 months ahead
fc = forecast(model, 12; conf_level=0.95)
report(fc)
```

```julia
# Step 5: Visualize forecast with recent history
p = plot_result(fc; history=y, n_history=50)
```

```@raw html
<iframe src="../assets/plots/forecast_arima.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The ADF statistic of ``-5.48`` clears the 1% critical value of ``-3.49``, so CPI inflation is stationary and no differencing is required --- consistent with `auto_arima` choosing ``d = 1`` for the *level* series above. The BIC grid search then picks ARMA(0,1), the plain MA(1). Its 12-month forecast starts at 0.334% for the month ahead, where the MA term still carries information, and settles on the unconditional mean 0.287% from ``h = 2`` onward, because an MA(1) has no memory beyond one period. The standard error rises only from 0.236% to 0.275% across the same horizons: for a short-memory model almost all of the forecast uncertainty is present at the first step.

---

## ARFIMA and Long Memory

Standard ARIMA differencing is restricted to integer orders: a series is either I(0) or I(1). Many economic series --- inflation, realized volatility, interest-rate spreads --- exhibit *long memory*: an autocorrelation function that decays hyperbolically (``\rho(k) \sim k^{2d-1}``) rather than geometrically, too slowly for a stationary ARMA yet without a unit root. The **ARFIMA(p, d, q)** model of Granger, Joyeux, and Hosking captures this with a *fractional* integration order ``d \in (-0.5, 0.5)``:

```math
\phi(L)\,(1-L)^d\,(y_t - \mu) = \theta(L)\,\varepsilon_t,
```

where the fractional-difference operator is defined by the binomial expansion

```math
(1-L)^d = \sum_{k=0}^{\infty} \pi_k L^k, \qquad
\pi_0 = 1,\quad \pi_k = \pi_{k-1}\,\frac{k-1-d}{k}.
```

For ``0 < d < 0.5`` the process is stationary but long-range dependent; for ``-0.5 < d < 0`` it is *anti-persistent* (intermediate memory). The weights ``\pi_k`` decay only as ``k^{-1-d}``, so the filter has effectively infinite memory.

### Estimating the fractional order

`estimate_arfima(y, p, q; method)` jointly estimates ``d`` and the ARMA parameters. Two methods are available:

- `:css` (default) --- conditional sum of squares. The series is fractionally differenced and the ARMA conditional likelihood is maximized. Fast, and adequate when the ARMA part is the object of interest.
- `:mle` --- exact Gaussian ML via the Durbin--Levinson recursion over the Sowell (1992) / Hosking (1981) ARFIMA autocovariances. ``O(T^2)``, and the accurate choice for ``d`` itself.

The order ``d`` is kept strictly inside ``(-0.5, 0.5)`` through an internal logit reparameterization, and its standard error is reported via the delta method.

```@example arima
nile = load_example(:nile)         # annual Nile flow at Aswan, 1871–1970
flow = to_vector(nile)

m = estimate_arfima(flow, 0, 0; method=:mle)
report(m)
```

Exact ML puts the fractional order at ``\hat d = 0.364`` (SE 0.069), squarely in the stationary long-memory region ``0 < d < 0.5`` and comfortably away from both the ``d = 0`` short-memory null and the ``d = 0.5`` nonstationary edge. The two semiparametric estimators below corroborate it. The `:css` default disagrees sharply on this series --- it returns ``\hat d = -0.081``, the anti-persistent sign --- so when ``d`` itself is the object of interest, fit with `:mle` and treat a CSS estimate as a starting value rather than a result.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:css` | `:css` (conditional sum of squares) or `:mle` (exact Gaussian ML) |
| `d0` | `Real` | `nothing` | Starting value for ``d``; derived from the data when omitted |
| `trunc` | `Int` | `200` | Truncation lag of the fractional-difference filter |
| `max_iter` | `Int` | `500` | Optimizer iteration cap |

### ARFIMAModel Return Values

| Field | Type | Description |
|-------|------|-------------|
| `d` | `T` | Estimated fractional integration order |
| `d_se` | `T` | Delta-method standard error of ``\hat d`` |
| `c` | `T` | Estimated mean ``\hat\mu`` |
| `phi`, `theta` | `Vector{T}` | ARMA coefficients of the fractionally differenced series |
| `sigma2` | `T` | Innovation variance |
| `residuals`, `fitted` | `Vector{T}` | Residuals and fitted values |
| `loglik`, `aic`, `bic` | `T` | Log-likelihood and information criteria |
| `method` | `Symbol` | `:css` or `:mle` |
| `converged` | `Bool` | Convergence indicator |

### Semiparametric estimators

Two log-periodogram estimators estimate ``d`` *without* specifying the short-memory ARMA structure, and test ``H_0: d = 0`` (no long memory):

- `gph_test(y; m, trim)` --- the Geweke--Porter-Hudak (1983) regression of ``\log I(\lambda_j)`` on ``-\log\!\big(4\sin^2(\lambda_j/2)\big)`` over the first ``m`` Fourier frequencies (default ``m = \lfloor\sqrt{T}\rfloor``). The estimate ``\hat d`` is the negated slope.
- `local_whittle(y; m)` --- the Robinson (1995) Gaussian semiparametric estimator, minimizing the local Whittle objective over the same frequency band.

```@example arima
g = gph_test(flow)
g
```

```@example arima
lw = local_whittle(flow)
lw
```

Both point estimates land near the classic value for this benchmark series: ``\hat d_{\text{GPH}} = 0.390`` and ``\hat d_{\text{LW}} = 0.464``, against ``0.364`` from exact ML. Only local Whittle rejects ``H_0: d = 0`` (``p = 0.003``); GPH does not (``p = 0.184``), because with ``T = 100`` the default bandwidth is ``m = \lfloor\sqrt{100}\rfloor = 10`` frequencies and the log-periodogram regression has a standard error of 0.294 on ten points. Widen ``m`` to trade that variance against the bias from letting short-memory dynamics leak into the band.

### Forecasting

`forecast(::ARFIMAModel, h)` propagates the point forecast through the truncated ``\mathrm{AR}(\infty)`` representation ``\pi(L) = \phi(L)(1-L)^d/\theta(L)``, and accumulates forecast-error variance from the ``\mathrm{MA}(\infty)`` ``\psi``-weights.

```@example arima
fc = forecast(m, 10)
report(fc)
```

The forecasts climb from 811.7 back toward the estimated mean ``\hat\mu = 929.9`` but are still 56 units short of it after ten years --- hyperbolic memory pulls the level home far more slowly than the geometric decay of an ARMA. The standard errors rise from 140.5 to 162.1 and then flatten: for ``0 < d < 0.5`` the ``\psi``-weights are square-summable, so the forecast variance approaches a finite limit instead of diverging as it would under a unit root.

The shared filter helpers `MacroEconometricModels._frac_diff_weights(d, K)` and `_frac_diff(y, d)` (with an ``O(T\log T)`` FFT path, Jensen--Nielsen 2014) are also used by the FIGARCH long-memory volatility model.

---

## Common Pitfalls

1. **Fitting ARMA to a non-stationary series**: Estimating ARMA(p,q) on an I(1) level series produces spurious coefficient estimates and unreliable forecasts. Always test for unit roots with `adf_test` or `kpss_test` before estimation, and use `estimate_arima` with ``d \geq 1`` for integrated processes.

2. **Over-differencing**: Applying ``d = 2`` to an I(1) series introduces an artificial MA unit root, inflating MA coefficient estimates toward ``-1`` and degrading forecast accuracy. Let `auto_arima` choose ``d`` via variance reduction, or determine ``d`` from unit root tests applied sequentially.

3. **CSS vs. MLE convergence**: CSS conditions on initial residuals being zero, which biases estimates in small samples (``n < 100``). MLE via Kalman filter is exact but can converge to local optima when started from poor initial values. The default `:css_mle` mitigates both problems --- use it unless there is a specific reason to prefer one method.

4. **`auto_arima` criteria selection**: AIC tends to select larger models that fit in-sample noise, while BIC selects more parsimonious models that often forecast better out of sample. For forecasting applications, prefer `criterion=:bic`. For structural analysis where capturing all dynamics matters, consider `criterion=:aic`.

5. **ARMA order identifiability**: An ARMA(p,q) model with common roots in ``\phi(z)`` and ``\theta(z)`` is not identified --- the common factor cancels. If `select_arima_order` returns similar IC values for ARMA(1,1) and AR(1), the MA component may not be contributing meaningfully. Inspect coefficient significance via `report()` before choosing the larger model.

6. **Forecast integration for ARIMA**: Forecasts from `forecast(::ARIMAModel, h)` are automatically integrated back to the original level scale. The returned `forecast` field contains level forecasts, not differenced forecasts. Standard errors account for the cumulative variance from integration.

7. **ARFIMA ``d``/AR identification**: In an ARFIMA(p, d, q) model, the fractional order ``d`` and the AR persistence both govern long-run behavior and are only weakly separable in moderate samples. Jointly estimated ``\hat d`` and ``\hat\phi`` can trade off substantially at ``T`` of a few hundred; use longer samples, or the semiparametric `gph_test` / `local_whittle` estimators (which do not require specifying the ARMA part) to corroborate ``\hat d``.

8. **ARFIMA `:css` and `:mle` can disagree on ``d``**: The two estimators optimize different objectives, and at ``T = 100`` the gap is not academic --- on the Nile series `:css` returns ``\hat d = -0.081`` and `:mle` returns ``\hat d = 0.364``. Estimate ``d`` with `method=:mle` and use the semiparametric estimators as a cross-check; reserve `:css` for long samples or for cases where only the ARMA part matters.

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

- Geweke, J., & Porter-Hudak, S. (1983). The Estimation and Application of Long Memory Time Series Models.
  *Journal of Time Series Analysis*, 4(4), 221-238. [DOI](https://doi.org/10.1111/j.1467-9892.1983.tb00371.x)

- Granger, C. W. J., & Joyeux, R. (1980). An Introduction to Long-Memory Time Series Models and Fractional Differencing.
  *Journal of Time Series Analysis*, 1(1), 15-29. [DOI](https://doi.org/10.1111/j.1467-9892.1980.tb00297.x)

- Hyndman, R. J., & Khandakar, Y. (2008). Automatic Time Series Forecasting: The forecast Package for R.
  *Journal of Statistical Software*, 27(3), 1-22. [DOI](https://doi.org/10.18637/jss.v027.i03)

- Hosking, J. R. M. (1981). Fractional Differencing.
  *Biometrika*, 68(1), 165-176. [DOI](https://doi.org/10.1093/biomet/68.1.165)

- Jensen, A. N., & Nielsen, M. Ø. (2014). A Fast Fractional Difference Algorithm.
  *Journal of Time Series Analysis*, 35(5), 428-436. [DOI](https://doi.org/10.1111/jtsa.12074)

- Robinson, P. M. (1995). Gaussian Semiparametric Estimation of Long Range Dependence.
  *The Annals of Statistics*, 23(5), 1630-1661. [DOI](https://doi.org/10.1214/aos/1176324317)

- Sowell, F. (1992). Maximum Likelihood Estimation of Stationary Univariate Fractionally Integrated Time Series Models.
  *Journal of Econometrics*, 53(1-3), 165-188. [DOI](https://doi.org/10.1016/0304-4076(92)90084-5)

- Hamilton, J. D. (1994). *Time Series Analysis*.
  Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Harvey, A. C. (1993). *Time Series Models*. 2nd ed.
  Cambridge, MA: MIT Press. ISBN 978-0-262-08224-2.

- Nelson, C. R., & Plosser, C. I. (1982). Trends and Random Walks in Macroeconomic Time Series.
  *Journal of Monetary Economics*, 10(2), 139-162. [DOI](https://doi.org/10.1016/0304-3932(82)90012-5)

- Schwarz, G. (1978). Estimating the Dimension of a Model.
  *The Annals of Statistics*, 6(2), 461-464. [DOI](https://doi.org/10.1214/aos/1176344136)
