# Univariate Time Series (ARIMA)

This chapter covers univariate ARIMA-class models: AR, MA, ARMA, and ARIMA. These models are fundamental building blocks for time series analysis, forecasting, and as diagnostic tools for checking residual autocorrelation in multivariate models.

## Introduction

Univariate time series models capture temporal dependence through autoregressive (AR) and moving average (MA) components. The general ARIMA(p,d,q) model nests:

1. **AR(p)**: Autoregressive model — current value depends on past values
2. **MA(q)**: Moving average model — current value depends on past shocks
3. **ARMA(p,q)**: Combined autoregressive–moving average
4. **ARIMA(p,d,q)**: Integrated ARMA — models non-stationary series via differencing

**References**: Box & Jenkins (1976), Hamilton (1994, Chapters 3–5), Brockwell & Davis (1991)

## Quick Start

```julia
using MacroEconometricModels

# Industrial production growth (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))    # log first difference

ar = estimate_ar(y, 2)                                      # AR(2) via OLS
ma = estimate_ma(y, 1; method=:css_mle)                     # MA(1) via CSS-MLE
arma = estimate_arma(y, 1, 1)                                # ARMA(1,1)
fc = forecast(arma, 12; conf_level=0.95)                     # Forecast IP growth 12 months ahead
sel = select_arima_order(y, 4, 4)                            # Grid search for best (p,q)

# For I(1) level modeling
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))
arima = estimate_arima(y_level, 1, 1, 0)                    # ARIMA(1,1,0) on log IP
```

---

## The AR(p) Model

### Model Specification

An autoregressive model of order ``p`` is:

```math
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t
```

where ``\varepsilon_t \sim \text{WN}(0, \sigma^2)`` is white noise. Using the lag operator ``L``:

```math
\phi(L) y_t = c + \varepsilon_t, \quad \phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p
```

### Stationarity

The process is stationary if all roots of the characteristic polynomial ``\phi(z) = 0`` lie outside the unit circle, equivalently if all eigenvalues of the companion matrix

```math
F = \begin{bmatrix}
\phi_1 & \phi_2 & \cdots & \phi_{p-1} & \phi_p \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{bmatrix}_{p \times p}
```

have modulus less than 1: ``|\lambda_i(F)| < 1`` for all ``i``.

### Estimation

AR models support two estimation methods:

**OLS** (`:ols`, default): For AR(p), construct the regression:

```math
y_t = \beta_0 + \beta_1 y_{t-1} + \cdots + \beta_p y_{t-p} + \varepsilon_t
```

and apply ordinary least squares. This is consistent and asymptotically efficient for stationary AR processes.

**Maximum Likelihood** (`:mle`): Maximizes the exact Gaussian log-likelihood via the Kalman filter (see [Exact MLE via Kalman Filter](@ref kalman_mle) below).

```julia
using MacroEconometricModels

# Industrial production growth (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

# OLS estimation (default)
ar_ols = estimate_ar(y, 2)

# MLE estimation
ar_mle = estimate_ar(y, 2; method=:mle)

# Access results
ar_ols.phi      # AR coefficients [φ₁, φ₂]
ar_ols.sigma2   # Innovation variance
ar_ols.aic      # Akaike Information Criterion
ar_ols.bic      # Bayesian Information Criterion
```

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
| `iterations` | `Int` | Number of optimization iterations |

**Reference**: Hamilton (1994, Section 5.2)

---

## The MA(q) Model

### Model Specification

A moving average model of order ``q`` is:

```math
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
```

or equivalently:

```math
y_t = c + \theta(L) \varepsilon_t, \quad \theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q
```

### Invertibility

The MA process is invertible if all roots of ``\theta(z) = 0`` lie outside the unit circle. Invertibility ensures a unique MA representation and is checked via the companion matrix eigenvalue condition, identical in form to the AR stationarity check.

### Estimation

MA parameters cannot be estimated by OLS. Three methods are available:

- **CSS** (`:css`): Conditional Sum of Squares — fast, approximate
- **MLE** (`:mle`): Exact MLE via Kalman filter
- **CSS-MLE** (`:css_mle`, default): CSS initialization followed by MLE refinement

```julia
ma_model = estimate_ma(y, 1; method=:css_mle)
ma_model.theta   # MA coefficient [θ₁]
```

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
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method (`:css`, `:mle`, `:css_mle`) |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of optimization iterations |

---

## The ARMA(p,q) Model

### Model Specification

The ARMA(p,q) model combines autoregressive and moving average components:

```math
y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}
```

or in lag-operator form:

```math
\phi(L) y_t = c + \theta(L) \varepsilon_t
```

The process is stationary and invertible when all roots of ``\phi(z)`` and ``\theta(z)`` lie outside the unit circle, respectively.

### Estimation

The same three methods (`:css`, `:mle`, `:css_mle`) are available. The unified internal pipeline `_estimate_arma_internal` dispatches to the appropriate method:

1. **CSS**: Minimizes the conditional sum of squared residuals using Nelder-Mead. Residuals are computed recursively: ``\hat{\varepsilon}_t = y_t - c - \sum_i \phi_i y_{t-i} - \sum_j \theta_j \hat{\varepsilon}_{t-j}``

2. **MLE**: Maximizes the exact Gaussian log-likelihood via the Kalman filter using L-BFGS optimization. The variance parameter is optimized on the log scale for unconstrained optimization.

3. **CSS-MLE** (default): Uses CSS estimates to initialize MLE, combining the robustness of CSS with the efficiency of exact MLE.

```julia
arma_model = estimate_arma(y, 1, 1; method=:css_mle)
arma_model.phi     # AR coefficients
arma_model.theta   # MA coefficients
arma_model.loglik  # Log-likelihood
```

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
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of iterations |

!!! note "Technical Note"
    CSS (Conditional Sum of Squares) is fast but approximate: it conditions on initial residuals being zero, which introduces bias in small samples. MLE via the Kalman filter provides exact inference by properly handling initialization but is computationally more expensive and can be sensitive to starting values. The default `:css_mle` combines both: CSS provides robust starting values, then MLE refines to the exact optimum. For pure AR models, OLS is equivalent to CSS and is preferred for speed.

**Reference**: Hamilton (1994, Chapter 5), Harvey (1993, Chapter 3)

---

## The ARIMA(p,d,q) Model

### Model Specification

The ARIMA(p,d,q) model applies ``d``-fold differencing to produce a stationary series, then fits an ARMA(p,q):

```math
\phi(L) (1-L)^d y_t = c + \theta(L) \varepsilon_t
```

where ``(1-L)^d y_t`` denotes the ``d``-th difference of ``y_t``. Common cases:
- ``d = 1``: ``\Delta y_t = y_t - y_{t-1}`` (first difference, for I(1) series)
- ``d = 2``: ``\Delta^2 y_t = \Delta y_t - \Delta y_{t-1}`` (second difference, for I(2) series)

### Estimation

The implementation differences the series ``d`` times, then estimates ARMA(p,q) on the differenced series using the unified estimation pipeline.

```julia
# Log industrial production — an I(1) series
fred = load_example(:fred_md)
y_level = filter(isfinite, log.(fred[:, "INDPRO"]))

# Fit ARIMA(1,1,0) — differenced once, then AR(1)
model = estimate_arima(y_level, 1, 1, 0)
model.phi    # AR coefficients on differenced series
model.d      # Integration order
```

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
| `aic` | `T` | AIC |
| `bic` | `T` | BIC |
| `method` | `Symbol` | Estimation method |
| `converged` | `Bool` | Convergence indicator |
| `iterations` | `Int` | Number of iterations |

---

## [Exact MLE via Kalman Filter](@id kalman_mle)

### State-Space Representation

For exact maximum likelihood estimation, the ARMA(p,q) model is cast into Harvey's (1993) state-space form:

**Observation equation**:
```math
y_t = c + Z \alpha_t
```

**State equation**:
```math
\alpha_{t+1} = T \alpha_t + R \eta_t, \quad \eta_t \sim N(0, Q)
```

where the state vector ``\alpha_t = [a_t, a_{t-1}, \ldots, a_{t-r+1}]'`` has dimension ``r = \max(p, q+1)``, and:

- ``Z = [1, \theta_1, \ldots, \theta_{r-1}]`` is the observation vector
- ``T`` is the ``r \times r`` companion matrix with AR coefficients in the first row
- ``R = [1, 0, \ldots, 0]'`` is the selection vector
- ``Q = [\sigma^2]`` is the innovation variance

### Kalman Filter Log-Likelihood

The log-likelihood is computed via the prediction error decomposition:

```math
\ell(\Theta) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \sum_{t=1}^n \left( \log f_t + \frac{v_t^2}{f_t} \right)
```

where
- ``v_t = y_t - \hat{y}_{t|t-1}`` is the one-step prediction error
- ``f_t = Z P_{t|t-1} Z' + H`` is its variance
- ``n`` is the number of observations
- ``\Theta`` denotes the full parameter vector ``(\phi_1, \ldots, \phi_p, \theta_1, \ldots, \theta_q, \sigma^2)``

**Initialization**: Uses the unconditional (stationary) distribution when the system is stable, falling back to diffuse initialization (``P_0 = 10^6 I``) for non-stationary parameters.

**Reference**: Harvey (1993, Chapters 3–4), Durbin & Koopman (2012, Chapter 2)

---

## Forecasting

### Point Forecasts

The optimal ``h``-step ahead forecast minimizes mean squared error. For an ARMA(p,q) process, forecasts are computed recursively:

```math
\hat{y}_{T+h|T} = c + \sum_{i=1}^p \phi_i \hat{y}_{T+h-i|T} + \sum_{j=1}^q \theta_j \hat{\varepsilon}_{T+h-j}
```

where ``\hat{y}_{T+k|T} = y_{T+k}`` for ``k \leq 0`` and ``\hat{\varepsilon}_{T+k} = 0`` for ``k \geq 1`` (future residuals are set to zero as the best linear predictor).

### Forecast Uncertainty

Forecast standard errors are derived from the MA(``\infty``) representation. The ``\psi``-weights satisfy:

```math
\psi_j = \sum_{i=1}^{\min(p,j)} \phi_i \psi_{j-i} + \theta_j \mathbb{1}(j \leq q), \quad \psi_0 = 1
```

where
- ``\psi_j`` is the ``j``-th coefficient in the MA(``\infty``) representation ``y_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}``
- ``\phi_i`` are the AR coefficients (zero for ``i > p``)
- ``\theta_j`` are the MA coefficients (zero for ``j > q``)

The ``h``-step ahead forecast variance is:

```math
\text{Var}(e_{T+h|T}) = \sigma^2 \left(1 + \psi_1^2 + \psi_2^2 + \cdots + \psi_{h-1}^2 \right)
```

Confidence intervals are symmetric: ``\hat{y}_{T+h|T} \pm z_{\alpha/2} \cdot \text{se}_h``.

### ARIMA Forecasting

For ARIMA(p,d,q) models, forecasts are computed on the differenced series and then integrated back to the original scale. For ``d = 1``:

```math
\hat{y}_{T+h} = y_T + \sum_{j=1}^h \widehat{\Delta y}_{T+j|T}
```

Standard errors are adjusted for the integration via cumulative variance accumulation.

```julia
# Forecast 12 steps ahead
fc = forecast(model, 12)
fc.forecast    # Point forecasts
fc.ci_lower    # Lower 95% confidence bound
fc.ci_upper    # Upper 95% confidence bound
fc.se          # Standard errors

# Forecast with different confidence level
fc99 = forecast(model, 12; conf_level=0.99)
```

### ARIMAForecast Return Values

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Vector{T}` | Point forecasts ``\hat{y}_{T+1}, \ldots, \hat{y}_{T+h}`` |
| `ci_lower` | `Vector{T}` | Lower confidence bound |
| `ci_upper` | `Vector{T}` | Upper confidence bound |
| `se` | `Vector{T}` | Forecast standard errors (from psi-weights) |
| `horizon` | `Int` | Forecast horizon ``h`` |
| `conf_level` | `T` | Confidence level (e.g., 0.95) |

---

## Order Selection

### Grid Search

`select_arima_order` evaluates all ARMA(p,q) combinations up to specified maxima and selects the best model by AIC or BIC:

```julia
# Search over p ∈ {0,...,4}, q ∈ {0,...,4}
selection = select_arima_order(y, 4, 4)
selection.best_p_bic    # Optimal AR order (BIC)
selection.best_q_bic    # Optimal MA order (BIC)
selection.best_p_aic    # Optimal AR order (AIC)
selection.best_q_aic    # Optimal MA order (AIC)
```

### ARIMAOrderSelection Return Values

| Field | Type | Description |
|-------|------|-------------|
| `best_p_aic` | `Int` | Optimal AR order by AIC |
| `best_q_aic` | `Int` | Optimal MA order by AIC |
| `best_p_bic` | `Int` | Optimal AR order by BIC |
| `best_q_bic` | `Int` | Optimal MA order by BIC |
| `aic_matrix` | `Matrix{T}` | ``(p_{max}+1) \times (q_{max}+1)`` matrix of AIC values |
| `bic_matrix` | `Matrix{T}` | ``(p_{max}+1) \times (q_{max}+1)`` matrix of BIC values |
| `best_model_aic` | `AbstractARIMAModel` | Best model by AIC |
| `best_model_bic` | `AbstractARIMAModel` | Best model by BIC |

### Automatic Selection

`auto_arima` implements an automatic model selection procedure:

```julia
best_model = auto_arima(y)
best_model.p     # Selected AR order
best_model.q     # Selected MA order (for ARMA) or d (for ARIMA)
```

### Information Criteria Table

`ic_table` provides a formatted comparison of models:

```julia
# Get IC values for a grid of models
table = ic_table(y, 3, 3)
```

---

## StatsAPI Interface

All ARIMA models implement the Julia `StatsAPI.RegressionModel` interface:

```julia
using StatsAPI

model = estimate_arma(y, 1, 1)

# StatsAPI accessors
coef(model)         # Coefficient vector
nobs(model)         # Number of observations
dof(model)          # Degrees of freedom (number of parameters)
dof_residual(model) # Residual degrees of freedom
loglikelihood(model) # Log-likelihood
aic(model)          # AIC
bic(model)          # BIC
residuals(model)    # Residual vector
fitted(model)       # Fitted values

# StatsAPI fit interface
model = fit(ARModel, y, 2)           # AR(2)
model = fit(MAModel, y, 1)           # MA(1)
model = fit(ARMAModel, y, 1, 1)      # ARMA(1,1)
model = fit(ARIMAModel, y, 1, 1, 1)  # ARIMA(1,1,1)

# Prediction
yhat = predict(model, 12)  # 12-step point forecasts
```

---

## Complete Example

```julia
using MacroEconometricModels

# Industrial production growth (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, apply_tcode(fred[:, "INDPRO"], 5))

# Step 1: Check for unit root
adf_result = adf_test(y; lags=:aic, regression=:constant)
# IP growth should be stationary — no differencing needed

# Step 2: Select ARMA order
sel = select_arima_order(y, 4, 4)
println("Best order: ARMA($(sel.best_p_bic), $(sel.best_q_bic))")

# Step 3: Estimate the model
model = estimate_arma(y, sel.best_p_bic, sel.best_q_bic)
println("φ = $(model.phi), θ = $(model.theta)")
println("σ² = $(model.sigma2)")
println("AIC = $(model.aic), BIC = $(model.bic)")

# Step 4: Forecast IP growth 12 months ahead
fc = forecast(model, 12; conf_level=0.95)
println("1-step forecast: $(fc.forecast[1]) ± $(1.96 * fc.se[1])")

# Step 5: Diagnostics
println("Converged: $(model.converged)")
println("Log-likelihood: $(model.loglik)")
```

---

## References

- Box, George E. P., and Gwilym M. Jenkins. 1976. *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day. ISBN 978-0-816-21104-3.
- Brockwell, Peter J., and Richard A. Davis. 1991. *Time Series: Theory and Methods*. 2nd ed. New York: Springer. ISBN 978-1-4419-0319-8.
- Durbin, James, and Siem Jan Koopman. 2012. *Time Series Analysis by State Space Methods*. 2nd ed. Oxford: Oxford University Press. [https://doi.org/10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
- Hamilton, James D. 1994. *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.
- Harvey, Andrew C. 1993. *Time Series Models*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-08224-2.
