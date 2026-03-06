# [Local Projections](@id lp_page)

**MacroEconometricModels.jl** provides a complete toolkit for estimating impulse response functions via Local Projections (Jordà 2005), an alternative to VAR-based methods that offers greater robustness to dynamic misspecification. The package implements five LP variants, structural identification, LP-based FEVD, and direct multi-step forecasting.

- **Standard LP**: Horizon-by-horizon OLS regressions with Newey-West HAC standard errors
- **LP-IV**: Two-stage least squares with external instruments for endogenous shocks (Stock & Watson 2018)
- **Smooth LP**: B-spline basis functions with roughness penalty for noise reduction (Barnichon & Brownlees 2019)
- **State-Dependent LP**: Logistic smooth-transition models for regime-varying responses (Auerbach & Gorodnichenko 2012)
- **Propensity Score LP**: Inverse propensity weighting and doubly robust estimation for discrete treatments (Angrist, Jordà & Kuersteiner 2018)
- **Structural LP**: VAR-based identification (Cholesky, sign restrictions, ICA, etc.) with LP estimation of dynamic responses (Plagborg-Møller & Wolf 2021)
- **LP-FEVD**: R²-based forecast error variance decomposition with bias correction (Gorodnichenko & Lee 2019)
- **LP Forecasting**: Direct multi-step forecasts with analytical or bootstrap confidence intervals

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

## Quick Start

**Recipe 1: Standard LP with HAC standard errors**

```julia
using MacroEconometricModels

# Load FRED-MD: industrial production, CPI, federal funds rate
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# LP-IRF of a federal funds rate shock up to horizon 20
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west)
result = lp_irf(lp; conf_level=0.95)
report(result)
```

**Recipe 2: LP-IV with external instruments**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Instrument FFR with its own lagged changes
Z = reshape([zeros(1); diff(Y[:, 3])], :, 1)
lpiv = estimate_lp_iv(Y, 3, Z, 20; lags=4, cov_type=:newey_west)
report(lpiv)
```

**Recipe 3: Smooth LP with B-splines**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

slp = estimate_smooth_lp(Y, 3, 20; lambda=1.0, n_knots=4, lags=4)
report(slp)
```

**Recipe 4: Structural LP with Cholesky identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Cholesky ordering: output -> prices -> monetary policy
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
plot_result(slp)
```

**Recipe 5: State-dependent LP (recession vs. expansion)**

```julia
using MacroEconometricModels, Statistics

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# 7-month MA of IP growth as state variable
ip_growth = Y[:, 1]
state_var = [mean(ip_growth[max(1, t-6):t]) for t in 1:length(ip_growth)]
state_var = Float64.((state_var .- mean(state_var)) ./ std(state_var))

slm = estimate_state_lp(Y, 3, state_var, 20; gamma=:estimate, threshold=:estimate, lags=4)
report(slm)
```

**Recipe 6: LP-FEVD with bias correction**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

slp = structural_lp(Y, 20; method=:cholesky, lags=4)
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=500)
plot_result(lfevd)
```

---

## Standard Local Projections

**Local Projections** (Jordà 2005) estimate impulse responses by running a separate predictive regression at each forecast horizon. Unlike VARs, which derive IRFs from a single dynamic system, LPs directly estimate the response at each horizon ``h`` without imposing autoregressive restrictions.

For each horizon ``h = 0, 1, \ldots, H``, the LP regression is:

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, x_t + \gamma_{i,h}' \, w_t + \varepsilon_{i,t+h}
```

where:
- ``y_{i,t+h}`` is the response variable ``i`` at time ``t+h``
- ``x_t`` is the shock variable at time ``t``
- ``w_t = (y_{t-1}', y_{t-2}', \ldots, y_{t-p}')' `` is the vector of lagged controls
- ``\beta_{i,h}`` is the impulse response of variable ``i`` to shock ``x`` at horizon ``h``
- ``\varepsilon_{i,t+h}`` is the regression error

OLS at each horizon yields:

```math
\hat{\beta}_h = (X'X)^{-1} X' Y_h
```

where:
- ``X`` is the ``T_{\text{eff}} \times k`` regressor matrix (intercept, shock, controls)
- ``Y_h`` is the ``T_{\text{eff}} \times 1`` response vector at horizon ``h``
- ``k = 2 + np`` (intercept + shock + ``p`` lags of ``n`` variables)

### HAC Standard Errors

LP residuals ``\varepsilon_{t+h}`` are serially correlated --- at least MA(``h-1``) under the null --- because overlapping forecast horizons create mechanical dependence. Newey-West HAC standard errors are therefore essential:

```math
\hat{V}_{\text{NW}} = (X'X)^{-1} \, \hat{S} \, (X'X)^{-1}
```

where:
- ``\hat{V}_{\text{NW}}`` is the HAC variance-covariance matrix of ``\hat{\beta}_h``
- ``\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')`` is the long-run covariance estimator
- ``w_j`` are Bartlett kernel weights and ``m`` is the bandwidth

!!! note "Automatic Bandwidth Selection"
    When `bandwidth=0` (the default), the effective bandwidth at each horizon ``h`` is `max(m̂_NW, h+1)` where `m̂_NW` is the Newey-West (1994) data-driven selection. This ensures the bandwidth always accounts for the MA(``h-1``) serial correlation structure induced by the overlapping projection.

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Estimate LP-IRF of a federal funds rate shock up to horizon 20
lp_model = estimate_lp(Y, 3, 20;       # shock_var=3 (FEDFUNDS)
    lags = 4,                           # Control lags
    cov_type = :newey_west,             # HAC standard errors
    bandwidth = 0                       # 0 = automatic bandwidth
)

# Extract IRF with confidence intervals
irf_result = lp_irf(lp_model; conf_level=0.95)
report(irf_result)
plot_result(irf_result)
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `irf_result.values` matrix has dimension ``(H+1) \times n_{\text{resp}}``, where each row gives the response at a particular horizon. At ``h = 0``, the coefficient ``\hat{\beta}_0`` captures the contemporaneous impact effect. Standard errors in `irf_result.se` widen as ``h`` increases because longer-horizon LP residuals exhibit stronger serial correlation and the effective sample shrinks by one observation per horizon.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `4` | Number of control lags ``p`` |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator (`:newey_west`, `:white`, `:ols`) |
| `bandwidth` | `Int` | `0` | HAC bandwidth (0 = automatic) |
| `response_vars` | `Vector{Int}` | all | Indices of response variables |
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |

### Return Values (`LPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Index of the shock variable |
| `response_vars` | `Vector{Int}` | Indices of response variables |
| `horizon` | `Int` | Maximum horizon ``H`` |
| `lags` | `Int` | Number of control lags |
| `B` | `Vector{Matrix{T}}` | Coefficient matrices (one per horizon) |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices (HAC) |
| `T_eff` | `Vector{Int}` | Effective sample size at each horizon |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

### Return Values (`LPImpulseResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Matrix{T}` | ``(H+1) \times n_{\text{resp}}`` IRF point estimates |
| `ci_lower` | `Matrix{T}` | Lower confidence bounds |
| `ci_upper` | `Matrix{T}` | Upper confidence bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum horizon |
| `response_vars` | `Vector{String}` | Response variable names |
| `shock_var` | `String` | Shock variable name |
| `cov_type` | `Symbol` | Covariance estimator type |
| `conf_level` | `T` | Confidence level |

---

## LP with Instrumental Variables

When the shock variable ``x_t`` is endogenous or measured with error, external instruments provide identification. Stock & Watson (2018) develop the **LP-IV** methodology using two-stage least squares at each horizon.

**First stage** --- regress the endogenous shock on instruments and controls:

```math
x_t = \pi_0 + \pi_1' z_t + \pi_2' w_t + v_t
```

**Second stage** --- use fitted values in the LP regression:

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, \hat{x}_t + \gamma_{i,h}' \, w_t + \varepsilon_{i,t+h}
```

where:
- ``z_t`` is the vector of external instruments
- ``\hat{x}_t`` is the first-stage fitted value
- ``\beta_{i,h}`` is the instrumented impulse response at horizon ``h``

### Instrument Relevance

The first-stage F-statistic tests whether instruments predict the shock:

```math
F = \frac{\hat{\pi}_1' \, \hat{V}_{\pi}^{-1} \, \hat{\pi}_1}{q}
```

where:
- ``\hat{\pi}_1`` is the vector of first-stage coefficients on the instruments
- ``\hat{V}_{\pi}`` is the HAC variance-covariance of ``\hat{\pi}_1``
- ``q`` is the number of instruments

A rule of thumb requires ``F > 10`` for strong instruments (Stock & Yogo 2005).

!!! note "HAC-Robust F-Statistic"
    The first-stage F-statistic uses Newey-West HAC standard errors, consistent with the second-stage inference. This accounts for the MA(``h-1``) serial correlation in LP residuals. The HAC bandwidth follows the same automatic selection as the second stage.

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Construct instrument: lagged changes in the federal funds rate
Z = reshape([zeros(1); diff(Y[:, 3])], :, 1)

# LP-IV: instrument FFR with its own lagged changes
lpiv_model = estimate_lp_iv(Y, 3, Z, 20;    # shock_var=3 (FEDFUNDS)
    lags = 4,
    cov_type = :newey_west
)

# Check first-stage strength
weak_test = weak_instrument_test(lpiv_model; threshold=10.0)
report(lpiv_model)
```

The `weak_test.min_F` reports the minimum first-stage F-statistic across all horizons. If it exceeds the Stock & Yogo (2005) threshold of 10, the instruments are strong at every horizon. First-stage strength typically declines at longer horizons because the instrument's predictive power weakens. If `weak_test.passes_threshold` is `false`, consider Anderson-Rubin confidence sets for robust inference at affected horizons.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `4` | Number of control lags |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |
| `bandwidth` | `Int` | `0` | HAC bandwidth (0 = automatic) |
| `response_vars` | `Vector{Int}` | all | Response variable indices |

### Return Values (`LPIVModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Index of the endogenous shock variable |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `instruments` | `Matrix{T}` | External instrument matrix |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `B` | `Vector{Matrix{T}}` | Second-stage coefficient matrices |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices |
| `first_stage_F` | `Vector{T}` | First-stage F-statistics by horizon |
| `first_stage_coef` | `Vector{Vector{T}}` | First-stage instrument coefficients |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Smooth Local Projections

Standard LPs produce noisy impulse responses because each horizon is estimated independently. Barnichon & Brownlees (2019) propose **Smooth Local Projections** that parameterize the IRF as a smooth function of the horizon using B-spline basis functions, trading some bias for substantial variance reduction.

The impulse response is modeled as:

```math
\beta(h) = \sum_{j=1}^{J} \theta_j \, B_j(h)
```

where:
- ``\theta_j`` are spline coefficients
- ``B_j(h)`` are cubic B-spline basis functions evaluated at horizon ``h``
- ``J`` is the number of basis functions (determined by the knot count and degree)

### Penalized Estimation

Estimation proceeds in two steps. First, standard LP produces ``\hat{\beta}_h`` and ``\text{Var}(\hat{\beta}_h)`` at each horizon. Second, a weighted penalized spline fit imposes smoothness:

```math
\hat{\theta} = \left( B' W B + \lambda R \right)^{-1} B' W \hat{\beta}
```

where:
- ``B`` is the ``(H+1) \times J`` basis matrix
- ``W = \text{diag}(1/\text{Var}(\hat{\beta}_h))`` is the precision-weight matrix
- ``R`` is the ``J \times J`` roughness penalty matrix with ``R_{ij} = \int B_i''(x) \, B_j''(x) \, dx``
- ``\lambda \geq 0`` is the smoothing parameter (``\lambda = 0`` gives unpenalized fit)

The smoothing parameter ``\lambda`` controls the bias-variance trade-off. Larger values impose more smoothness, shrinking the IRF toward a low-frequency polynomial. Cross-validation selects the ``\lambda`` that minimizes out-of-sample prediction error.

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Smooth LP with cubic splines
smooth_model = estimate_smooth_lp(Y, 3, 20;   # shock_var=3 (FEDFUNDS)
    degree = 3,           # Cubic splines
    n_knots = 4,          # Interior knots
    lambda = 1.0,         # Smoothing penalty
    lags = 4
)
report(smooth_model)

# Automatic lambda selection via cross-validation
optimal_lambda = cross_validate_lambda(Y, 3, 20;
    lambda_grid = 10.0 .^ (-4:0.5:2),
    k_folds = 5
)

# Compare smooth vs standard LP
comparison = compare_smooth_lp(Y, 3, 20; lambda=optimal_lambda)
```

The `smooth_model.irf_values` matrix contains the smoothed impulse responses. Comparing `comparison.variance_reduction` against zero reveals whether the smooth IRF achieves lower pointwise variance --- a favorable trade-off in moderate samples where standard LP confidence bands are wide. Cross-validation balances bias and variance automatically.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `3` | B-spline degree (3 = cubic) |
| `n_knots` | `Int` | `4` | Number of interior knots |
| `lambda` | `Real` | `1.0` | Smoothing penalty parameter |
| `lags` | `Int` | `4` | Number of control lags |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |

### Return Values (`SmoothLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Shock variable index |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `spline_basis` | `BSplineBasis{T}` | B-spline basis (knots, degree, basis matrix) |
| `theta` | `Matrix{T}` | Spline coefficients |
| `vcov_theta` | `Matrix{T}` | Variance-covariance of spline coefficients |
| `lambda` | `T` | Smoothing penalty parameter |
| `irf_values` | `Matrix{T}` | Smoothed IRF point estimates |
| `irf_se` | `Matrix{T}` | Standard errors of smoothed IRF |
| `residuals` | `Matrix{T}` | Regression residuals |
| `T_eff` | `Int` | Effective sample size |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## State-Dependent Local Projections

Economic responses may differ across states of the economy. Auerbach & Gorodnichenko (2012, 2013) develop **state-dependent LPs** using smooth transition functions to estimate regime-varying impulse responses --- for example, whether fiscal multipliers differ between recessions and expansions.

The state-dependent model is:

```math
y_{t+h} = F(z_t) \left[ \alpha_E + \beta_E \, x_t + \gamma_E' \, w_t \right] + (1 - F(z_t)) \left[ \alpha_R + \beta_R \, x_t + \gamma_R' \, w_t \right] + \varepsilon_{t+h}
```

where:
- ``F(z_t)`` is the logistic smooth transition function
- ``z_t`` is the state variable (e.g., moving average of GDP growth)
- ``\beta_E`` is the expansion regime impulse response (``F \to 0``)
- ``\beta_R`` is the recession regime impulse response (``F \to 1``)

### Logistic Transition Function

```math
F(z_t) = \frac{\exp(-\gamma(z_t - c))}{1 + \exp(-\gamma(z_t - c))}
```

where:
- ``\gamma > 0`` controls the transition speed (higher = sharper regime switching)
- ``c`` is the threshold parameter (often 0 for standardized ``z_t``)

The function satisfies ``F(z) \to 1`` as ``z \to -\infty`` (deep recession), ``F(z) \to 0`` as ``z \to +\infty`` (strong expansion), and ``F(c) = 0.5`` (neutral state).

### Regime Difference Test

Testing whether responses differ across regimes uses a Wald-type test at each horizon:

```math
t = \frac{\hat{\beta}_E - \hat{\beta}_R}{\sqrt{\text{Var}(\hat{\beta}_E) + \text{Var}(\hat{\beta}_R) - 2\text{Cov}(\hat{\beta}_E, \hat{\beta}_R)}}
```

where:
- ``\hat{\beta}_E, \hat{\beta}_R`` are the regime-specific impulse responses
- The variance-covariance terms use HAC standard errors

!!! note "Optimization of Transition Parameters"
    When `gamma=:estimate` and `threshold=:estimate`, the transition parameters ``(\gamma, c)`` are jointly optimized using Nelder-Mead over the nonlinear least squares objective. The threshold ``c`` is box-constrained within the data's interquartile range, and ``\gamma > 0`` is enforced.

```julia
using MacroEconometricModels, Statistics

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Construct state variable: 7-month MA of industrial production growth
ip_growth = Y[:, 1]
state_var = [mean(ip_growth[max(1, t-6):t]) for t in 1:length(ip_growth)]
state_var = Float64.((state_var .- mean(state_var)) ./ std(state_var))

# Estimate state-dependent LP: FFR shock with IP growth as state
state_model = estimate_state_lp(Y, 3, state_var, 20;
    gamma = :estimate,
    threshold = :estimate,
    lags = 4
)
report(state_model)

# Extract regime-specific IRFs
irf_expansion = state_irf(state_model; regime=:expansion)
irf_recession = state_irf(state_model; regime=:recession)

# Test for regime differences at each horizon
diff_test = test_regime_difference(state_model)
```

The `irf_expansion` and `irf_recession` objects contain regime-specific impulse responses. Comparing them reveals whether a monetary policy shock has asymmetric effects across the business cycle. The `test_regime_difference` function computes a Wald-type test of ``H_0: \beta_E = \beta_R`` at each horizon using HAC standard errors; rejection implies statistically significant state dependence.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `gamma` | `Real` or `Symbol` | `:estimate` | Transition speed (`:estimate` for optimization) |
| `threshold` | `Real` or `Symbol` | `:estimate` | Threshold parameter (`:estimate` for optimization) |
| `lags` | `Int` | `4` | Number of control lags |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |

### Return Values (`StateLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Shock variable index |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `state` | `StateTransition{T}` | State transition function (``\gamma``, threshold, ``F(z_t)`` values) |
| `B_expansion` | `Vector{Matrix{T}}` | Expansion regime coefficients |
| `B_recession` | `Vector{Matrix{T}}` | Recession regime coefficients |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov_expansion` | `Vector{Matrix{T}}` | Expansion regime variance-covariance |
| `vcov_recession` | `Vector{Matrix{T}}` | Recession regime variance-covariance |
| `vcov_diff` | `Vector{Matrix{T}}` | Variance-covariance of regime difference |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Propensity Score Local Projections

When the shock is a discrete treatment (e.g., a policy intervention), selection bias may confound causal inference. Angrist, Jordà & Kuersteiner (2018) develop **LP with inverse propensity weighting (IPW)** to address treatment selection. The package provides two estimators: IPW and doubly robust (AIPW).

### IPW Estimator

Let ``D_t \in \{0, 1\}`` be a binary treatment indicator. The propensity score is:

```math
p(X_t) = P(D_t = 1 \mid X_t) = \frac{1}{1 + \exp(-X_t' \beta)}
```

where:
- ``D_t`` is the treatment indicator
- ``X_t`` is the vector of covariates
- ``p(X_t)`` is the estimated probability of treatment

The IPW-LP estimator reweights observations via weighted least squares:

```math
y_{t+h} = \alpha_h + \beta_h \, D_t + \gamma_h' \, W_t + \varepsilon_{t+h}
```

where:
- ``W_t`` includes lagged outcomes and covariates
- Weights are ``w_t = 1/\hat{p}(X_t)`` for treated and ``w_t = 1/(1 - \hat{p}(X_t))`` for control observations
- ``\beta_h`` is the Average Treatment Effect (ATE) at horizon ``h``

### Doubly Robust Estimator

The **doubly robust (DR)** estimator combines IPW with separate outcome regressions for treated and control groups. It computes the ATE from the influence function:

```math
\hat{\text{ATE}}_h^{\text{DR}} = \frac{1}{n} \sum_{t=1}^{n} \hat{\psi}_t
```

where:
- ``\hat{\psi}_t`` is the doubly robust influence function combining IPW and outcome model predictions
- ``\hat{\mu}_1(X_t) = E[y_{t+h} \mid D_t = 1, X_t]`` and ``\hat{\mu}_0(X_t) = E[y_{t+h} \mid D_t = 0, X_t]`` are outcome regressions

The DR estimator is consistent if **either** the propensity score model **or** the outcome regression model is correctly specified, providing insurance against single-model misspecification.

| Feature | `estimate_propensity_lp` (IPW) | `doubly_robust_lp` (DR/AIPW) |
|---------|-------------------------------|------------------------------|
| **Method** | WLS with inverse propensity weights | Influence function combining IPW + outcome regression |
| **Consistency requires** | Correct propensity model | Correct propensity **or** outcome model |
| **Best when** | Propensity model well-specified | Uncertainty about either model |

!!! note "Recommendation"
    Use `doubly_robust_lp` as the default. It is never worse than IPW asymptotically, and can be substantially better when the propensity model is misspecified. Use `estimate_propensity_lp` when you have strong confidence in the propensity score specification or want direct WLS coefficients.

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Construct binary treatment: large absolute FFR changes (top quartile)
ffr_changes = abs.(diff(Y[:, 3]))
treatment = Bool.(ffr_changes .> quantile(ffr_changes, 0.75))
Y_trim = Y[2:end, :]
covariates = Y_trim[:, 1:2]

# IPW estimation
ipw_model = estimate_propensity_lp(Y_trim, treatment, covariates, 20;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4
)

# Doubly robust estimation
dr_model = doubly_robust_lp(Y_trim, treatment, covariates, 20;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4
)

# Extract ATE impulse responses
ate_irf = propensity_irf(ipw_model)
dr_irf = propensity_irf(dr_model)
report(dr_model)

# Diagnostics: overlap and covariate balance
diagnostics = propensity_diagnostics(ipw_model)
```

The `ate_irf` and `dr_irf` objects contain the estimated Average Treatment Effect of large FFR changes at each horizon. The diagnostics check overlap (sufficient common support between treated and control distributions) and balance (covariate means equalized after reweighting, with standardized differences below 0.1).

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ps_method` | `Symbol` | `:logit` | Propensity score model (`:logit`, `:probit`) |
| `trimming` | `Tuple` | `(0.01, 0.99)` | Propensity score trimming bounds |
| `lags` | `Int` | `4` | Number of control lags |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |

### Return Values (`PropensityLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `treatment` | `Vector{Bool}` | Binary treatment indicator |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `covariates` | `Matrix{T}` | Selection-relevant covariates |
| `horizon` | `Int` | Maximum horizon |
| `propensity_scores` | `Vector{T}` | Estimated propensity scores ``\hat{p}(X_t)`` |
| `ipw_weights` | `Vector{T}` | Inverse propensity weights |
| `B` | `Vector{Matrix{T}}` | Weighted regression coefficients |
| `residuals` | `Vector{Matrix{T}}` | Weighted residuals |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices |
| `ate` | `Matrix{T}` | Average treatment effect estimates |
| `ate_se` | `Matrix{T}` | Standard errors of ATE |
| `config` | `PropensityScoreConfig{T}` | Configuration (method, trimming, normalize) |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Structural Local Projections

**Structural Local Projections** combine VAR-based identification with LP estimation of dynamic responses. Plagborg-Møller & Wolf (2021) show that under correct specification, LP and VAR estimate the same impulse responses. Structural LP leverages this equivalence by using the VAR only for shock identification (computing the rotation matrix ``Q``), then estimating dynamics via LP regressions --- gaining LP's robustness while retaining SVAR's structural interpretability.

The procedure proceeds in four steps:

1. **Estimate VAR(p)**: Fit a VAR on ``Y`` to obtain the residual covariance ``\hat{\Sigma}`` and reduced-form residuals ``\hat{u}_t``
2. **Identify structural shocks**: Compute the rotation matrix ``Q`` via the chosen identification method
3. **Recover structural shocks**: Compute ``\hat{\varepsilon}_t = Q' L^{-1} \hat{u}_t`` where ``L = \text{chol}(\hat{\Sigma})``
4. **Run LP regressions**: For each structural shock ``j``, estimate:

```math
y_{i,t+h} = \alpha_{i,h}^{(j)} + \beta_{i,h}^{(j)} \, \hat{\varepsilon}_{j,t} + \gamma_{i,h}^{(j)\prime} \, w_t + u_{i,t+h}^{(j)}
```

where:
- ``\hat{\varepsilon}_{j,t}`` is the identified structural shock ``j``
- ``\beta_{i,h}^{(j)}`` is the structural impulse response of variable ``i`` to shock ``j`` at horizon ``h``
- ``w_t`` contains lagged values of ``Y`` as controls

The 3D IRF array stores ``\Theta[h, i, j] = \hat{\beta}_{i,h}^{(j)}`` for ``h = 1, \ldots, H``.

### Identification Methods

| Method | Keyword | Description |
|--------|---------|-------------|
| Cholesky | `:cholesky` | Recursive ordering (lower triangular ``B_0``) |
| Sign restrictions | `:sign` | Constrain signs of responses (Uhlig 2005) |
| Long-run | `:long_run` | Blanchard-Quah (1989) zero long-run effect |
| Narrative | `:narrative` | Historical events + sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018) |
| FastICA | `:fastica` | Non-Gaussian ICA (Hyvärinen 1999) |
| JADE | `:jade` | Joint Approximate Diagonalization of Eigenmatrices |
| SOBI | `:sobi` | Second-Order Blind Identification |
| dCov | `:dcov` | Distance covariance independence criterion |
| HSIC | `:hsic` | Hilbert-Schmidt independence criterion |
| Student-t ML | `:student_t` | Maximum likelihood with Student-t errors |
| Mixture-normal ML | `:mixture_normal` | Gaussian mixture ML |
| PML | `:pml` | Pseudo maximum likelihood |

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD monetary policy dataset: [INDPRO, CPI, FFR]
# Cholesky ordering: output -> prices -> monetary policy
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
report(slp)
plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `slp.irf.values` array has shape ``H \times n \times n``, where `values[h, i, j]` gives the response of variable ``i`` to structural shock ``j`` at horizon ``h``. Under Cholesky identification with ordering [INDPRO, CPI, FFR], the monetary policy shock (shock 3) affects all variables contemporaneously, but the federal funds rate does not respond to output or price shocks within the period. Standard errors in `slp.se` are computed from HAC-corrected LP regressions and tend to be wider than VAR-based IRF confidence bands, reflecting the efficiency cost of LP's robustness.

```julia
# With bootstrap confidence intervals
slp_ci = structural_lp(Y, 20; method=:cholesky, ci_type=:bootstrap, reps=500)

# With sign restrictions: positive supply shock raises output and lowers prices
check_fn(irf) = irf[1, 1, 1] > 0 && irf[1, 2, 1] < 0
slp_sign = structural_lp(Y, 20; method=:sign, check_func=check_fn)

# Dispatch to FEVD and historical decomposition
decomp = fevd(slp, 20)
hd = historical_decomposition(slp)
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (see table above) |
| `lags` | `Int` | `4` | Number of LP control lags |
| `cov_type` | `Symbol` | `:newey_west` | HAC estimator type |
| `ci_type` | `Symbol` | `:none` | CI method (`:none`, `:bootstrap`) |
| `reps` | `Int` | `500` | Bootstrap replications |
| `check_func` | `Function` | `nothing` | Sign restriction check function |

### Return Values (`StructuralLP`)

| Field | Type | Description |
|-------|------|-------------|
| `irf` | `ImpulseResponse{T}` | 3D IRF result (``H \times n \times n``) with optional bootstrap CIs |
| `structural_shocks` | `Matrix{T}` | ``T_{\text{eff}} \times n`` recovered structural shocks |
| `var_model` | `VARModel{T}` | Underlying VAR model used for identification |
| `Q` | `Matrix{T}` | ``n \times n`` rotation/identification matrix |
| `method` | `Symbol` | Identification method used |
| `lags` | `Int` | Number of LP control lags |
| `cov_type` | `Symbol` | HAC estimator type |
| `se` | `Array{T,3}` | ``H \times n \times n`` standard errors |
| `lp_models` | `Vector{LPModel{T}}` | Individual LP model per shock |

---

## LP Forecasting

LP-based forecasts use horizon-specific regression coefficients directly --- no VAR recursion required. For each horizon ``h = 1, \ldots, H``, the direct multi-step forecast is:

```math
\hat{y}_{T+h} = \hat{\alpha}_h + \hat{\beta}_h \cdot s_h + \hat{\Gamma}_h \, w_T
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step-ahead point forecast
- ``s_h`` is the assumed shock path value at horizon ``h``
- ``\hat{\Gamma}_h`` is the coefficient vector on controls ``w_T`` (last ``p`` observations)
- ``\hat{\alpha}_h`` is the horizon-specific intercept

This direct approach avoids compounding misspecification errors across horizons, unlike iterated VAR forecasts.

### Confidence Intervals

| Method | Description |
|--------|-------------|
| `:analytical` | HAC standard errors + normal quantiles: ``\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \hat{\sigma}_h`` |
| `:bootstrap` | Residual resampling with percentile CIs |
| `:none` | Point forecasts only |

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Estimate LP model
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west)

# Forecast with a unit shock path
shock_path = ones(20)
fc = forecast(lp, shock_path; ci_method=:analytical, conf_level=0.95)
report(fc)
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_lp.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `fc.forecast` matrix has shape ``H \times n_{\text{resp}}``, where each row gives the point forecast at a given horizon. Analytical CIs widen with the horizon because LP regression residuals exhibit increasing variance at longer horizons and the effective sample shrinks. Bootstrap CIs are more reliable in small samples because they do not rely on the normal approximation.

```julia
# Structural LP forecast with monetary policy shock
using Random; Random.seed!(42)
slp = structural_lp(Y, 20; method=:cholesky)
fc_struct = forecast(slp, 3, shock_path;  # shock_idx=3 (monetary policy)
                     ci_method=:bootstrap, n_boot=500)
report(fc_struct)
```

### Return Values (`LPForecast`)

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Matrix{T}` | ``H \times n_{\text{resp}}`` point forecasts |
| `ci_lower` | `Matrix{T}` | Lower CI bounds |
| `ci_upper` | `Matrix{T}` | Upper CI bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum forecast horizon ``H`` |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `shock_var` | `Int` | Shock variable index |
| `shock_path` | `Vector{T}` | Assumed shock trajectory |
| `conf_level` | `T` | Confidence level |
| `ci_method` | `Symbol` | CI method used |

---

## LP-Based FEVD

Standard FEVD computes variance shares from the VMA representation, inheriting any VAR misspecification. Gorodnichenko & Lee (2019) propose an **LP-based FEVD** that estimates variance shares directly via R² regressions, inheriting LP's robustness properties.

### The R² Estimator

At each horizon ``h``, the share of variable ``i``'s forecast error variance due to shock ``j`` is:

```math
\widehat{\text{FEVD}}_{ij}(h) = R^2\!\left(\hat{f}_{i,t+h|t-1} \sim \hat{\varepsilon}_{j,t+h}, \hat{\varepsilon}_{j,t+h-1}, \ldots, \hat{\varepsilon}_{j,t}\right)
```

where:
- ``\hat{f}_{i,t+h|t-1}`` are LP forecast error residuals for variable ``i`` at horizon ``h``
- ``\hat{\varepsilon}_{j,t+k}`` are leads and current values of structural shock ``j``
- ``R^2`` measures the fraction of forecast error variance explained by shock ``j``

### Alternative Estimators

**LP-A estimator** (Gorodnichenko & Lee 2019, Eq. 9):

```math
\hat{s}_{ij}^{A}(h) = \frac{\sum_{k=0}^{h} (\hat{\beta}_{0,ik}^{\text{LP}})^2 \, \hat{\sigma}_{\varepsilon_j}^2}{\text{Var}(\hat{f}_{i,t+h|t-1})}
```

where:
- ``\hat{\beta}_{0,ik}^{\text{LP}}`` is the LP coefficient on shock ``j`` at horizon ``k``
- ``\hat{\sigma}_{\varepsilon_j}^2`` is the variance of structural shock ``j``

**LP-B estimator** (Gorodnichenko & Lee 2019, Eq. 10):

```math
\hat{s}_{ij}^{B}(h) = \frac{\text{numerator}^A}{\text{numerator}^A + \text{Var}(\tilde{v}_{t+h})}
```

where:
- ``\tilde{v}_{t+h}`` are the residuals from the R² regression

LP-B replaces the total forecast error variance denominator with the sum of explained and unexplained components, improving finite-sample performance.

### Bias Correction

LP-FEVD estimates can be biased in finite samples. Following Kilian (1998), the package implements VAR-based bootstrap bias correction:

1. Fit a bivariate VAR(``L``) on ``(z, y)`` with HQIC-selected lag order
2. Compute the "true" FEVD from this VAR
3. Simulate ``B`` bootstrap samples and compute LP-FEVD for each
4. Estimate bias = mean(bootstrap) - true
5. Bias-corrected estimate = raw - bias

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD monetary policy dataset
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Structural LP with Cholesky ordering [INDPRO, CPI, FFR]
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# R²-based LP-FEVD with bias correction
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=500)
report(lfevd)
plot_result(lfevd)
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The raw FEVD proportions in `lfevd.proportions[i, j, h]` give the R² from regressing variable ``i``'s forecast error on shock ``j``'s leads at horizon ``h``. Bias correction matters most at short horizons where finite-sample bias is largest. Comparing the three estimators (`:r2`, `:lp_a`, `:lp_b`) provides a robustness check --- substantial disagreement suggests the VAR specification may be unreliable, in which case LP-based estimates are preferred.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:r2` | Estimator (`:r2`, `:lp_a`, `:lp_b`) |
| `bias_correct` | `Bool` | `false` | Apply bootstrap bias correction |
| `n_boot` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |

### Return Values (`LPFEVD`)

| Field | Type | Description |
|-------|------|-------------|
| `proportions` | `Array{T,3}` | ``n \times n \times H`` raw FEVD estimates |
| `bias_corrected` | `Array{T,3}` | ``n \times n \times H`` bias-corrected FEVD |
| `se` | `Array{T,3}` | Bootstrap standard errors |
| `ci_lower` | `Array{T,3}` | Lower CI bounds |
| `ci_upper` | `Array{T,3}` | Upper CI bounds |
| `method` | `Symbol` | Estimator used |
| `horizon` | `Int` | Maximum FEVD horizon |
| `n_boot` | `Int` | Number of bootstrap replications |
| `conf_level` | `T` | Confidence level |
| `bias_correction` | `Bool` | Whether bias correction was applied |

---

## LP vs. VAR

Plagborg-Møller & Wolf (2021) show that under correct specification, LP and VAR IRFs are asymptotically equivalent:

```math
\sqrt{T}(\hat{\beta}_h^{\text{LP}} - \beta_h) \xrightarrow{d} N(0, V^{\text{LP}}), \qquad \sqrt{T}(\hat{\theta}_h^{\text{VAR}} - \theta_h) \xrightarrow{d} N(0, V^{\text{VAR}})
```

where:
- ``V^{\text{LP}} \geq V^{\text{VAR}}`` (VAR is weakly more efficient under correct specification)
- ``\beta_h = \theta_h`` (both target the same population IRF)

The key trade-off is bias vs. variance:

| Aspect | VAR | Local Projections |
|--------|-----|-------------------|
| **Efficiency** | More efficient if correctly specified | Less efficient, but robust |
| **Bias** | Biased if dynamics misspecified | Consistent under weak conditions |
| **Long horizons** | Compounds specification error | Each horizon estimated directly |
| **Nonlinearities** | Requires extensions | Easy to incorporate |
| **External instruments** | SVAR-IV | LP-IV |

Use LP when concerned about VAR misspecification, when incorporating external instruments or nonlinearities, when working with discrete treatments, or at long horizons where VAR error compounds.

---

## Complete Example

This example demonstrates a full LP workflow --- estimation, structural identification, IRF extraction, FEVD, and forecasting --- using FRED-MD monetary policy data.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD: industrial production, CPI, federal funds rate
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Step 1: Standard LP-IRF with Newey-West standard errors
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west)
irf_result = lp_irf(lp; conf_level=0.95)
report(irf_result)

# Step 2: Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
report(slp)

# Step 3: LP-FEVD with bias correction
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=500)
report(lfevd)

# Step 4: Direct multi-step forecast
shock_path = zeros(20); shock_path[1] = 1.0  # unit impulse
fc = forecast(lp, shock_path; ci_method=:analytical, conf_level=0.95)
report(fc)

# Step 5: Visualize
plot_result(irf_result)
plot_result(slp)
plot_result(lfevd)
plot_result(fc)
```

The `estimate_lp` call fits 21 horizon-specific OLS regressions (``h = 0, \ldots, 20``) with Newey-West HAC standard errors, producing the reduced-form IRF of a federal funds rate innovation. The `structural_lp` call estimates a VAR(4) for Cholesky identification, recovers orthogonalized structural shocks, and re-estimates the LP regressions using each structural shock as the impulse variable. The LP-FEVD uses R² regressions with bootstrap bias correction to decompose forecast error variance without relying on the VMA representation. The direct forecast projects each response variable forward using the LP coefficients and an assumed unit-impulse shock path.

---

## Common Pitfalls

1. **Wider confidence intervals than VAR**: LP confidence bands are wider than VAR-based bands by construction. This reflects the efficiency cost of not imposing dynamic restrictions, not a deficiency. If LP and VAR point estimates agree but LP CIs are much wider, the VAR specification is likely correct and the VAR-based inference is more powerful.

2. **Newey-West bandwidth too small**: The default automatic bandwidth ensures ``m \geq h + 1`` at each horizon ``h``, accounting for the MA(``h-1``) serial correlation. Manually setting a small bandwidth (e.g., `bandwidth=1`) produces invalid standard errors at horizons ``h > 1``. Use `bandwidth=0` for automatic selection.

3. **State variable choice in state-dependent LP**: The state variable ``z_t`` must be predetermined (known at time ``t`` before the shock realization). Using a contemporaneous variable creates endogeneity. The standard choice is a backward-looking moving average of GDP growth, standardized to zero mean and unit variance.

4. **Propensity score overlap**: Extreme propensity scores (near 0 or 1) produce large inverse weights that inflate variance and can cause numerical instability. Always set `trimming=(0.01, 0.99)` to cap extreme weights. Check `propensity_diagnostics()` for overlap violations before interpreting ATE estimates.

5. **Effective sample shrinks with horizon**: Each horizon ``h`` loses ``h`` observations from the end of the sample. At ``h = 20`` with ``T = 100``, only 80 observations remain. With short samples and long horizons, estimates at large ``h`` are unreliable regardless of the standard error correction.

6. **Smooth LP overfitting with few knots**: Too few interior knots in the B-spline basis restrict the IRF to low-frequency shapes that cannot capture sharp impact effects. Too many knots reduce the smoothing benefit. Use `cross_validate_lambda` to select the smoothing parameter automatically.

---

## References

- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited.
  *Journal of Business & Economic Statistics*, 36(3), 371-387. [DOI](https://doi.org/10.1080/07350015.2016.1204919)

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802-2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the Output Responses to Fiscal Policy.
  *American Economic Journal: Economic Policy*, 4(2), 1-27. [DOI](https://doi.org/10.1257/pol.4.2.1)

- Auerbach, A. J., & Gorodnichenko, Y. (2013). Fiscal Multipliers in Recession and Expansion. In *Fiscal Policy after the Financial Crisis*, 63-98. University of Chicago Press. [DOI](https://doi.org/10.7208/chicago/9780226018584.003.0003)

- Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth Local Projections.
  *Review of Economics and Statistics*, 101(3), 522-530. [DOI](https://doi.org/10.1162/rest_a_00778)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655-673. [DOI](https://doi.org/10.2307/1827924)

- Gorodnichenko, Y., & Lee, B. (2019). Forecast Error Variance Decompositions with Local Projections.
  *Journal of Business & Economic Statistics*, 38(4), 921-933. [DOI](https://doi.org/10.1080/07350015.2019.1610661)

- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score.
  *Econometrica*, 71(4), 1161-1189. [DOI](https://doi.org/10.1111/1468-0262.00442)

- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections.
  *American Economic Review*, 95(1), 161-182. [DOI](https://doi.org/10.1257/0002828053828518)

- Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions.
  *Review of Economics and Statistics*, 80(2), 218-230. [DOI](https://doi.org/10.1162/003465398557465)

- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica*, 55(3), 703-708. [DOI](https://doi.org/10.2307/1913610)

- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
  *Review of Economic Studies*, 61(4), 631-653. [DOI](https://doi.org/10.2307/2297912)

- Plagborg-Møller, M., & Wolf, C. K. (2021). Local Projections and VARs Estimate the Same Impulse Responses.
  *Econometrica*, 89(2), 955-980. [DOI](https://doi.org/10.3982/ECTA17813)

- Ramey, V. A., & Zubairy, S. (2018). Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data.
  *Journal of Political Economy*, 126(2), 850-901. [DOI](https://doi.org/10.1086/696277)

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed.
  *Journal of the American Statistical Association*, 89(427), 846-866. [DOI](https://doi.org/10.1080/01621459.1994.10476818)

- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments.
  *Economic Journal*, 128(610), 917-948. [DOI](https://doi.org/10.1111/ecoj.12593)

- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In D. W. K. Andrews & J. H. Stock (Eds.),
  *Identification and Inference for Econometric Models*, 80-108. Cambridge University Press. [DOI](https://doi.org/10.1017/CBO9780511614491.006)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381-419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)
