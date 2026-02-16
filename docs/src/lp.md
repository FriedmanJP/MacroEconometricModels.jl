# Local Projections

This chapter provides a comprehensive treatment of Local Projection (LP) methods for estimating impulse response functions, an alternative to the VAR-based approach that offers greater robustness and flexibility.

## Introduction

Local Projections, introduced by Jordà (2005), estimate impulse responses by running a series of predictive regressions at each forecast horizon. Unlike VARs, which derive IRFs from a single estimated dynamic system, LPs directly estimate the response at each horizon without imposing the dynamic restrictions inherent in VAR specifications.

### Key Advantages of Local Projections

1. **Robustness to Misspecification**: LPs do not impose the lag structure of VARs, making them robust to dynamic misspecification
2. **Flexibility**: Easy to incorporate nonlinearities, state-dependence, and instrumental variables
3. **Transparency**: Each horizon's estimate is independent, making the source of identification transparent
4. **Inference**: Standard regression-based inference applies (with HAC corrections)

**Reference**: Jordà (2005), Plagborg-Møller & Wolf (2021)

## Quick Start

```julia
lp = estimate_lp(Y, 1, 20; lags=4, cov_type=:newey_west)               # Standard LP
lpiv = estimate_lp_iv(Y, 3, Z, 20; lags=4)                             # LP-IV
slp = estimate_smooth_lp(Y, 1, 20; lambda=1.0, n_knots=4)              # Smooth LP
sdlp = estimate_state_lp(Y, 1, state, 20; gamma=1.5)                   # State-dependent LP
plp = estimate_propensity_lp(Y, treatment, covariates, 20)              # Propensity LP
irf_result = lp_irf(lp; conf_level=0.95)                               # Extract IRF
struc = structural_lp(Y, 20; method=:cholesky)                          # Structural LP
fc = forecast(lp, ones(20); ci_method=:analytical)                      # LP forecast
lfevd = lp_fevd(struc, 20; method=:r2, bias_correct=true)              # LP-FEVD
```

---

## Standard Local Projections

### The LP Regression

For each horizon ``h = 0, 1, \ldots, H``, we estimate:

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} x_t + \gamma_{i,h}' w_t + \varepsilon_{i,t+h}
```

where:
- ``y_{i,t+h}`` is the response variable ``i`` at time ``t+h``
- ``x_t`` is the shock/treatment variable at time ``t``
- ``w_t`` is a vector of controls (typically lagged ``y`` and ``x``)
- ``\beta_{i,h}`` is the impulse response of variable ``i`` to shock ``x`` at horizon ``h``

### Control Variables

Standard controls include lags of all endogenous variables:

```math
w_t = (y_{t-1}', y_{t-2}', \ldots, y_{t-p}', x_{t-1}, \ldots, x_{t-p})'
```

The number of lags ``p`` is typically selected using information criteria or set to match the VAR lag order.

### Estimation

At each horizon ``h``, OLS yields:

```math
\hat{\beta}_h = (X'X)^{-1} X'Y_h
```

where
- ``\hat{\beta}_h`` is the ``k \times 1`` OLS coefficient vector at horizon ``h``
- ``X`` is the ``T_{eff} \times k`` regressor matrix containing the intercept, shock variable, and controls
- ``Y_h`` is the ``T_{eff} \times 1`` vector of responses at horizon ``h``
- ``k = 2 + np`` (intercept + shock + ``p`` lags of ``n`` variables)

### HAC Standard Errors

Since ``\varepsilon_{t+h}`` is serially correlated (at least MA(h-1) under the null), we use Newey-West standard errors:

```math
\hat{V}_{NW} = (X'X)^{-1} \hat{S} (X'X)^{-1}
```

where
- ``\hat{V}_{NW}`` is the HAC variance-covariance matrix of ``\hat{\beta}_h``
- ``\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')`` is the long-run covariance
- ``w_j`` are Bartlett kernel weights, ``m`` is the bandwidth (typically ``h + 1``)

**Reference**: Jordà (2005), Newey & West (1987)

### Julia Implementation

!!! note "Technical Note"
    LP residuals ``\varepsilon_{t+h}`` are serially correlated at least MA(``h-1``) under the null of correct specification, even when the true DGP has i.i.d. errors. This is because overlapping forecast horizons create mechanical dependence. HAC standard errors (Newey-West) are therefore essential for all horizons ``h > 0``. The default bandwidth is set to ``h + 1`` following standard practice.

```julia
using MacroEconometricModels

# Data: Y is T×n matrix of variables
# shock_var is the index of the shock variable
# Estimate LP-IRF up to horizon H

lp_model = estimate_lp(Y, shock_var, H;
    lags = 4,                  # Control lags
    cov_type = :newey_west,    # HAC standard errors
    bandwidth = 0              # 0 = automatic bandwidth
)

# Extract IRF with confidence intervals
irf_result = lp_irf(lp_model; conf_level = 0.95)
```

The `irf_result.values` matrix has dimension ``(H+1) \times n_{resp}``, where each row gives the response at a particular horizon. At ``h = 0``, the coefficient ``\hat{\beta}_0`` captures the contemporaneous (impact) effect of a one-unit innovation in the shock variable on each response variable. The standard errors in `irf_result.se` widen as ``h`` increases because longer-horizon LP residuals exhibit stronger serial correlation, and the effective sample shrinks by one observation per horizon.

### LPModel Return Values

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

### LPImpulseResponse Return Values

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Matrix{T}` | ``(H+1) \times n_{resp}`` IRF point estimates |
| `ci_lower` | `Matrix{T}` | Lower confidence bounds |
| `ci_upper` | `Matrix{T}` | Upper confidence bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum horizon |
| `response_vars` | `Vector{String}` | Response variable names |
| `shock_var` | `String` | Shock variable name |
| `cov_type` | `Symbol` | Covariance estimator type |
| `conf_level` | `T` | Confidence level (e.g., 0.95) |

---

## Local Projections with Instrumental Variables (LP-IV)

### Motivation

When the shock variable ``x_t`` is endogenous or measured with error, we need external instruments for identification. Stock & Watson (2018) develop the LP-IV methodology for using external instruments in a local projection framework.

### The LP-IV Model

We use two-stage least squares (2SLS) at each horizon:

**First Stage**: Regress the endogenous shock on instruments and controls:
```math
x_t = \pi_0 + \pi_1' z_t + \pi_2' w_t + v_t
```

**Second Stage**: Use fitted values in the LP regression:
```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \hat{x}_t + \gamma_{i,h}' w_t + \varepsilon_{i,t+h}
```

where ``z_t`` is the vector of external instruments.

### Identification Assumptions

1. **Relevance**: ``E[z_t x_t] \neq 0`` (instruments predict the shock)
2. **Exogeneity**: ``E[z_t \varepsilon_{t+h}] = 0`` (instruments are uncorrelated with structural errors)

### First-Stage F-Statistic

The first-stage F-statistic tests instrument relevance:

```math
F = \frac{(\hat{\pi}_1' \hat{V}_{\pi}^{-1} \hat{\pi}_1)}{q}
```

where
- ``\hat{\pi}_1`` is the vector of first-stage coefficients on the instruments
- ``\hat{V}_{\pi}`` is the estimated variance-covariance of ``\hat{\pi}_1``
- ``q`` is the number of instruments

A rule of thumb is ``F > 10`` for strong instruments (Stock & Yogo, 2005).

### Weak Instrument Robust Inference

When instruments are weak, standard 2SLS inference is unreliable. Options include:
- Anderson-Rubin confidence sets
- Conditional likelihood ratio tests
- Weak-instrument robust standard errors

**Reference**: Stock & Watson (2018), Stock & Yogo (2005)

### Julia Implementation

```julia
using MacroEconometricModels

# Y: T×n data matrix
# shock_var: index of endogenous shock variable
# Z: T×q matrix of external instruments

lpiv_model = estimate_lp_iv(Y, shock_var, Z, H;
    lags = 4,
    cov_type = :newey_west
)

# Check first-stage strength
weak_test = weak_instrument_test(lpiv_model; threshold = 10.0)
println("Minimum F-statistic: ", weak_test.min_F)
println("All horizons pass: ", weak_test.passes_threshold)

# Extract IRF
irf_iv = lp_iv_irf(lpiv_model)
```

The `weak_test.min_F` reports the minimum first-stage F-statistic across all horizons. If it exceeds the Stock & Yogo (2005) threshold of 10, instruments are considered strong at every horizon. First-stage strength typically declines at longer horizons because the instrument's predictive power for the endogenous shock weakens. If `weak_test.passes_threshold` is `false`, the IV estimates at affected horizons should be interpreted cautiously — consider Anderson-Rubin confidence sets for robust inference.

### LPIVModel Return Values

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

### Motivation

Standard LPs can produce noisy, erratic impulse responses because each horizon is estimated independently. Barnichon & Brownlees (2019) propose **Smooth Local Projections** that parameterize the IRF as a smooth function of the horizon using B-spline basis functions.

### B-Spline Representation

The impulse response is modeled as:

```math
\beta(h) = \sum_{j=1}^{J} \theta_j B_j(h)
```

where ``B_j(h)`` are B-spline basis functions and ``\theta_j`` are spline coefficients.

### Cubic B-Splines

For degree ``d = 3`` (cubic splines), the basis functions are computed recursively using the Cox-de Boor formula:

```math
B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
```

```math
B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x) + \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x)
```

### Smoothness Penalty

To enforce smoothness, we add a roughness penalty on the second derivative:

```math
\min_{\theta} \sum_{h=0}^{H} \left( \hat{\beta}_h - B(h)'\theta \right)^2 + \lambda \int \left( \beta''(h) \right)^2 dh
```

where
- ``\hat{\beta}_h`` are the standard LP estimates at horizon ``h``
- ``B(h)`` is the ``J \times 1`` B-spline basis vector evaluated at ``h``
- ``\theta`` is the ``J \times 1`` vector of spline coefficients
- ``\lambda \geq 0`` is the smoothing penalty (``\lambda = 0`` gives unpenalized fit)

The penalty is computed as ``\theta' R \theta`` where:

```math
R_{ij} = \int B''_i(x) B''_j(x) dx
```

### Two-Step Estimation

1. Estimate standard LP to get ``\hat{\beta}_h`` and ``\text{Var}(\hat{\beta}_h)``
2. Fit weighted penalized spline:
```math
\hat{\theta} = \left( B'WB + \lambda R \right)^{-1} B'W \hat{\beta}
```

where
- ``B`` is the ``(H+1) \times J`` basis matrix
- ``W = \text{diag}(1/\text{Var}(\hat{\beta}_h))`` is the precision-weight matrix
- ``R`` is the ``J \times J`` roughness penalty matrix
- ``\hat{\beta}`` is the ``(H+1) \times 1`` vector of standard LP estimates

### Cross-Validation for λ Selection

The smoothing parameter ``\lambda`` can be selected by k-fold cross-validation to minimize out-of-sample prediction error.

**Reference**: Barnichon & Brownlees (2019)

### Julia Implementation

```julia
using MacroEconometricModels

# Smooth LP with cubic splines
smooth_model = estimate_smooth_lp(Y, shock_var, H;
    degree = 3,           # Cubic splines
    n_knots = 4,          # Interior knots
    lambda = 1.0,         # Smoothing penalty
    lags = 4
)

# Automatic lambda selection via CV
optimal_lambda = cross_validate_lambda(Y, shock_var, H;
    lambda_grid = 10.0 .^ (-4:0.5:2),
    k_folds = 5
)

# Compare smooth vs standard LP
comparison = compare_smooth_lp(Y, shock_var, H; lambda = optimal_lambda)
println("Variance reduction: ", comparison.variance_reduction)
```

Larger ``\lambda`` values impose more smoothness, shrinking the IRF toward a low-frequency polynomial. When `variance_reduction` is positive, the smooth IRF achieves lower pointwise variance at the cost of some bias — a favorable trade-off in moderate samples where standard LP confidence bands are wide. Cross-validation selects the ``\lambda`` that minimizes out-of-sample prediction error, automatically balancing the bias-variance trade-off.

### SmoothLPModel Return Values

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

### Motivation

Economic responses may differ across states of the economy (e.g., recessions vs. expansions). Auerbach & Gorodnichenko (2012, 2013) develop **state-dependent LPs** using smooth transition functions.

### The State-Dependent Model

```math
y_{t+h} = F(z_t) \left[ \alpha_E + \beta_E x_t + \gamma_E' w_t \right] + (1 - F(z_t)) \left[ \alpha_R + \beta_R x_t + \gamma_R' w_t \right] + \varepsilon_{t+h}
```

where:
- ``F(z_t)`` is the smooth transition function
- ``z_t`` is the state variable (e.g., moving average of GDP growth)
- Subscript ``E`` denotes "expansion" regime (``F \to 0``)
- Subscript ``R`` denotes "recession" regime (``F \to 1``)

### Logistic Transition Function

The standard specification uses a logistic function:

```math
F(z_t) = \frac{\exp(-\gamma(z_t - c))}{1 + \exp(-\gamma(z_t - c))}
```

where:
- ``\gamma > 0`` controls the transition speed (higher = sharper)
- ``c`` is the threshold (often set to 0 for standardized ``z_t``)

**Properties**:
- ``F(z) \to 1`` as ``z \to -\infty`` (deep recession)
- ``F(z) \to 0`` as ``z \to +\infty`` (strong expansion)
- ``F(c) = 0.5`` (neutral state)

### State Variable Construction

Following Auerbach & Gorodnichenko, the state variable is typically:

```math
z_t = \frac{1}{k} \sum_{j=0}^{k-1} \Delta y_{t-j}
```

A ``k = 7`` quarter moving average of GDP growth is common, then standardized to have zero mean and unit variance.

### Estimation

The model is estimated by nonlinear least squares or by treating it as a linear regression in the interaction terms. The parameters ``(\gamma, c)`` can be:
1. Fixed based on prior research
2. Estimated via grid search or NLS
3. Selected to maximize fit

### Testing for Regime Differences

Test whether responses differ across regimes:

```math
H_0: \beta_E - \beta_R = 0
```

using a t-test with HAC standard errors:

```math
t = \frac{\hat{\beta}_E - \hat{\beta}_R}{\sqrt{\text{Var}(\hat{\beta}_E) + \text{Var}(\hat{\beta}_R) - 2\text{Cov}(\hat{\beta}_E, \hat{\beta}_R)}}
```

**Reference**: Auerbach & Gorodnichenko (2012, 2013), Ramey & Zubairy (2018)

### Julia Implementation

```julia
using MacroEconometricModels

# Construct state variable (e.g., 7-quarter MA of GDP growth)
gdp_growth = diff(log.(Y[:, 1]))
state_var = [mean(gdp_growth[max(1, t-6):t]) for t in 1:length(gdp_growth)]
state_var = (state_var .- mean(state_var)) ./ std(state_var)

# Estimate state-dependent LP
state_model = estimate_state_lp(Y, shock_var, state_var, H;
    gamma = :estimate,      # Estimate transition speed
    threshold = :median,    # Set threshold at median
    lags = 4
)

# Extract regime-specific IRFs
irf_both = state_irf(state_model; regime = :both)
irf_expansion = state_irf(state_model; regime = :expansion)
irf_recession = state_irf(state_model; regime = :recession)

# Test for regime differences
diff_test = test_regime_difference(state_model)
```

The `irf_expansion` and `irf_recession` objects contain regime-specific impulse responses. Comparing them reveals whether a shock (e.g., fiscal spending) has asymmetric effects across the business cycle — a prediction of many New Keynesian models with binding ZLB or liquidity traps. The `test_regime_difference` function computes a Wald-type test of ``H_0: \beta_E = \beta_R`` at each horizon using HAC standard errors; rejection implies statistically significant state dependence.

### StateLPModel Return Values

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

### Motivation

When the shock is a discrete treatment (e.g., policy intervention), selection bias may confound causal inference. Angrist, Jordà & Kuersteiner (2018) develop **LP with inverse propensity weighting (IPW)** to address selection.

### The Setup

Let ``D_t \in \{0, 1\}`` be a binary treatment indicator. We want to estimate the Average Treatment Effect (ATE):

```math
\text{ATE}_h = E[y_{t+h}(1) - y_{t+h}(0)]
```

where ``y_{t+h}(d)`` is the potential outcome under treatment status ``d``.

### Propensity Score

The propensity score is the probability of treatment given covariates:

```math
p(X_t) = P(D_t = 1 | X_t)
```

estimated via logit or probit:

```math
p(X_t) = \frac{1}{1 + \exp(-X_t'\beta)}
```

### Inverse Propensity Weighting (IPW)

The IPW estimator weights observations by the inverse of their selection probability:

- Treated: weight ``= 1/p(X_t)``
- Control: weight ``= 1/(1-p(X_t))``

This reweighting creates a pseudo-population where treatment is independent of covariates.

### IPW-LP Estimation (`estimate_propensity_lp`)

The IPW estimator reweights observations so that, in the pseudo-population, treatment is independent of covariates. Estimation proceeds via **weighted least squares** (WLS):

```math
y_{t+h} = \alpha_h + \beta_h D_t + \gamma_h' W_t + \varepsilon_{t+h}
```

where ``W_t`` includes lagged outcomes and covariates, and the weights are ``w_t = 1/\hat{p}(X_t)`` for treated and ``w_t = 1/(1-\hat{p}(X_t))`` for control observations. The ATE estimate is the treatment coefficient ``\hat{\beta}_h`` from the WLS regression. Standard errors are computed from HAC-robust covariance on the weighted residuals.

**Consistency requirement**: The propensity score model ``\hat{p}(X_t)`` must be correctly specified. If the logit/probit model misses important nonlinearities or interactions in the true treatment assignment mechanism, the IPW estimator will be biased.

### Doubly Robust LP Estimation (`doubly_robust_lp`)

The doubly robust (DR) estimator, also known as the **augmented IPW (AIPW)** estimator, combines inverse propensity weighting with separate **outcome regressions** for treated and control groups. It first fits:

- ``\hat{\mu}_1(X_t) = E[y_{t+h} | D_t=1, X_t]`` — predicted outcome under treatment
- ``\hat{\mu}_0(X_t) = E[y_{t+h} | D_t=0, X_t]`` — predicted outcome under control

via OLS on the respective subsamples. Then the ATE is computed from the influence function:

```math
\hat{\psi}_t = \begin{cases} \displaystyle\frac{D_t(y_{t+h} - \hat{\mu}_1(X_t))}{\hat{p}(X_t)} + \hat{\mu}_1(X_t) - \hat{\mu}_0(X_t), & \text{if } D_t = 1 \\[6pt] \displaystyle\hat{\mu}_1(X_t) - \frac{(1-D_t)(y_{t+h} - \hat{\mu}_0(X_t))}{1-\hat{p}(X_t)} - \hat{\mu}_0(X_t), & \text{if } D_t = 0 \end{cases}
```

```math
\hat{\text{ATE}}^{DR}_h = \frac{1}{n}\sum_{t=1}^{n} \hat{\psi}_t, \qquad \text{SE}_h = \frac{\text{sd}(\hat{\psi})}{\sqrt{n}}
```

**Double robustness property**: The DR estimator is consistent if **either** the propensity score model **or** the outcome regression model is correctly specified. This provides insurance against misspecification of one component, making it the safer default choice in applied work.

### When Do the Two Estimators Differ?

| Feature | `estimate_propensity_lp` (IPW) | `doubly_robust_lp` (DR/AIPW) |
|---------|-------------------------------|------------------------------|
| **Method** | WLS with inverse propensity weights | Influence function combining IPW + outcome regression |
| **Consistency requires** | Correct propensity model | Correct propensity **or** outcome model |
| **ATE computation** | Treatment coefficient from WLS | Mean of doubly robust influence function ``\hat{\psi}_t`` |
| **Standard errors** | HAC-robust on weighted residuals | ``\text{sd}(\hat{\psi}_t)/\sqrt{n}`` |
| **Best when** | Propensity model well-specified | Uncertainty about either model |

The two estimators coincide only when the propensity score model is exactly correct and the outcome is truly linear in covariates. In practice, they produce different ATE estimates because:

1. **Outcome model bias correction**: DR subtracts the predicted outcome ``\hat{\mu}_d(X_t)`` from the observed outcome before applying IPW. This correction reduces the sensitivity to propensity score misspecification.
2. **Different variance estimation**: IPW uses WLS residual-based HAC standard errors; DR uses the cross-sectional standard deviation of the influence function.

!!! note "Recommendation"
    Use `doubly_robust_lp` as the default. It is never worse than IPW asymptotically, and can be substantially better when the propensity model is misspecified. Use `estimate_propensity_lp` when you have strong confidence in the propensity score specification or want direct WLS coefficients for all regressors.

### Practical Considerations

1. **Trimming**: Propensity scores near 0 or 1 lead to extreme weights. Trim at [0.01, 0.99].
2. **Overlap**: Verify that treated and control groups have overlapping covariate distributions.
3. **Balance**: Check that covariates are balanced after reweighting (standardized mean differences < 0.1).

**Reference**: Angrist, Jordà & Kuersteiner (2018), Hirano, Imbens & Ridder (2003), Robins, Rotnitzky & Zhao (1994)

### Julia Implementation

```julia
using MacroEconometricModels

# treatment: Bool vector of treatment indicators
# covariates: matrix of selection-relevant covariates

# IPW estimation — consistent if propensity model is correct
ipw_model = estimate_propensity_lp(Y, treatment, covariates, H;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4
)

# Doubly robust estimation — consistent if EITHER model is correct
dr_model = doubly_robust_lp(Y, treatment, covariates, H;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4
)

# Compare ATE estimates (they will generally differ)
println("IPW ATE at h=0: ", ipw_model.ate[1, :])
println("DR  ATE at h=0: ", dr_model.ate[1, :])

# Extract ATE impulse response
ate_irf = propensity_irf(ipw_model)
dr_irf  = propensity_irf(dr_model)

# Diagnostics (works for both model types)
diagnostics = propensity_diagnostics(ipw_model)
println("Propensity score overlap: ", diagnostics.overlap)
println("Max covariate imbalance: ", diagnostics.balance.max_weighted)
```

The `ate_irf` object contains the estimated Average Treatment Effect at each horizon. The diagnostics check two key assumptions: overlap (sufficient common support between treated and control distributions) and balance (covariate means equalized after reweighting, with standardized differences below 0.1).

### PropensityLPModel Return Values

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

### Motivation

Standard LP estimates the response to a single shock variable. In multivariate settings, however, we often want to trace the dynamic effects of *orthogonalized structural shocks* — just as in SVAR analysis. **Structural Local Projections** combine VAR-based identification with LP estimation to achieve this.

Plagborg-Møller & Wolf (2021) establish a deep connection: under correct specification, LP and VAR estimate the same impulse responses. Structural LP leverages this equivalence by using the VAR only for shock identification (computing the rotation matrix ``Q``), then estimating the dynamic responses via LP regressions — gaining the robustness of LP while retaining the structural interpretability of SVAR.

### Algorithm

The structural LP procedure proceeds in five steps:

1. **Estimate VAR(p)**: Fit a VAR on the data ``Y`` to obtain the residual covariance ``\hat{\Sigma}`` and reduced-form residuals ``\hat{u}_t``
2. **Identify structural shocks**: Compute the rotation matrix ``Q`` via the chosen identification method (Cholesky, sign restrictions, long-run, ICA, etc.)
3. **Recover structural shocks**: Compute ``\hat{\varepsilon}_t = Q'L^{-1}\hat{u}_t`` where ``L = \text{chol}(\hat{\Sigma})``
4. **Run LP regressions**: For each structural shock ``j``, estimate LP regressions using ``\hat{\varepsilon}_{j,t}`` as the shock variable:

```math
y_{i,t+h} = \alpha_{i,h}^{(j)} + \beta_{i,h}^{(j)} \hat{\varepsilon}_{j,t} + \gamma_{i,h}^{(j)\prime} w_t + u_{i,t+h}^{(j)}
```

where
- ``y_{i,t+h}`` is the response variable ``i`` at horizon ``t+h``
- ``\hat{\varepsilon}_{j,t}`` is the identified structural shock ``j``
- ``w_t`` contains lagged values of ``Y`` as controls
- ``\beta_{i,h}^{(j)}`` is the structural impulse response of variable ``i`` to shock ``j`` at horizon ``h``

5. **Stack into 3D IRF array**: ``\Theta[h, i, j] = \hat{\beta}_{i,h}^{(j)}`` for ``h = 1, \ldots, H``

### Identification Methods

Structural LP supports all identification methods available for SVAR:

| Method | Keyword | Description |
|--------|---------|-------------|
| Cholesky | `:cholesky` | Recursive ordering (lower triangular ``B_0``) |
| Sign restrictions | `:sign` | Constrain signs of responses (Uhlig, 2005) |
| Long-run | `:long_run` | Blanchard-Quah (1989) zero long-run effect |
| Narrative | `:narrative` | Historical events + sign restrictions (Antolín-Díaz & Rubio-Ramírez, 2018) |
| FastICA | `:fastica` | Non-Gaussian ICA (Hyvärinen, 1999) |
| JADE | `:jade` | Joint Approximate Diagonalization of Eigenmatrices |
| SOBI | `:sobi` | Second-Order Blind Identification |
| dCov | `:dcov` | Distance covariance independence criterion |
| HSIC | `:hsic` | Hilbert-Schmidt independence criterion |
| Student-t ML | `:student_t` | Maximum likelihood with Student-t errors |
| Mixture-normal ML | `:mixture_normal` | Gaussian mixture ML |
| PML | `:pml` | Pseudo maximum likelihood |

### Julia Implementation

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
T, n = 200, 3
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(n)
end

# Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# Access 3D IRF array: irfs.values[h, i, j]
println("Shock 1 → Var 1 at h=1: ", round(slp.irf.values[1, 1, 1], digits=4))
println("Shock 2 → Var 1 at h=8: ", round(slp.irf.values[8, 1, 2], digits=4))

# Standard errors
println("SE at h=1: ", round(slp.se[1, 1, 1], digits=4))

# With bootstrap CIs
slp_ci = structural_lp(Y, 20; method=:cholesky, ci_type=:bootstrap, reps=500)

# With sign restrictions
check_fn(irf) = irf[1, 1, 1] > 0 && irf[1, 2, 1] > 0
slp_sign = structural_lp(Y, 20; method=:sign, check_func=check_fn)

# Dispatch to IRF, FEVD, HD
irf_result = irf(slp)           # Returns the ImpulseResponse from StructuralLP
decomp = fevd(slp, 20)          # LP-FEVD (Gorodnichenko & Lee 2019)
hd = historical_decomposition(slp)  # LP-based historical decomposition
```

The `slp.irf.values` array has shape ``H \times n \times n``, where `values[h, i, j]` gives the response of variable ``i`` to structural shock ``j`` at horizon ``h``. Under Cholesky identification, the ordering determines which variables respond contemporaneously to each shock — variable 1 responds only to shock 1 at impact, variable 2 responds to shocks 1 and 2, and so on. The standard errors in `slp.se` are computed from HAC-corrected LP regressions and tend to be wider than VAR-based IRF confidence bands, reflecting the efficiency cost of LP's robustness to dynamic misspecification.

### StructuralLP Return Values

| Field | Type | Description |
|-------|------|-------------|
| `irf` | `ImpulseResponse{T}` | 3D IRF result (``H \times n \times n``) with optional bootstrap CIs |
| `structural_shocks` | `Matrix{T}` | ``T_{eff} \times n`` recovered structural shocks |
| `var_model` | `VARModel{T}` | Underlying VAR model used for identification |
| `Q` | `Matrix{T}` | ``n \times n`` rotation/identification matrix |
| `method` | `Symbol` | Identification method used |
| `lags` | `Int` | Number of LP control lags |
| `cov_type` | `Symbol` | HAC estimator type (`:newey_west`, `:white`) |
| `se` | `Array{T,3}` | ``H \times n \times n`` standard errors |
| `lp_models` | `Vector{LPModel{T}}` | Individual LP model per shock |

**Reference**: Plagborg-Møller & Wolf (2021)

---

## LP Forecasting

### Direct Multi-Step Forecasts

LP-based forecasts use horizon-specific regression coefficients directly — no VAR recursion required. For each horizon ``h = 1, \ldots, H``, the forecast is:

```math
\hat{y}_{T+h} = \hat{\alpha}_h + \hat{\beta}_h \cdot s_h + \hat{\Gamma}_h w_T
```

where
- ``\hat{y}_{T+h}`` is the ``h``-step-ahead point forecast
- ``\hat{\alpha}_h`` is the horizon-specific intercept
- ``\hat{\beta}_h`` is the coefficient on the assumed shock path value ``s_h``
- ``\hat{\Gamma}_h`` is the coefficient vector on controls ``w_T`` (last ``p`` observations of ``Y``)

This "direct" approach has a key advantage over recursive (iterated) VAR forecasts: each horizon uses its own regression, so misspecification in the short-horizon model does not compound into longer horizons.

### Confidence Intervals

Three CI methods are available:

| Method | Description |
|--------|-------------|
| `:analytical` | HAC standard errors + normal quantiles: ``\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \hat{\sigma}_h`` |
| `:bootstrap` | Residual resampling with percentile CIs |
| `:none` | Point forecasts only (no CIs) |

### Julia Implementation

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
T, n = 200, 3
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(n)
end

# Estimate LP model
lp = estimate_lp(Y, 1, 20; lags=4, cov_type=:newey_west)

# Forecast with a unit shock path (1 at all horizons)
shock_path = ones(20)
fc = forecast(lp, shock_path; ci_method=:analytical, conf_level=0.95)

println("Forecast at h=1: ", round(fc.forecasts[1, 1], digits=4))
println("Forecast at h=8: ", round(fc.forecasts[8, 1], digits=4))
println("95% CI at h=8: [", round(fc.ci_lower[8, 1], digits=4),
        ", ", round(fc.ci_upper[8, 1], digits=4), "]")

# Structural LP forecast with a specific shock
slp = structural_lp(Y, 20; method=:cholesky)
fc_struct = forecast(slp, 1, shock_path;  # shock_idx=1
                     ci_method=:bootstrap, n_boot=500)
```

The `fc.forecasts` matrix has shape ``H \times n_{resp}``, where each row gives the point forecast at a given horizon. The analytical CIs widen with the horizon because the LP regression residuals exhibit increasing variance at longer horizons and the effective sample shrinks. The bootstrap CIs are generally more reliable in small samples because they do not rely on the normal approximation; however, they require the LP residuals to be approximately exchangeable, which holds under correct specification.

### LPForecast Return Values

| Field | Type | Description |
|-------|------|-------------|
| `forecasts` | `Matrix{T}` | ``H \times n_{resp}`` point forecasts |
| `ci_lower` | `Matrix{T}` | Lower CI bounds |
| `ci_upper` | `Matrix{T}` | Upper CI bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum forecast horizon ``H`` |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `shock_var` | `Int` | Shock variable index |
| `shock_path` | `Vector{T}` | Assumed shock trajectory |
| `conf_level` | `T` | Confidence level |
| `ci_method` | `Symbol` | CI method used (`:analytical`, `:bootstrap`, `:none`) |

**Reference**: Jordà (2005), Plagborg-Møller & Wolf (2021)

---

## LP-Based FEVD

### Motivation

Standard FEVD computes the share of forecast error variance attributable to each structural shock using the VMA (Vector Moving Average) representation. However, if the VAR is misspecified, VMA-based FEVD inherits those errors. Gorodnichenko & Lee (2019) propose an **LP-based FEVD** that estimates variance shares directly via R² regressions, inheriting the robustness properties of LP.

### The R² Estimator

At each horizon ``h``, the share of variable ``i``'s forecast error variance due to shock ``j`` is estimated by:

1. Obtain LP forecast error residuals ``\hat{f}_{t+h|t-1}`` from the LP regression
2. Regress these residuals on structural shock leads ``[\hat{\varepsilon}_{j,t+h}, \hat{\varepsilon}_{j,t+h-1}, \ldots, \hat{\varepsilon}_{j,t}]``
3. The R² from this regression is the FEVD share:

```math
\widehat{\text{FEVD}}_{ij}(h) = R^2\left(\hat{f}_{i,t+h|t-1} \sim \hat{\varepsilon}_{j,t+h}, \hat{\varepsilon}_{j,t+h-1}, \ldots, \hat{\varepsilon}_{j,t}\right)
```

where
- ``\hat{f}_{i,t+h|t-1}`` are LP forecast error residuals for variable ``i`` at horizon ``h``
- ``\hat{\varepsilon}_{j,t+k}`` are leads and current values of structural shock ``j``
- ``R^2`` measures the fraction of forecast error variance explained by shock ``j``

### Alternative Estimators

Two additional estimators are available:

**LP-A Estimator** (Gorodnichenko & Lee 2019, Eq. 9):
```math
\hat{s}_{ij}^{A}(h) = \frac{\sum_{k=0}^{h} (\hat{\beta}_{0,ik}^{LP})^2 \hat{\sigma}_{\varepsilon_j}^2}{\text{Var}(\hat{f}_{i,t+h|t-1})}
```

where
- ``\hat{\beta}_{0,ik}^{LP}`` is the LP coefficient on shock ``j`` at horizon ``k``
- ``\hat{\sigma}_{\varepsilon_j}^2`` is the variance of structural shock ``j``

**LP-B Estimator** (Gorodnichenko & Lee 2019, Eq. 10):
```math
\hat{s}_{ij}^{B}(h) = \frac{\text{numerator}^A}{\text{numerator}^A + \text{Var}(\tilde{v}_{t+h})}
```

where ``\tilde{v}_{t+h}`` are the residuals from the R² regression. LP-B replaces the total forecast error variance in the denominator with the sum of explained and unexplained components, which can improve finite-sample performance.

### Bias Correction

LP-FEVD estimates can be biased in finite samples. Following Kilian (1998), the package implements VAR-based bootstrap bias correction:

1. Fit a bivariate VAR(``L``) on ``(z, y)`` with HQIC-selected lag order
2. Compute the "true" FEVD from this VAR (theoretical benchmark)
3. Simulate ``B`` bootstrap samples from the VAR
4. For each simulation, compute LP-FEVD and estimate bias = ``\text{mean}(\text{boot}) - \text{true}``
5. Bias-corrected estimate = raw - bias
6. CIs from the centered bootstrap distribution

### Julia Implementation

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
T, n = 200, 3
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(n)
end

# First estimate structural LP
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# R²-based LP-FEVD with bias correction
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=500)

# Access results
println("FEVD of Var 1 due to Shock 1:")
for h in [1, 4, 8, 12, 20]
    raw = round(lfevd.proportions[1, 1, h] * 100, digits=1)
    bc = round(lfevd.bias_corrected[1, 1, h] * 100, digits=1)
    println("  h=$h: raw=$(raw)%, bias-corrected=$(bc)%")
end

# Alternative estimators
lfevd_a = lp_fevd(slp, 20; method=:lp_a)
lfevd_b = lp_fevd(slp, 20; method=:lp_b)

# Via dispatch
decomp = fevd(slp, 20)  # Equivalent to lp_fevd(slp, 20)
```

The raw FEVD proportions in `lfevd.proportions[i, j, h]` give the R² from regressing variable ``i``'s forecast error on shock ``j``'s leads at horizon ``h``. Bias correction typically matters most at short horizons where finite-sample bias is largest. At long horizons, the R²-based and VMA-based FEVD should converge under correct specification. Comparing the three estimators (`:r2`, `:lp_a`, `:lp_b`) provides a robustness check — substantial disagreement suggests the VAR specification may be unreliable, in which case the LP-based estimates are preferred.

### LPFEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `proportions` | `Array{T,3}` | ``n \times n \times H`` raw FEVD estimates: `proportions[i, j, h]` = share of variable ``i``'s FEV due to shock ``j`` at horizon ``h`` |
| `bias_corrected` | `Array{T,3}` | ``n \times n \times H`` bias-corrected FEVD |
| `se` | `Array{T,3}` | Bootstrap standard errors |
| `ci_lower` | `Array{T,3}` | Lower CI bounds |
| `ci_upper` | `Array{T,3}` | Upper CI bounds |
| `method` | `Symbol` | Estimator used (`:r2`, `:lp_a`, `:lp_b`) |
| `horizon` | `Int` | Maximum FEVD horizon |
| `n_boot` | `Int` | Number of bootstrap replications |
| `conf_level` | `T` | Confidence level for CIs |
| `bias_correction` | `Bool` | Whether bias correction was applied |

**Reference**: Gorodnichenko, Yuriy, and Byoungchan Lee. 2019. "Forecast Error Variance Decompositions with Local Projections." *Journal of Business & Economic Statistics* 38 (4): 921–933. [https://doi.org/10.1080/07350015.2019.1610661](https://doi.org/10.1080/07350015.2019.1610661)

---

## Comparing LP and VAR

### LP vs. VAR Trade-offs

| Aspect | VAR | Local Projections |
|--------|-----|-------------------|
| **Efficiency** | More efficient if correctly specified | Less efficient, but robust |
| **Bias** | Biased if dynamics misspecified | Consistent under weak conditions |
| **Long horizons** | Compounds specification error | Each horizon estimated directly |
| **Nonlinearities** | Requires extensions | Easy to incorporate |
| **External instruments** | SVAR-IV | LP-IV |

### Asymptotic Equivalence

Plagborg-Møller & Wolf (2021) show that under correct specification, LP and VAR IRFs are asymptotically equivalent:

```math
\sqrt{T}(\hat{\beta}_h^{LP} - \beta_h) \xrightarrow{d} N(0, V^{LP})
```
```math
\sqrt{T}(\hat{\theta}_h^{VAR} - \theta_h) \xrightarrow{d} N(0, V^{VAR})
```

with ``V^{LP} \geq V^{VAR}`` (VAR is weakly more efficient).

### When to Use LP

- Concerned about VAR misspecification
- Need to incorporate external instruments
- Interested in nonlinear/state-dependent responses
- Working with discrete treatments
- Long horizons where VAR error compounds

**Reference**: Plagborg-Møller & Wolf (2021)

---

### See Also

- [VAR Estimation](manual.md) -- VAR-based impulse responses for comparison with LP estimates
- [Innovation Accounting](innovation_accounting.md) -- LP-FEVD and structural decomposition details
- [Hypothesis Tests](hypothesis_tests.md) -- Unit root and Granger causality tests
- [API Reference](api_functions.md) -- Complete function signatures

## References

### Local Projections - Core

- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161–182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica* 89 (2): 955–980. [https://doi.org/10.3982/ECTA17813](https://doi.org/10.3982/ECTA17813)

### LP-IV

- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917–948. [https://doi.org/10.1111/ecoj.12593](https://doi.org/10.1111/ecoj.12593)
- Stock, James H., and Motohiro Yogo. 2005. "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*, edited by Donald W. K. Andrews and James H. Stock, 80–108. Cambridge: Cambridge University Press.

### Smooth LP

- Barnichon, Regis, and Christian Brownlees. 2019. "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics* 101 (3): 522–530. [https://doi.org/10.1162/rest_a_00778](https://doi.org/10.1162/rest_a_00778)

### State-Dependent LP

- Auerbach, Alan J., and Yuriy Gorodnichenko. 2012. "Measuring the Output Responses to Fiscal Policy." *American Economic Journal: Economic Policy* 4 (2): 1–27. [https://doi.org/10.1257/pol.4.2.1](https://doi.org/10.1257/pol.4.2.1)
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63–98. Chicago: University of Chicago Press. [https://doi.org/10.7208/chicago/9780226018584.003.0003](https://doi.org/10.7208/chicago/9780226018584.003.0003)
- Ramey, Valerie A., and Sarah Zubairy. 2018. "Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data." *Journal of Political Economy* 126 (2): 850–901. [https://doi.org/10.1086/696277](https://doi.org/10.1086/696277)

### Structural LP and LP-FEVD

- Gorodnichenko, Yuriy, and Byoungchan Lee. 2019. "Forecast Error Variance Decompositions with Local Projections." *Journal of Business & Economic Statistics* 38 (4): 921–933. [https://doi.org/10.1080/07350015.2019.1610661](https://doi.org/10.1080/07350015.2019.1610661)
- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions." *Review of Economics and Statistics* 80 (2): 218–230. [https://doi.org/10.1162/003465398557465](https://doi.org/10.1162/003465398557465)

### Propensity Score Methods

- Angrist, Joshua D., Òscar Jordà, and Guido M. Kuersteiner. 2018. "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics* 36 (3): 371–387. [https://doi.org/10.1080/07350015.2016.1204919](https://doi.org/10.1080/07350015.2016.1204919)
- Hirano, Keisuke, Guido W. Imbens, and Geert Ridder. 2003. "Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score." *Econometrica* 71 (4): 1161–1189. [https://doi.org/10.1111/1468-0262.00442](https://doi.org/10.1111/1468-0262.00442)

### Inference

- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
