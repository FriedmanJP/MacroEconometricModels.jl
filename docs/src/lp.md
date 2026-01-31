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

where ``Y_h`` is the matrix of responses at horizon ``h`` and ``X`` contains the shock variable and controls.

### HAC Standard Errors

Since ``\varepsilon_{t+h}`` is serially correlated (at least MA(h-1) under the null), we use Newey-West standard errors:

```math
\hat{V}_{NW} = (X'X)^{-1} \hat{S} (X'X)^{-1}
```

with bandwidth typically set to ``h + 1`` or determined automatically.

**Reference**: Jordà (2005), Newey & West (1987)

### Julia Implementation

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

where ``q`` is the number of instruments. A rule of thumb is ``F > 10`` for strong instruments (Stock & Yogo, 2005).

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
where ``W = \text{diag}(1/\text{Var}(\hat{\beta}_h))``

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

### IPW-LP Estimation

At each horizon ``h``:

```math
\hat{\text{ATE}}_h = \frac{1}{n} \sum_{t: D_t=1} \frac{y_{t+h}}{\hat{p}(X_t)} - \frac{1}{n} \sum_{t: D_t=0} \frac{y_{t+h}}{1-\hat{p}(X_t)}
```

Or via weighted regression:

```math
y_{t+h} = \alpha_h + \beta_h D_t + \gamma_h' X_t + \varepsilon_{t+h}
```

estimated by WLS with IPW weights.

### Doubly Robust Estimation

The doubly robust (DR) estimator combines IPW with outcome regression:

```math
\hat{\text{ATE}}^{DR}_h = \frac{1}{n} \sum_t \left[ \frac{D_t(y_{t+h} - \mu_1(X_t))}{\hat{p}(X_t)} + \mu_1(X_t) \right] - \frac{1}{n} \sum_t \left[ \frac{(1-D_t)(y_{t+h} - \mu_0(X_t))}{1-\hat{p}(X_t)} + \mu_0(X_t) \right]
```

where ``\mu_d(X_t) = E[y_{t+h} | D_t = d, X_t]`` is the outcome regression.

**Property**: DR is consistent if either the propensity score or the outcome model is correctly specified.

### Practical Considerations

1. **Trimming**: Propensity scores near 0 or 1 lead to extreme weights. Trim at [0.01, 0.99].
2. **Overlap**: Verify that treated and control groups have overlapping covariate distributions.
3. **Balance**: Check that covariates are balanced after reweighting (standardized mean differences < 0.1).

**Reference**: Angrist, Jordà & Kuersteiner (2018), Hirano, Imbens & Ridder (2003)

### Julia Implementation

```julia
using MacroEconometricModels

# treatment: Bool vector of treatment indicators
# covariates: matrix of selection-relevant covariates

# IPW estimation
prop_model = estimate_propensity_lp(Y, treatment, covariates, H;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4
)

# Doubly robust estimation
dr_model = doubly_robust_lp(Y, treatment, covariates, H)

# Extract ATE impulse response
ate_irf = propensity_irf(prop_model)

# Diagnostics
diagnostics = propensity_diagnostics(prop_model)
println("Propensity score overlap: ", diagnostics.overlap)
println("Max covariate imbalance: ", diagnostics.balance.max_weighted)
```

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

## References

### Local Projections - Core

- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review*, 95(1), 161-182.
- Plagborg-Møller, M., & Wolf, C. K. (2021). "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica*, 89(2), 955-980.

### LP-IV

- Stock, J. H., & Watson, M. W. (2018). "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *The Economic Journal*, 128(610), 917-948.
- Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models*.

### Smooth LP

- Barnichon, R., & Brownlees, C. (2019). "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics*, 101(3), 522-530.

### State-Dependent LP

- Auerbach, A. J., & Gorodnichenko, Y. (2012). "Measuring the Output Responses to Fiscal Policy." *American Economic Journal: Economic Policy*, 4(2), 1-27.
- Auerbach, A. J., & Gorodnichenko, Y. (2013). "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*.
- Ramey, V. A., & Zubairy, S. (2018). "Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data." *Journal of Political Economy*, 126(2), 850-901.

### Propensity Score Methods

- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics*, 36(3), 371-387.
- Hirano, K., Imbens, G. W., & Ridder, G. (2003). "Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score." *Econometrica*, 71(4), 1161-1189.

### Inference

- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.
