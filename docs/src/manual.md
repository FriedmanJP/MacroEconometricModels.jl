# [VAR](@id var_page)

**MacroEconometricModels.jl** provides a complete implementation of Vector Autoregression (VAR) models, from reduced-form estimation through structural identification and robust inference. The VAR framework, introduced by Sims (1980), remains the workhorse of empirical macroeconomics for analyzing the dynamic interactions among multiple time series.

- **Estimation**: OLS estimation of reduced-form VAR(p) with automatic information criteria (AIC, BIC, HQIC) and stability checking
- **Lag Selection**: Data-driven lag order selection via AIC, BIC, or HQIC minimization
- **Structural Identification**: Six methods --- Cholesky (recursive), sign restrictions, narrative restrictions, long-run (Blanchard-Quah), Arias et al. (2018) zero + sign, and Mountford-Uhlig (2009) penalty function
- **Robust Inference**: Newey-West HAC, White heteroscedasticity-robust (HC0), and Driscoll-Kraay panel-robust covariance estimators
- **Innovation Accounting**: IRF, FEVD, and historical decomposition with bootstrap or asymptotic confidence intervals; see [Innovation Accounting](innovation_accounting.md)
- **Forecasting**: Multi-step ahead point forecasts with bootstrap confidence intervals

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

```@setup var
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
```

## Quick Start

**Recipe 1: Estimate VAR(p)**

```@example var
model = estimate_var(Y, 4)
report(model)
```

**Recipe 2: Lag selection**

```@example var
# Select lag order minimizing BIC (default)
p_bic = select_lag_order(Y, 13)

# Select via AIC
p_aic = select_lag_order(Y, 13; criterion=:aic)

model = estimate_var(Y, p_bic)
report(model)
```

**Recipe 3: Cholesky IRF**

```@example var
model = estimate_var(Y, 4)

# Cholesky IRF with bootstrap confidence intervals
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
```

```julia
plot_result(result)
```

**Recipe 4: Sign restrictions**

```@example var
model = estimate_var(Y, 4)

# Contractionary monetary shock: FFR rises, INDPRO and CPI fall on impact
check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0 && irf[1, 2, 3] < 0
result = irf(model, 20; method=:sign, check_func=check)
```

```julia
plot_result(result)
```

**Recipe 5: Arias identification**

```@example var
model = estimate_var(Y, 4)

# Zero + sign restrictions on the monetary policy shock (shock 3)
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(1, 3; horizon=0)],       # No impact on INDPRO on impact
    signs = [sign_restriction(3, 3, :positive),          # FFR rises
             sign_restriction(2, 3, :negative; horizon=1)] # CPI falls at h=1
)
result = identify_arias(model, restrictions, 20; n_draws=500)
result
```

**Recipe 6: Uhlig identification**

```@example var
model = estimate_var(Y, 4)

# Mountford-Uhlig penalty function: one optimal rotation
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(3, 1; horizon=0)],   # Fiscal shock has no impact on FFR
    signs = [sign_restriction(1, 1, :positive),     # Fiscal shock raises INDPRO
             sign_restriction(3, 3, :positive)]     # Monetary shock raises FFR
)
result = identify_uhlig(model, restrictions, 20)
result
```

---

## Reduced-Form VAR

### The Model

A **VAR(p)** model for an ``n``-dimensional vector of endogenous variables ``y_t`` is:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
- ``c`` is the ``n \times 1`` vector of intercepts
- ``A_i`` is the ``n \times n`` coefficient matrix for lag ``i = 1, \ldots, p``
- ``u_t`` is the ``n \times 1`` vector of reduced-form innovations with ``E[u_t] = 0`` and ``E[u_t u_t'] = \Sigma``

### OLS Estimation

Stack the observations into matrices. Let ``T`` denote the total sample size and define the effective sample as ``T_{\text{eff}} = T - p`` observations after accounting for lags:

```math
Y = \begin{bmatrix} y_{p+1}' \\ y_{p+2}' \\ \vdots \\ y_T' \end{bmatrix}_{T_{\text{eff}} \times n}, \quad
X = \begin{bmatrix} 1 & y_p' & y_{p-1}' & \cdots & y_1' \\
1 & y_{p+1}' & y_p' & \cdots & y_2' \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & y_{T-1}' & y_{T-2}' & \cdots & y_{T-p}' \end{bmatrix}_{T_{\text{eff}} \times (1+np)}
```

where:
- ``Y`` is the ``T_{\text{eff}} \times n`` matrix of dependent variables
- ``X`` is the ``T_{\text{eff}} \times k`` matrix of regressors with ``k = 1 + np``

The compact form ``Y = XB + U`` yields the OLS estimator:

```math
\hat{B} = (X'X)^{-1} X'Y
```

where:
- ``\hat{B}`` is the ``k \times n`` coefficient matrix ``[c, A_1, \ldots, A_p]'``

The residual covariance matrix is:

```math
\hat{\Sigma} = \frac{1}{T_{\text{eff}} - k} \hat{U}'\hat{U}
```

where:
- ``\hat{U} = Y - X\hat{B}`` is the ``T_{\text{eff}} \times n`` residual matrix
- ``k = 1 + np`` is the number of regressors per equation

```@example var
model = estimate_var(Y, 4; varnames=["INDPRO", "CPI", "FFR"])
report(model)
```

The `report` output displays the VAR specification (number of variables, lags, observations) alongside the AIC, BIC, and HQIC values. The coefficient matrix `model.B` stores the intercept in row 1, followed by ``A_1, A_2, \ldots, A_p`` stacked vertically. To extract lag-``i`` coefficients for an ``n``-variable system: `A_i = model.B[(i-1)*n+2 : i*n+1, :]`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `check_stability` | `Bool` | `true` | Warn if estimated VAR is non-stationary |
| `varnames` | `Vector{String}` | `nothing` | Variable display names (default: `y1`, `y2`, ...) |

### Return Value

`estimate_var` returns a `VARModel{T}` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original ``T \times n`` data matrix |
| `p` | `Int` | Number of lags |
| `B` | `Matrix{T}` | ``(1+np) \times n`` coefficient matrix ``[c, A_1, \ldots, A_p]'`` |
| `U` | `Matrix{T}` | ``T_{\text{eff}} \times n`` residual matrix |
| `Sigma` | `Matrix{T}` | ``n \times n`` residual covariance matrix |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `hqic` | `T` | Hannan-Quinn Information Criterion |
| `varnames` | `Vector{String}` | Variable display names |

---

## Stability and Lag Selection

### Companion Form and Stability

A VAR(p) is **stable** (stationary) if all eigenvalues of the companion matrix ``F`` lie inside the unit circle. The companion form rewrites the VAR(p) as a VAR(1) in the ``np``-dimensional state vector:

```math
F = \begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_n & 0 & \cdots & 0 & 0 \\
0 & I_n & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_n & 0
\end{bmatrix}_{np \times np}
```

where:
- ``A_i`` is the ``n \times n`` VAR coefficient matrix for lag ``i``
- ``I_n`` is the ``n \times n`` identity matrix
- ``F`` is the ``np \times np`` companion matrix

The stability condition requires ``|\lambda_i| < 1`` for all eigenvalues ``\lambda_i`` of ``F``. The function `is_stationary` checks this condition and returns the companion matrix eigenvalues:

```@example var
model = estimate_var(Y, 4)
stab = is_stationary(model)
stab
```

The `max_modulus` field reports the largest eigenvalue modulus. A value below 1.0 confirms stationarity; values near 1.0 indicate near-unit-root behavior suggesting the system may require differencing or a VECM specification.

### Information Criteria

The optimal lag length minimizes an information criterion that balances fit against model complexity. For a Gaussian VAR:

```math
\text{AIC}(p) = \log|\hat{\Sigma}| + \frac{2}{T}(n^2 p + n)
```

```math
\text{BIC}(p) = \log|\hat{\Sigma}| + \frac{\log T}{T}(n^2 p + n)
```

```math
\text{HQ}(p) = \log|\hat{\Sigma}| + \frac{2 \log(\log T)}{T}(n^2 p + n)
```

where:
- ``\hat{\Sigma}`` is the ML residual covariance at lag order ``p``
- ``T`` is the effective sample size
- ``n`` is the number of endogenous variables
- ``n^2 p + n`` is the total number of free parameters (``n^2`` per lag plus ``n`` intercepts)

AIC tends to overfit in finite samples; BIC is consistent (selects the true order with probability approaching 1 as ``T \to \infty``); HQIC offers an intermediate trade-off.

```@example var
# BIC-optimal lag order (default)
p_bic = select_lag_order(Y, 13)

# AIC-optimal lag order
p_aic = select_lag_order(Y, 13; criterion=:aic)

model = estimate_var(Y, p_bic)
report(model)
```

The function `select_lag_order` evaluates all lag orders from 1 to `max_p` and returns the integer lag order minimizing the selected criterion. The BIC default provides the most parsimonious specification and is the standard choice for structural analysis (Lütkepohl 2005).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `criterion` | `Symbol` | `:bic` | Information criterion: `:aic`, `:bic`, or `:hqic` |

---

## Structural VAR and Identification

The reduced-form residuals ``u_t`` are linear combinations of orthogonal structural shocks ``\varepsilon_t``:

```math
u_t = B_0 \varepsilon_t
```

where:
- ``B_0`` is the ``n \times n`` contemporaneous impact matrix
- ``\varepsilon_t`` are structural shocks with ``E[\varepsilon_t \varepsilon_t'] = I_n``

The identifying restriction ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns, leaving ``n(n-1)/2`` free parameters. Additional restrictions are required to achieve **exact identification**. The package provides six identification strategies.

### Cholesky (Recursive)

The Cholesky decomposition imposes a lower triangular structure on ``B_0``:

```math
B_0 = \text{chol}(\Sigma)
```

where:
- ``B_0`` is lower triangular, implying variable ``i`` responds contemporaneously only to shocks ``1, \ldots, i``

The ordering reflects economic assumptions about the speed of adjustment. Variables ordered first respond only to their own shocks on impact. In the standard monetary VAR ordering [INDPRO, CPI, FFR], the federal funds rate shock (shock 3) has no contemporaneous effect on output or prices, consistent with the information and implementation lags in monetary policy transmission (Christiano, Eichenbaum & Evans 1999).

```@example var
model = estimate_var(Y, 4)

# Cholesky IRF with bootstrap 90% CI
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50, conf_level=0.90)
```

```julia
plot_result(result)
```

### Sign Restrictions

Sign restrictions identify structural shocks by constraining the signs of impulse responses at selected horizons, following Rubio-Ramírez, Waggoner & Zha (2010). The algorithm draws random orthogonal matrices ``Q`` from the Haar measure and retains only those producing IRFs consistent with the sign constraints:

1. Compute the Cholesky factor: ``P = \text{chol}(\Sigma)``
2. Draw ``Q`` uniformly from ``O(n)`` via QR decomposition of a random matrix
3. Compute the candidate impact matrix: ``B_0 = PQ``
4. Compute IRFs ``\Theta_0 = B_0, \Theta_1, \ldots`` from the candidate ``B_0``
5. Accept if all sign conditions hold; otherwise discard and repeat

!!! note "Technical Note"
    With `store_all=true`, `identify_sign` returns a `SignIdentifiedSet` containing all accepted rotation matrices and their IRFs, enabling characterization of the full identified set (Baumeister & Hamilton 2015). Use `irf_bounds` and `irf_median` to summarize the identified set.

```@example var
model = estimate_var(Y, 4)

# Contractionary monetary shock: FFR rises, INDPRO and CPI fall
check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0 && irf[1, 2, 3] < 0

# Full identified set
id_set = identify_sign(model, 20, check; max_draws=5000, store_all=true)
id_set

# Pointwise median and 68% credible bands
med = irf_median(id_set)
lower, upper = irf_bounds(id_set; quantiles=[0.16, 0.84])
```

The acceptance rate indicates what fraction of random draws satisfy all sign conditions simultaneously. Rates below 1% suggest the restrictions may be overly stringent or nearly contradictory.

### Narrative Restrictions

Narrative restrictions augment sign restrictions with historical information about specific shocks at particular dates, following Antolín-Díaz & Rubio-Ramírez (2018). Two types of narrative constraints are supported:

1. **Shock sign narrative**: at date ``t^*``, structural shock ``j`` was positive (or negative)
2. **Shock contribution narrative**: at date ``t^*``, shock ``j`` was the dominant driver of variable ``i``

```@example var
model = estimate_var(Y, 4)

# Sign restrictions on impact
sign_check = irf -> irf[1, 3, 3] > 0 && irf[1, 1, 3] < 0

# Narrative: monetary shock was positive at observation 20
narrative_check = shocks -> shocks[20, 3] > 0

Q, irfs, shocks = identify_narrative(model, 20, sign_check, narrative_check; max_draws=5000)
```

The algorithm first filters for sign-satisfying rotations, then checks whether the recovered structural shocks ``\varepsilon = B_0^{-1} u`` satisfy the narrative conditions. This sequential filtering sharply reduces the identified set.

### Long-Run (Blanchard-Quah)

Long-run restrictions constrain the cumulative effect of structural shocks on selected variables. The long-run impact matrix is:

```math
C(1) = (I_n - A_1 - A_2 - \cdots - A_p)^{-1} B_0
```

where:
- ``C(1)`` is the ``n \times n`` long-run cumulative response matrix
- ``A(1) = A_1 + A_2 + \cdots + A_p`` is the sum of VAR coefficient matrices

Blanchard & Quah (1989) impose that ``C(1)`` is lower triangular, so that shocks ordered later have zero long-run effect on variables ordered earlier. The typical application restricts demand shocks to have no long-run effect on output, identifying supply-driven long-run fluctuations.

```@example var
model = estimate_var(Y, 4)
result = irf(model, 40; method=:long_run)
```

```julia
plot_result(result)
```

### Arias et al. (2018) Zero + Sign Restrictions

When sign restrictions alone are insufficient, zero restrictions on specific impulse responses can be imposed alongside sign constraints. Arias, Rubio-Ramírez & Waggoner (2018) develop an algorithm that draws rotation matrices ``Q`` uniformly over the set satisfying zero restrictions, then filters for sign satisfaction. Importance weights correct for non-uniform sampling induced by the zero-restriction constraint manifold.

The algorithm constructs ``Q`` column-by-column via QR decomposition in the null space of the zero restriction matrix, then checks sign restrictions on the candidate IRF ``\Theta_h = \Phi_h L Q``.

| Type | Function | Description |
|------|----------|-------------|
| Zero | `zero_restriction(var, shock; horizon=0)` | Variable `var` does not respond to `shock` at `horizon` |
| Sign | `sign_restriction(var, shock, :positive; horizon=0)` | Response has required sign at `horizon` |

```@example var
model = estimate_var(Y, 4)

# Monetary policy shock (shock 3):
# Zero: INDPRO does not respond on impact
# Sign: FFR rises on impact, CPI falls at h=1
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(1, 3; horizon=0)],
    signs = [sign_restriction(3, 3, :positive),
             sign_restriction(2, 3, :negative; horizon=1)]
)

result = identify_arias(model, restrictions, 20; n_draws=1000)
result

# Weighted IRF percentiles (importance-weight-corrected)
pct = irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
```

The acceptance rate reports the fraction of draws satisfying all restrictions. Low rates (below 1%) suggest overly stringent restrictions. The importance weights correct for non-uniform sampling --- the weighted percentiles provide correctly calibrated credible intervals.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_draws` | `Int` | `1000` | Target number of accepted draws |
| `n_rotations` | `Int` | `1000` | Maximum attempts per target draw |
| `compute_weights` | `Bool` | `true` | Compute importance weights (set `false` for faster exploratory analysis) |

**AriasSVARResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{\text{draws}} \times H \times n \times n`` IRF draws |
| `weights` | `Vector{T}` | Importance weights (normalized to sum to 1) |
| `acceptance_rate` | `T` | Fraction of draws satisfying all restrictions |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |

### Mountford-Uhlig (2009) Penalty Function

When a single best rotation is preferred over a distribution of draws, Mountford & Uhlig (2009) provide a penalty function approach. Zero restrictions are enforced exactly via null-space projection; sign restrictions are encouraged through a penalty function minimized with two-phase Nelder-Mead optimization.

The penalty for each sign restriction ``s`` is:

```math
\text{penalty} = -\sum_{s} w_s \cdot \text{sign}_s \cdot \frac{\text{IRF}_s}{\sigma_s}
```

where:
- ``w_s = 100`` if the sign restriction is satisfied, ``w_s = 1`` if violated
- ``\text{sign}_s \in \{+1, -1\}`` is the required sign direction
- ``\text{IRF}_s`` is the impulse response value at the restricted horizon
- ``\sigma_s`` is the standard deviation of the response variable (normalization)

!!! note "When to use Uhlig vs Arias"
    Use `identify_uhlig` when a single point-identified rotation is needed --- for example, as a starting point for policy analysis. Use `identify_arias` when the full identified set is required for inference with credible intervals.

```@example var
model = estimate_var(Y, 4)

# Fiscal vs monetary separation
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(3, 1; horizon=0)],   # Fiscal shock has no impact on FFR
    signs = [sign_restriction(1, 1, :positive),     # Fiscal shock raises INDPRO
             sign_restriction(3, 3, :positive)]     # Monetary shock raises FFR
)

result = identify_uhlig(model, restrictions, 20)
result
```

The `converged` field indicates whether all sign restrictions are satisfied at the optimum. A `false` value means the optimizer found a local minimum where some sign conditions are violated --- increasing `n_starts` or relaxing restrictions may help.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_starts` | `Int` | `50` | Random starting points for coarse search |
| `n_refine` | `Int` | `10` | Top candidates refined in second phase |
| `max_iter_coarse` | `Int` | `500` | Maximum Nelder-Mead iterations (coarse phase) |
| `max_iter_fine` | `Int` | `2000` | Maximum iterations (refinement phase) |

**UhligSVARResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Optimal rotation matrix |
| `irf` | `Array{T,3}` | ``H \times n \times n`` impulse responses |
| `penalty` | `T` | Total penalty at optimum (more negative = better) |
| `shock_penalties` | `Vector{T}` | Per-shock penalty values |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |
| `converged` | `Bool` | Whether all sign restrictions are satisfied |

---

## Covariance Estimation

### Newey-West HAC Estimator

For robust inference in the presence of heteroscedasticity and autocorrelation, the Newey-West (1987, 1994) estimator computes a heteroscedasticity and autocorrelation consistent (HAC) covariance matrix:

```math
\hat{V}_{\text{NW}} = (X'X)^{-1} \hat{S} (X'X)^{-1}
```

where:
- ``\hat{V}_{\text{NW}}`` is the HAC covariance matrix of the coefficient estimator
- ``\hat{S}`` is the long-run covariance estimator

The long-run covariance ``\hat{S}`` is:

```math
\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')
```

where:
- ``\hat{\Gamma}_j = \sum_{t=j+1}^{T} \hat{u}_t \hat{u}_{t-j}' x_t x_{t-j}'`` is the ``j``-th order autocovariance
- ``w_j`` is the kernel weight at lag ``j``
- ``m`` is the bandwidth (truncation parameter)

### Kernel Functions

The weight function ``w_j`` depends on the kernel choice:

**Bartlett** (default):

```math
w_j = 1 - \frac{j}{m+1}
```

**Parzen**:

```math
w_j = \begin{cases}
1 - 6x^2 + 6|x|^3 & |x| \leq 0.5 \\
2(1-|x|)^3 & 0.5 < |x| \leq 1
\end{cases}
```

where:
- ``x = j/(m+1)``

**Quadratic spectral** (Andrews 1991):

```math
w_j = \frac{25}{12\pi^2 x^2} \left( \frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5) \right)
```

where:
- ``x = j/(m+1)``

### Automatic Bandwidth Selection

Newey & West (1994) provide a data-driven bandwidth:

```math
m^* = 1.1447 \left( \hat{\alpha} \cdot T \right)^{1/3}
```

where:
- ``\hat{\alpha} = 4\hat{\rho}^2 / (1-\hat{\rho})^4`` is estimated from AR(1) fits to the residuals
- ``T`` is the sample size

### White Heteroscedasticity-Robust Estimator

When errors are heteroscedastic but serially uncorrelated, the White (1980) HC0 estimator provides consistent standard errors without bandwidth selection:

```math
\hat{V}_{W} = (X'X)^{-1} \left( \sum_{t=1}^{T} \hat{u}_t^2 x_t x_t' \right) (X'X)^{-1}
```

where:
- ``\hat{u}_t`` is the OLS residual at time ``t``
- ``x_t`` is the ``k \times 1`` regressor vector at time ``t``

### Driscoll-Kraay Panel-Robust Estimator

For panel data with both cross-sectional and temporal dependence, the Driscoll & Kraay (1998) estimator applies HAC estimation to the cross-sectional averages of moment conditions. This produces standard errors robust to heteroscedasticity, serial correlation, and cross-sectional dependence.

```@example var
# Construct VAR design matrices
Y_eff, X = construct_var_matrices(Y, 2)
residuals = Y_eff - X * ((X'X) \ (X'Y_eff))

# Newey-West HAC (Bartlett kernel, automatic bandwidth)
V_nw = newey_west(X, residuals; bandwidth=0, kernel=:bartlett)

# White heteroscedasticity-robust (HC0)
V_w = white_vcov(X, residuals)

# Automatic bandwidth selection
bw = optimal_bandwidth_nw(residuals)
```

The Newey-West estimator is the standard choice for time series with heteroscedastic and serially correlated errors --- the default for VAR and LP applications. The White estimator is simpler but inconsistent when errors are autocorrelated. The Driscoll-Kraay estimator extends HAC to panel settings where cross-sectional units may be correlated.

---

## Forecasting

The VAR generates multi-step ahead forecasts by iterating the estimated recursion forward from the last ``p`` observations:

```math
\hat{y}_{T+h} = \hat{c} + \hat{A}_1 \hat{y}_{T+h-1} + \cdots + \hat{A}_p \hat{y}_{T+h-p}
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step ahead point forecast
- ``\hat{y}_{T+j}`` for ``j \leq 0`` uses the observed data

Bootstrap confidence intervals resample the residuals and simulate forecast paths to construct empirical prediction intervals.

```@example var
model = estimate_var(Y, 4; varnames=["INDPRO", "CPI", "FFR"])
fc = forecast(model, 12; ci_method=:bootstrap, reps=50, conf_level=0.95)
fc
```

The forecast object displays per-variable tables with point forecasts and confidence interval bounds at each horizon. The bootstrap intervals account for both parameter uncertainty and future shock uncertainty.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:bootstrap` | Confidence interval method: `:bootstrap` or `:none` |
| `reps` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for intervals |

---

## Innovation Accounting and Bayesian VAR

For detailed coverage of impulse response functions, forecast error variance decomposition, and historical decomposition, see the dedicated [Innovation Accounting](innovation_accounting.md) page. For Bayesian VAR estimation with Minnesota priors, conjugate NIW sampling, and hyperparameter optimization, see [Bayesian VAR](bayesian.md).

---

## Complete Example

This example demonstrates an end-to-end VAR workflow from data loading through structural analysis using FRED-MD monetary policy variables.

```@example var
# Step 1: Select lag order
p_opt = select_lag_order(Y, 13)

# Step 2: Estimate VAR
model = estimate_var(Y, p_opt; varnames=["INDPRO", "CPI", "FFR"])
report(model)

# Step 3: Check stability
stab = is_stationary(model)
stab

# Step 4: Cholesky IRF with bootstrap CI
# Ordering: [INDPRO, CPI, FFR] — monetary policy shock is shock 3
result = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50)
```

```julia
plot_result(result)
```

```@example var
# Step 5: FEVD
decomp = fevd(model, 20)
decomp

# Step 6: Historical decomposition
hd = historical_decomposition(model, size(model.U, 1))
verify_decomposition(hd)

# Step 7: Forecast
fc = forecast(model, 12)
fc
```

The BIC selects a parsimonious lag order for the 3-variable system. The Cholesky ordering [INDPRO, CPI, FFR] implements the standard recursive identification of Christiano, Eichenbaum & Evans (1999): a monetary policy shock (shock 3) raises the federal funds rate on impact while output and prices respond with a lag. The FEVD reveals the fraction of industrial production forecast error variance attributable to monetary shocks at each horizon. The historical decomposition identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` holds exactly up to numerical precision, verified by `verify_decomposition`.

---

## Common Pitfalls

1. **Variable ordering matters for Cholesky identification.** The Cholesky decomposition imposes a recursive causal structure where variable ``i`` responds contemporaneously only to shocks ``1, \ldots, i``. Reordering the columns of ``Y`` changes the economic interpretation of the structural shocks. The standard monetary VAR ordering places slow-moving variables first (output, prices) and the policy instrument last.

2. **Non-stationary VAR produces unreliable inference.** If `is_stationary` returns `false`, the companion matrix has unit-root eigenvalues and the asymptotic distribution theory underlying OLS standard errors and bootstrap confidence intervals is invalid. Consider differencing the data, applying the appropriate transformation codes via `apply_tcode`, or estimating a [VECM](vecm.md) for cointegrated systems.

3. **Too many lags exhaust degrees of freedom.** Each additional lag adds ``n^2`` parameters. For an ``n = 7`` variable system, each lag costs 49 parameters. With moderate sample sizes (``T < 200``), overfitting degrades forecast accuracy and inflates IRF confidence intervals. Use `select_lag_order` with BIC for parsimony.

4. **Low acceptance rate for sign restrictions.** When `identify_sign` or `identify_arias` reports an acceptance rate below 1%, the imposed sign conditions are difficult to satisfy jointly. This may indicate contradictory economic restrictions or an overidentified specification. Relaxing some conditions (e.g., restricting only impact responses rather than multiple horizons) typically improves acceptance.

5. **Uhlig penalty function finds local minima.** The `identify_uhlig` optimizer uses multi-start Nelder-Mead, but non-convexity of the penalty landscape means the solution depends on initial conditions. If `converged` is `false`, increase `n_starts` or verify that the sign restrictions are economically coherent.

---

## References

- Andrews, D. W. K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation.
  *Econometrica*, 59(3), 817-858. [DOI](https://doi.org/10.2307/2938229)

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802-2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications.
  *Econometrica*, 86(2), 685-720. [DOI](https://doi.org/10.3982/ECTA14468)

- Baumeister, C., & Hamilton, J. D. (2015). Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information.
  *Econometrica*, 83(5), 1963-1999. [DOI](https://doi.org/10.3982/ECTA12356)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655-673. [DOI](https://doi.org/10.3386/w2737)

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End?
  In *Handbook of Macroeconomics*, Vol. 1, edited by J. B. Taylor & M. Woodford, 65-148. Amsterdam: Elsevier. [DOI](https://doi.org/10.1016/S1574-0048(99)01005-8)

- Driscoll, J. C., & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data.
  *Review of Economics and Statistics*, 80(4), 549-560. [DOI](https://doi.org/10.1162/003465398557825)

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Mountford, A., & Uhlig, H. (2009). What Are the Effects of Fiscal Policy Shocks?
  *Journal of Applied Econometrics*, 24(6), 960-992. [DOI](https://doi.org/10.1002/jae.1079)

- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica*, 55(3), 703-708. [DOI](https://doi.org/10.2307/1913610)

- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
  *Review of Economic Studies*, 61(4), 631-653. [DOI](https://doi.org/10.2307/2297912)

- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference.
  *Review of Economic Studies*, 77(2), 665-696. [DOI](https://doi.org/10.1111/j.1467-937X.2009.00578.x)

- Sims, C. A. (1980). Macroeconomics and Reality.
  *Econometrica*, 48(1), 1-48. [DOI](https://doi.org/10.2307/1912017)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381-419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)

- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.
  *Econometrica*, 48(4), 817-838. [DOI](https://doi.org/10.2307/1912934)
