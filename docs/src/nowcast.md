# Nowcasting

This page documents the nowcasting module in **MacroEconometricModels.jl**, implementing three state-of-the-art approaches for real-time macroeconomic prediction with mixed-frequency data and ragged edges: Dynamic Factor Model (DFM), Large Bayesian VAR (BVAR), and Bridge Equations.

## Quick Start

```julia
using MacroEconometricModels

# Load FRED-MD monthly indicators and prepare mixed-frequency panel
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]   # last 100 observations

# Convert last column (FEDFUNDS) to "quarterly" by masking non-quarter months
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN       # simulate missing latest observation (ragged edge)

# DFM nowcasting (Bańbura & Modugno 2014)
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)

# Large BVAR nowcasting (Cimadomo et al. 2022)
bvar = nowcast_bvar(Y, nM, nQ; lags=5)

# Bridge equation nowcasting (Bańbura et al. 2023)
bridge = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1)

# Extract nowcast and forecast
result = nowcast(dfm)
result.nowcast    # current-quarter estimate
result.forecast   # next-quarter forecast
```

---

## The Nowcasting Problem

Central banks and forecasters face a fundamental timing problem: key macroeconomic aggregates like GDP are released with a significant delay and at low frequency (quarterly), while a rich set of monthly and weekly indicators is available in real time. **Nowcasting** produces current-quarter estimates by exploiting the information content of timely high-frequency releases.

The key challenges are:

1. **Mixed frequencies** --- monthly indicators and quarterly targets coexist in the same model
2. **Ragged edges** --- not all series are updated simultaneously; the most recent months have missing observations for slower-release variables
3. **Large cross-sections** --- dozens to hundreds of indicators provide complementary information

```math
\underbrace{Y_t}_{\text{target (quarterly)}} = f\big(\underbrace{X_{1,t}, \ldots, X_{N,t}}_{\text{monthly indicators}}\big) + \varepsilon_t
```

where the challenge is that ``Y_t`` and some ``X_{j,t}`` are unobserved at the forecast origin.

!!! note "Data Layout Convention"
    All nowcasting functions expect a ``T \times N`` matrix where the first `nM` columns are monthly variables and the last `nQ` columns are quarterly variables. Quarterly observations appear every 3rd row (months 3, 6, 9, 12, ...) with `NaN` for non-quarter-end months.

---

## Method 1: Dynamic Factor Model (DFM)

The DFM approach extracts a small number of latent factors from a large cross-section and uses them to nowcast the target variable. This is the workhorse model used by the ECB, Federal Reserve Bank of New York, and many other institutions.

### Model Specification

The observation equation links observed data to latent factors:

```math
x_{i,t} = \lambda_i' f_t + e_{i,t}
```

The factor dynamics follow a VAR(p):

```math
f_t = A_1 f_{t-1} + \cdots + A_p f_{t-p} + u_t, \quad u_t \sim N(0, Q)
```

where ``f_t \in \mathbb{R}^r`` are the latent factors, ``\lambda_i`` are factor loadings, and ``e_{i,t}`` are idiosyncratic components.

**Quarterly temporal aggregation.** Quarterly variables are linked to the monthly factors via Mariano-Murasawa weights ``[1, 2, 3, 2, 1]``, representing the flow nature of quarterly data as a weighted average of monthly latent values.

**Estimation.** The EM algorithm alternates between:
- **E-step**: Kalman smoother with NaN-aware observation equations extracts factors
- **M-step**: Update state-space parameters (A, C, Q, R) from sufficient statistics

### Usage

```julia
dfm = nowcast_dfm(Y, nM, nQ;
    r = 2,             # number of factors
    p = 1,             # VAR lags in factor dynamics
    idio = :ar1,       # idiosyncratic dynamics (:ar1 or :iid)
    blocks = nothing,  # block structure (N × n_blocks matrix)
    max_iter = 100,    # maximum EM iterations
    thresh = 1e-4      # convergence threshold
)
```

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data (NaN filled) |
| `F` | `Matrix{T}` | Smoothed factors (T × state_dim) |
| `C` | `Matrix{T}` | Observation loadings |
| `A` | `Matrix{T}` | State transition matrix |
| `Q` | `Matrix{T}` | State innovation covariance |
| `R` | `Matrix{T}` | Observation noise covariance (diagonal) |
| `loglik` | `T` | Log-likelihood at convergence |
| `n_iter` | `Int` | EM iterations used |
| `r` | `Int` | Number of factors |
| `p` | `Int` | VAR lags |

!!! note "Technical Note: Block Structure"
    The `blocks` argument accepts an ``N \times B`` binary matrix, where entry ``(i,b) = 1`` indicates variable ``i`` loads on block ``b``. When `blocks=nothing` (default), all variables load on a single global factor. Block structures are useful when variables naturally group (e.g., real activity, prices, financial) and you want block-specific factors plus a global factor.

---

## Method 2: Large Bayesian VAR

The BVAR approach estimates a large VAR directly on the mixed-frequency data, using informative priors to handle the curse of dimensionality. This follows Giannone, Lenza, and Primiceri (2015) with extensions for mixed-frequency data.

### Model Specification

```math
y_t = c + B_1 y_{t-1} + \cdots + B_p y_{t-p} + u_t, \quad u_t \sim N(0, \Sigma)
```

**Prior structure.** The Normal-Inverse-Wishart prior implements four types of shrinkage via dummy observations:

1. **Tightness** (``\lambda``): lag-decaying overall shrinkage
2. **Cross-variable** (``\theta``): shrinks cross-variable coefficients relative to own-lag
3. **Sum-of-coefficients** (``\mu``): unit root prior (random walk for each variable)
4. **Co-persistence** (``\alpha``): common stochastic trend prior

Hyperparameters are optimized via marginal log-likelihood maximization using Nelder-Mead.

### Usage

```julia
bvar = nowcast_bvar(Y, nM, nQ;
    lags = 5,         # number of VAR lags
    thresh = 1e-6,    # optimization convergence threshold
    max_iter = 200,   # max optimization iterations
    lambda0 = 0.2,    # initial overall shrinkage
    theta0 = 1.0,     # initial cross-variable shrinkage
    miu0 = 1.0,       # initial sum-of-coefficients weight
    alpha0 = 2.0      # initial co-persistence weight
)
```

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data (NaN filled) |
| `beta` | `Matrix{T}` | Posterior mode VAR coefficients |
| `sigma` | `Matrix{T}` | Posterior mode error covariance |
| `lambda` | `T` | Optimized overall shrinkage |
| `theta` | `T` | Optimized cross-variable shrinkage |
| `miu` | `T` | Optimized sum-of-coefficients weight |
| `alpha` | `T` | Optimized co-persistence weight |
| `loglik` | `T` | Marginal log-likelihood |

---

## Method 3: Bridge Equations

Bridge equations provide a simple, transparent approach by regressing the quarterly target on aggregated monthly indicators using OLS. Multiple equations (one per pair of monthly indicators) are combined via median to produce a robust nowcast.

### Model Specification

For each pair ``(m_1, m_2)`` of monthly indicators:

```math
Y_t^Q = \beta_0 + \sum_{l=1}^{L_M} \beta_{m_1,l} X_{m_1,t-l}^Q + \sum_{l=1}^{L_M} \beta_{m_2,l} X_{m_2,t-l}^Q + \sum_{l=1}^{L_Q} \gamma_l X_t^Q + \sum_{l=1}^{L_Y} \delta_l Y_{t-l}^Q + \varepsilon_t
```

where ``X^Q`` denotes monthly data aggregated to quarterly frequency (3-month moving average). The combination across ``\binom{n_M}{2} + n_M`` equations uses the **median**, which is robust to individual equation failures.

### Usage

```julia
bridge = nowcast_bridge(Y, nM, nQ;
    lagM = 1,    # monthly indicator lags (after quarterly aggregation)
    lagQ = 1,    # quarterly indicator lags
    lagY = 1     # autoregressive lags for target
)
```

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data (NaN filled by interpolation) |
| `Y_nowcast` | `Vector{T}` | Combined nowcast (per quarter, median) |
| `Y_individual` | `Matrix{T}` | Individual equation nowcasts |
| `n_equations` | `Int` | Number of bridge equations |
| `coefficients` | `Vector{Vector{T}}` | OLS coefficients per equation |

---

## Nowcast and Forecast Extraction

### Current-Quarter Nowcast

```julia
result = nowcast(model)        # works with any AbstractNowcastModel
result.nowcast                 # current-quarter value
result.forecast                # next-quarter forecast
result.method                  # :dfm, :bvar, or :bridge
```

The `nowcast` function extracts the current-quarter estimate from the smoothed data and produces a one-quarter-ahead forecast:

- **DFM**: 3-step state evolution (one quarter = 3 months)
- **BVAR**: one-step VAR forecast from last smoothed values
- **Bridge**: median nowcast from individual equations

### Multi-Step Forecast

```julia
# DFM forecast (state-space projection)
fc = forecast(dfm, 6)                # 6-step ahead
fc = forecast(dfm, 6; target_var=10) # specific target variable

# BVAR forecast (VAR iteration)
fc = forecast(bvar, 6)
```

---

## News Decomposition

When new data releases arrive, the nowcast changes. The **news decomposition** (Banbura and Modugno 2014) attributes this revision to individual data releases, answering: *Which data releases drove the revision?*

```math
\hat{y}^{\text{new}} - \hat{y}^{\text{old}} = \underbrace{\sum_{j \in \mathcal{J}} w_j \cdot (x_j^{\text{actual}} - x_j^{\text{forecast}})}_{\text{news}} + \underbrace{\Delta_{\text{revision}}}_{\text{data revisions}} + \underbrace{\Delta_{\text{re-estimation}}}_{\text{parameter updates}}
```

### Usage

```julia
# Two data vintages (X_old has more NaN than X_new)
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN  # these releases were not available before

# Compute news decomposition
news = nowcast_news(X_new, X_old, dfm, size(Y, 1);
    target_var = size(Y, 2),   # target variable (default: last column)
    groups = nothing           # optional group assignment for aggregation
)

news.old_nowcast              # previous nowcast
news.new_nowcast              # updated nowcast
news.impact_news              # per-release impact vector
news.impact_reestimation      # residual re-estimation effect
news.group_impacts            # aggregated by variable group
news.variable_names           # release identifiers
```

!!! note "Interpretation"
    A positive `impact_news[j]` means the actual value of release ``j`` was higher than expected (given the old information set), contributing to an upward revision of the nowcast. The sum of all news impacts plus the re-estimation residual equals the total revision.

---

## Balancing Panels

The `balance_panel` utility fills missing values in `TimeSeriesData` or `PanelData` using DFM imputation:

```julia
# Fill NaN in time series data
ts = TimeSeriesData(Y; varnames=["x1","x2","x3"], frequency=Monthly)
ts_balanced = balance_panel(ts; r=2, p=1, method=:dfm)

# Fill NaN in panel data
pd_balanced = balance_panel(pd; r=2)
```

Observed values are preserved; only NaN entries are replaced with DFM-smoothed estimates.

---

## TimeSeriesData Dispatch

All nowcasting functions accept `TimeSeriesData` directly:

```julia
ts = TimeSeriesData(Y; varnames=varnames, frequency=Monthly)
dfm = nowcast_dfm(ts, nM, nQ; r=2)
bvar = nowcast_bvar(ts, nM, nQ; lags=5)
bridge = nowcast_bridge(ts, nM, nQ)
```

---

## StatsAPI Interface

| Function | DFM | BVAR | Bridge |
|----------|-----|------|--------|
| `loglikelihood(m)` | Log-likelihood at convergence | Marginal log-likelihood | --- |
| `predict(m)` | Smoothed data `X_sm` | Smoothed data `X_sm` | Smoothed data `X_sm` |
| `nobs(m)` | Number of time periods | Number of time periods | Number of time periods |

---

## Choosing a Method

| Criterion | DFM | BVAR | Bridge |
|-----------|-----|------|--------|
| **Cross-section size** | Large (50--200 variables) | Medium-large (10--50) | Small-medium (5--20) |
| **Interpretability** | Factors are latent | Direct variable coefficients | Simple OLS regressions |
| **News decomposition** | Native support | --- | --- |
| **Computational cost** | Moderate (EM iterations) | Moderate (hyperparameter optimization) | Fast (closed-form OLS) |
| **Ragged edge handling** | Kalman smoother | Kalman smoother | Interpolation |
| **Mixed frequency** | Mariano-Murasawa temporal aggregation | Kalman smoother | Quarterly aggregation |
| **Best for** | Large mixed-frequency panels | Medium panels with strong priors | Quick baseline, transparent models |

---

## Complete Example

```julia
using MacroEconometricModels

# === Step 1: Prepare FRED-MD mixed-frequency panel ===
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]   # last 100 observations
T_obs = size(Y, 1)

# Convert last column (FEDFUNDS) to "quarterly" target
nM, nQ = 4, 1
N = nM + nQ
for t in 1:T_obs
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN       # simulate missing latest observation (ragged edge)

println("Data: T=$T_obs, nM=$nM monthly, nQ=$nQ quarterly")
println("Missing: ", sum(isnan.(Y)), " / ", length(Y), " entries")

# === Step 2: Estimate all three models ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=100)
bvar = nowcast_bvar(Y, nM, nQ; lags=5)
bridge = nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1)

# === Step 3: Compare nowcasts ===
r_dfm = nowcast(dfm)
r_bvar = nowcast(bvar)
r_bridge = nowcast(bridge)

println("\nNowcast comparison (target: FEDFUNDS quarterly):")
println("  DFM:    nowcast = ", round(r_dfm.nowcast, digits=3),
        ", forecast = ", round(r_dfm.forecast, digits=3))
println("  BVAR:   nowcast = ", round(r_bvar.nowcast, digits=3),
        ", forecast = ", round(r_bvar.forecast, digits=3))
println("  Bridge: nowcast = ", round(r_bridge.nowcast, digits=3),
        ", forecast = ", round(r_bridge.forecast, digits=3))

# === Step 4: DFM forecasting ===
fc = forecast(dfm, 6; target_var=N)
println("\nDFM 6-step forecast for target variable:")
for h in 1:6
    println("  h=$h: ", round(fc[h], digits=3))
end

# === Step 5: News decomposition ===
# Simulate that INDPRO, UNRATE, CPIAUCSL were just released for the latest month
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN

news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N)
println("\nNews decomposition:")
println("  Old nowcast: ", round(news.old_nowcast, digits=3))
println("  New nowcast: ", round(news.new_nowcast, digits=3))
println("  Total revision: ", round(news.new_nowcast - news.old_nowcast, digits=3))
println("  Top release impacts:")
sorted_idx = sortperm(abs.(news.impact_news), rev=true)
for k in 1:min(3, length(sorted_idx))
    j = sorted_idx[k]
    println("    ", news.variable_names[j], ": ", round(news.impact_news[j], digits=4))
end
```

**Interpretation.** The DFM extracts common factors from the 4 monthly FRED-MD indicators (INDPRO, UNRATE, CPIAUCSL, M2SL) and the quarterly FEDFUNDS target, filling the ragged edge via the Kalman smoother. The BVAR estimates all cross-variable dynamics directly with informative priors. Bridge equations provide a simple, transparent baseline. The news decomposition shows which data releases drove the most recent nowcast revision --- for example, a surprise in industrial production or unemployment may revise the quarterly target estimate.

---

## References

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Banbura, Marta, Irina Belousova, Katalin Bodnar, and Mate Barnabas Toth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.
- Delle Chiaie, Simona, Florian Heider, Soren Hoppe, Alexander Melemenidis, Mate Barnabas Toth, and Stefan Zeugner. 2022. "Real-time Data and Advance Estimates in Nowcasting." *ECB Research Bulletin* No. 75.
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436--451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Mariano, Roberto S., and Yasutomo Murasawa. 2003. "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18 (4): 427--443. [https://doi.org/10.1002/jae.695](https://doi.org/10.1002/jae.695)
