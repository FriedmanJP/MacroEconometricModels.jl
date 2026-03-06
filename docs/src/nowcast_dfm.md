# [DFM Nowcasting](@id nowcast_dfm_page)

The Dynamic Factor Model (DFM) extracts a small number of latent factors from a large panel of indicators and uses them to produce real-time nowcasts of target variables. This is the workhorse model used by the ECB, the Federal Reserve Bank of New York, and many other central banks for real-time macroeconomic monitoring. The implementation follows Banbura & Modugno (2014) with Mariano & Murasawa (2003) temporal aggregation for mixed-frequency data.

For an overview of all nowcasting methods and method comparison, see [Nowcasting](@ref). For BVAR-based nowcasting, see [BVAR Nowcasting](@ref nowcast_bvar_page).

## Quick Start

**Recipe 1: Basic DFM nowcast**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# Two-factor DFM with AR(1) idiosyncratic components
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
report(dfm)
```

**Recipe 2: DFM with IID idiosyncratic errors**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# IID idiosyncratic errors simplify the state space
dfm_iid = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:iid)
report(dfm_iid)
```

**Recipe 3: DFM with block structure**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# Block structure: real activity (cols 1-2), nominal (cols 3-4), target (col 5)
blocks = [1 0; 1 0; 0 1; 0 1; 1 1]
dfm_block = nowcast_dfm(Y, nM, nQ; r=1, p=1, blocks=blocks)
report(dfm_block)
```

**Recipe 4: DFM forecast**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# 6-step ahead forecast for the quarterly target variable
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)
fc = forecast(dfm, 6; target_var=5)
```

**Recipe 5: DFM with TimeSeriesData dispatch**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# TimeSeriesData dispatch works identically to raw matrices
ts = TimeSeriesData(Y; varnames=["INDPRO","UNRATE","CPI","M2","FEDFUNDS"], frequency=Monthly)
dfm = nowcast_dfm(ts, nM, nQ; r=2, p=1)
report(dfm)
```

---

## Model Specification

The DFM represents a large panel of ``N`` observed indicators as linear combinations of ``r`` unobserved common factors plus variable-specific idiosyncratic components.

The **observation equation** links observed data to latent factors:

```math
x_{i,t} = \lambda_i' f_t + e_{i,t}
```

where:
- ``x_{i,t}`` is the ``i``-th observed variable at time ``t``
- ``f_t \in \mathbb{R}^r`` is the vector of latent common factors
- ``\lambda_i`` is the ``r \times 1`` loading vector for variable ``i``
- ``e_{i,t}`` is the idiosyncratic component (`:ar1` or `:iid`)

The **factor dynamics** follow a VAR(p):

```math
f_t = A_1 f_{t-1} + \cdots + A_p f_{t-p} + u_t, \quad u_t \sim N(0, Q)
```

where:
- ``A_1, \ldots, A_p`` are ``r \times r`` autoregressive coefficient matrices
- ``Q`` is the ``r \times r`` state innovation covariance matrix

### Temporal Aggregation

Quarterly variables require special treatment because they aggregate over three monthly periods. The Mariano & Murasawa (2003) temporal aggregation maps quarterly flow variables to monthly factors using the **triangular weights** ``[1, 2, 3, 2, 1]``:

```math
x_{i,t}^Q = \lambda_i' \big( f_t + 2 f_{t-1} + 3 f_{t-2} + 2 f_{t-3} + f_{t-4} \big) + e_{i,t}^Q
```

The state vector is augmented to include 5 lags of the factors, and the observation equation for quarterly variables sets ``C[i, k \cdot r + c] = w_k \cdot \lambda_{i,c}`` for lags ``k = 0, \ldots, 4`` with weights ``w = [1, 2, 3, 2, 1]``. This augmentation ensures that quarterly observations at quarter-end months inform factor estimation at all constituent months.

!!! note "Technical Note"
    The effective lag order in the state space is ``p_{\text{eff}} = \max(p, 5)`` when quarterly variables are present. The companion form stacks all ``p_{\text{eff}}`` lags of the ``r \times n_{\text{blocks}}`` factor vector, plus idiosyncratic states for AR(1) monthly components and 5-state shift registers for quarterly idiosyncratic temporal aggregation.

---

## Estimation

The EM algorithm (Banbura & Modugno 2014) estimates all state-space parameters jointly, handling arbitrary patterns of missing data:

1. **E-step**: The Kalman smoother runs on the current state-space parameters with NaN-aware observation equations. At each time step, only the rows of the observation equation corresponding to non-missing variables are used, producing smoothed state means ``E[z_t | \mathcal{I}_T]`` and covariances ``V[z_t | \mathcal{I}_T]``.

2. **M-step**: Sufficient statistics from the smoother update all parameters:
   - Factor VAR coefficients ``A`` via OLS on smoothed states
   - Observation loadings ``C`` via per-variable OLS (with Mariano-Murasawa constraints for quarterly variables)
   - State noise ``Q`` and observation noise ``R`` from residual covariances
   - Idiosyncratic AR(1) coefficients from per-variable autocovariances
   - Initial state mean and covariance

The algorithm iterates until the relative change in log-likelihood falls below `thresh`:

```math
\frac{|\ell^{(k)} - \ell^{(k-1)}|}{|\ell^{(k-1)}|} < \text{thresh}
```

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# Estimate with tighter convergence and more iterations
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=200, thresh=1e-6)
println("Converged in $(dfm.n_iter) iterations, log-likelihood: $(round(dfm.loglik, digits=2))")
```

The log-likelihood value measures the overall fit of the estimated factors to the observed data. Higher values indicate better fit, and the convergence path should be monotonically increasing (a property guaranteed by the EM algorithm).

---

## Block Structure

The `blocks` argument accepts an ``N \times B`` binary matrix specifying which variables load on which block factors. Entry ``(i, b) = 1`` indicates that variable ``i`` loads on block ``b``. This structure is useful when variables fall into natural groups --- for example, real activity indicators, price indices, and financial variables.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# Two blocks: real activity (INDPRO, UNRATE) and nominal (CPI, M2)
# FEDFUNDS loads on both blocks
blocks = [1 0;    # INDPRO → block 1
          1 0;    # UNRATE → block 1
          0 1;    # CPI → block 2
          0 1;    # M2 → block 2
          1 1]    # FEDFUNDS → both blocks
dfm_block = nowcast_dfm(Y, nM, nQ; r=1, p=1, blocks=blocks)
report(dfm_block)
```

When `blocks=nothing` (default), all variables load on a single global factor block. Block-restricted models are appropriate when economic theory suggests distinct latent drivers for different variable groups, and a variable loading on multiple blocks captures cross-group comovement.

---

## Forecasting

The `forecast` function projects the state vector forward using the estimated transition equation ``z_{t+h} = A^h z_t`` and maps the forecast states back to observables via the observation equation.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
N = nM + nQ
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)

# Forecast all variables 6 months ahead
fc_all = forecast(dfm, 6)

# Forecast a specific target variable
fc_target = forecast(dfm, 6; target_var=N)
```

The forecast horizon `h` counts monthly steps. For quarterly targets, a 3-step forecast corresponds to one quarter ahead, and a 6-step forecast corresponds to two quarters ahead. The `nowcast` function uses 3-step state projection internally to produce the next-quarter forecast.

---

## Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `r` | `Int` | `2` | Number of latent factors |
| `p` | `Int` | `1` | VAR lags in factor dynamics |
| `idio` | `Symbol` | `:ar1` | Idiosyncratic dynamics (`:ar1` or `:iid`) |
| `blocks` | `Matrix{Int}` or `nothing` | `nothing` | ``N \times B`` binary block structure matrix |
| `max_iter` | `Int` | `100` | Maximum EM iterations |
| `thresh` | `Real` | ``10^{-4}`` | Convergence threshold for relative log-likelihood change |

---

## NowcastDFM Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed data with all NaN values filled |
| `F` | `Matrix{T}` | Smoothed factors (``T \times r \cdot p``) |
| `C` | `Matrix{T}` | Observation loadings (``N \times \text{state\_dim}``) |
| `A` | `Matrix{T}` | State transition matrix |
| `Q` | `Matrix{T}` | State innovation covariance |
| `R` | `Matrix{T}` | Observation noise covariance (diagonal) |
| `loglik` | `T` | Log-likelihood at convergence |
| `n_iter` | `Int` | Number of EM iterations used |
| `r` | `Int` | Number of factors |
| `p` | `Int` | VAR lags in factor dynamics |

---

## Balancing Panels

The `balance_panel` utility fills missing values in `TimeSeriesData` or `PanelData` using DFM imputation. Observed values are preserved; only NaN entries are replaced with DFM-smoothed estimates.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]

# Introduce some missing values
Y[end, 1:3] .= NaN

ts = TimeSeriesData(Y; varnames=["INDPRO","UNRATE","CPI","M2","FEDFUNDS"], frequency=Monthly)
ts_balanced = balance_panel(ts; r=2, p=1, method=:dfm)
```

The function uses `nowcast_dfm` internally with `nQ=0` (all variables treated as monthly). The `r` and `p` arguments control the number of factors and lags in the imputation model.

---

## Complete Example

```julia
using MacroEconometricModels

# === Step 1: Prepare FRED-MD mixed-frequency panel ===
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
T_obs = size(Y, 1)
nM, nQ = 4, 1
N = nM + nQ
for t in 1:T_obs
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# === Step 2: Estimate DFM ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=100, thresh=1e-4)
report(dfm)

# === Step 3: Extract nowcast and forecast ===
result = nowcast(dfm)
println("Nowcast: ", round(result.nowcast, digits=3))
println("Forecast: ", round(result.forecast, digits=3))

# === Step 4: Multi-step forecast ===
fc = forecast(dfm, 6; target_var=N)

# === Step 5: News decomposition ===
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN    # simulate that 3 releases were not yet available

news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N)
println("Old nowcast: ", round(news.old_nowcast, digits=3))
println("New nowcast: ", round(news.new_nowcast, digits=3))
println("Total revision: ", round(news.new_nowcast - news.old_nowcast, digits=3))

# === Step 6: Balance panel for further analysis ===
ts = TimeSeriesData(Y; varnames=["INDPRO","UNRATE","CPI","M2","FEDFUNDS"], frequency=Monthly)
ts_balanced = balance_panel(ts; r=2, p=1)
```

**Interpretation.** The DFM extracts two common factors from the four monthly FRED-MD indicators (INDPRO, UNRATE, CPIAUCSL, M2SL) and the quarterly FEDFUNDS target. The EM algorithm iterates between Kalman smoothing and parameter updates until convergence. The smoothed data matrix fills all NaN entries --- both the systematic quarterly pattern and the ragged edge --- using the estimated factor structure. The news decomposition attributes the nowcast revision to individual data releases, identifying which monthly indicators drove the largest updates to the quarterly target estimate.

---

## Common Pitfalls

1. **Column ordering matters.** The first `nM` columns must be monthly variables and the last `nQ` columns must be quarterly variables. Quarterly observations appear every 3rd row with NaN elsewhere.

2. **Too many factors relative to variables.** Setting `r` close to ``N`` removes the dimension reduction benefit. A practical rule is ``r \leq N / 3``. Use `ic_criteria(X, r_max)` from the factor model module to select ``r`` via information criteria.

3. **Convergence failures.** If the EM algorithm reaches `max_iter` without converging, increase `max_iter` or relax `thresh`. Slow convergence often indicates weak factor structure or too few observations.

4. **AR(1) vs IID idiosyncratic.** The `:ar1` option adds ``n_M`` states to the state vector, increasing computational cost. Use `:iid` when idiosyncratic serial correlation is not a concern or the panel is large.

5. **Quarterly masking pattern.** Quarterly observations must appear at rows where `mod(t, 3) == 0` (i.e., every 3rd row). An incorrect masking pattern causes the Mariano-Murasawa aggregation to misalign with the data.

---

## References

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [DOI: 10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Mariano, Roberto S., and Yasutomo Murasawa. 2003. "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18 (4): 427--443. [DOI: 10.1002/jae.695](https://doi.org/10.1002/jae.695)
- Giannone, Domenico, Lucrezia Reichlin, and David Small. 2008. "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55 (4): 665--676. [DOI: 10.1016/j.jmoneco.2008.05.010](https://doi.org/10.1016/j.jmoneco.2008.05.010)
