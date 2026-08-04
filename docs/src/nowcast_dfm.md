# [DFM Nowcasting](@id nowcast_dfm_page)

The dynamic factor model (DFM) compresses a large panel of indicators into a handful of latent factors and reads the current-quarter estimate off the smoothed target series. This is the workhorse of real-time monitoring at the ECB, the Federal Reserve Bank of New York, and most other central banks (Giannone, Reichlin & Small 2008). The implementation estimates every state-space parameter by the EM algorithm of Bańbura & Modugno (2014), so arbitrary missing-data patterns — quarterly series observed one month in three, ragged edges, holes in the interior — need no pre-balancing.

For the shared data layout, the `nowcast()` interface, and the result visualizations, see [Nowcasting](@ref nowcast_page). Sibling estimators: [BVAR Nowcasting](@ref nowcast_bvar_page) and [Bridge Equations](@ref nowcast_bridge_page). To attribute a revision to individual releases, see [News Decomposition](@ref nowcast_news_page).

```@setup nc_dfm
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
nM, nQ = 4, 1
N = nM + nQ
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN
T_obs = size(Y, 1)
```

The examples run on a FRED-MD panel in the standard mixed-frequency layout: four monthly indicators (`INDPRO`, `UNRATE`, `CPIAUCSL`, `M2SL`) in the leading columns and one quarterly target (`FEDFUNDS`) in the last, observed only at quarter-end months and missing in the most recent month.

## Quick Start

**Recipe 1: Two-factor DFM with AR(1) idiosyncratic components**

```@example nc_dfm
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
report(dfm)
```

**Recipe 2: I.i.d. idiosyncratic components**

```@example nc_dfm
dfm_iid = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:iid)
report(dfm_iid)
```

**Recipe 3: Block structure — real activity versus nominal**

```@example nc_dfm
blocks = [1 0; 1 0; 0 1; 0 1; 1 1]
dfm_block = nowcast_dfm(Y, nM, nQ; r=1, p=1, blocks=blocks)
report(dfm_block)
```

**Recipe 4: Read off the nowcast and the one-quarter-ahead forecast**

```@example nc_dfm
result = nowcast(dfm)
report(result)
```

**Recipe 5: Multi-step monthly forecast path**

```@example nc_dfm
forecast(dfm, 6; target_var=N)
```

**Recipe 6: `TimeSeriesData` dispatch**

```@example nc_dfm
ts = TimeSeriesData(Y; varnames=["INDPRO", "UNRATE", "CPI", "M2", "FEDFUNDS"],
                    frequency=Monthly)
report(nowcast_dfm(ts, nM, nQ; r=2, p=1))
```

---

## Model Specification

The DFM writes each of the ``N`` observed indicators as a linear combination of ``r`` unobserved common factors plus a variable-specific idiosyncratic component. The factors carry the comovement that makes a large panel informative about a single target; the idiosyncratic terms absorb measurement error and series-specific dynamics.

The **observation equation** links the data to the latent factors:

```math
x_{i,t} = \lambda_i' f_t + e_{i,t}
```

where:
- ``x_{i,t}`` is the ``i``-th standardized indicator at time ``t``
- ``f_t \in \mathbb{R}^r`` is the vector of latent common factors
- ``\lambda_i`` is the ``r \times 1`` loading vector for variable ``i``
- ``e_{i,t}`` is the idiosyncratic component, either AR(1) (`idio=:ar1`) or white noise (`idio=:iid`)

The **factor dynamics** follow a VAR(``p``):

```math
f_t = A_1 f_{t-1} + \cdots + A_p f_{t-p} + u_t, \quad u_t \sim N(0, Q)
```

where:
- ``A_1, \ldots, A_p`` are ``r \times r`` autoregressive coefficient matrices
- ``Q`` is the ``r \times r`` state innovation covariance matrix

Estimation runs on standardized data. `nowcast_dfm` stores each column mean in `Mx` and each column standard deviation in `Wx`, both computed over the non-missing entries, and restores the original units when it writes `X_sm`.

### Temporal Aggregation

Quarterly variables need special treatment because a quarterly growth rate aggregates three monthly ones. Mariano & Murasawa (2003) show that for a variable in log-differences the correct monthly approximation uses the **triangular weights** ``[1, 2, 3, 2, 1]``:

```math
x_{i,t}^Q = \lambda_i' \big( f_t + 2 f_{t-1} + 3 f_{t-2} + 2 f_{t-3} + f_{t-4} \big) + e_{i,t}^Q
```

where:
- ``x_{i,t}^Q`` is the quarterly variable observed at quarter-end month ``t``
- ``f_{t-k}`` are the factors at the constituent and preceding months
- ``e_{i,t}^Q`` is the quarterly idiosyncratic component, aggregated with the same weights

The state vector is augmented with five lags of the factor vector, and the observation row for a quarterly variable is written as ``C[i, k \cdot n_f + c] = w_k \lambda_{i,c}`` for ``k = 0, \ldots, 4`` with ``w = [1, 2, 3, 2, 1]``. The M-step re-imposes this structure at every iteration, so the loadings stay Mariano-Murasawa consistent instead of drifting to unrestricted values. A quarterly observation therefore informs the factor estimate at all three months of its quarter, which is what lets a monthly release move the current-quarter number.

!!! note "State dimension"
    With quarterly variables present the effective lag order is ``p_{\text{eff}} = \max(p, 5)``. The state stacks ``p_{\text{eff}}`` lags of the ``r \cdot n_{\text{blocks}}`` factors, then one AR(1) state per monthly variable when `idio=:ar1`, then a five-state shift register per quarterly variable:
    ``\dim(z_t) = r \cdot n_{\text{blocks}} \cdot p_{\text{eff}} + n_M \mathbb{1}_{\{\texttt{:ar1}\}} + 5 n_Q``.
    The default panel here gives ``2 \cdot 1 \cdot 5 + 4 + 5 = 19`` states.

---

## Estimation

The EM algorithm (Bańbura & Modugno 2014) estimates all state-space parameters jointly and never requires a balanced panel:

1. **E-step.** The Kalman smoother runs with NaN-aware observation equations: at each period only the rows of ``C`` for observed variables enter the update, so a month with three of five series released is handled exactly like a complete month with a smaller cross-section. The step returns the smoothed means ``E[z_t \mid \mathcal{I}_T]`` and covariances ``V[z_t \mid \mathcal{I}_T]``.

2. **M-step.** The smoothed second moments update every parameter in closed form: the factor VAR coefficients ``A`` by OLS on the smoothed states, the loadings ``C`` by per-variable OLS (Mariano-Murasawa constrained on quarterly rows), the covariances ``Q`` and ``R`` from the implied residual second moments, the idiosyncratic AR(1) coefficients from their own autocovariances, and the initial condition ``(Z_0, V_0)`` from the smoothed first period.

Iteration stops when the relative change in the log-likelihood falls below `thresh`:

```math
\frac{|\ell^{(k)} - \ell^{(k-1)}|}{|\ell^{(k-1)}|} < \text{thresh}
```

!!! warning "`n_iter == max_iter` means the run did not converge"
    `n_iter` records the iteration the loop exited on. A converging run exits early, so `n_iter < max_iter`; a run that exhausts its budget reports `n_iter == max_iter` and returns whatever parameters it had reached. Compare the two before using the estimates.

```@example nc_dfm
loose = nowcast_dfm(Y, nM, nQ; r=2, p=1, max_iter=100, thresh=1e-4)
tight = nowcast_dfm(Y, nM, nQ; r=2, p=1, max_iter=200, thresh=1e-6)
(loose = (iter=loose.n_iter, loglik=round(loose.loglik, digits=2)),
 tight = (iter=tight.n_iter, loglik=round(tight.loglik, digits=2)))
```

The default tolerance is met after 55 iterations at a log-likelihood of ``-342.84``. Tightening `thresh` by two orders of magnitude exhausts the 200-iteration budget — `iter` equals `max_iter`, the signature of a run that stopped on the cap rather than on the criterion — and buys 0.86 log-likelihood points. EM guarantees a monotonically non-decreasing likelihood, so those iterations are not wasted, but the flat tail shows the parameters have reached a stationary region and the tighter tolerance is chasing numerical noise. On this panel the AR(1) specification (``-342.84``) fits better than the i.i.d. one (``-350.16``): the idiosyncratic autoregressions absorb serial correlation that would otherwise be forced through the common factors.

---

## Block Structure

The `blocks` keyword takes an ``N \times B`` binary matrix whose ``(i, b)`` entry marks whether variable ``i`` loads on block ``b``. Each block carries its own set of ``r`` factors, so the model estimates ``r \cdot B`` factors under the restriction that a variable's loadings vanish outside its blocks. Use this when theory says different groups of series are driven by different latent forces — real activity, prices, financial conditions — and let a variable load on several blocks to capture the comovement between them.

```@example nc_dfm
# Real activity (INDPRO, UNRATE) and nominal (CPI, M2); FEDFUNDS loads on both
blocks = [1 0;    # INDPRO   → real
          1 0;    # UNRATE   → real
          0 1;    # CPI      → nominal
          0 1;    # M2       → nominal
          1 1]    # FEDFUNDS → both
dfm_block = nowcast_dfm(Y, nM, nQ; r=1, p=1, blocks=blocks)
(blocks=size(dfm_block.blocks, 2), factors_per_block=dfm_block.r,
 iter=dfm_block.n_iter, loglik=round(dfm_block.loglik, digits=2))
```

One factor on each of two blocks uses as many factors as the unrestricted model but forces each to load on its own group, and reaches ``-339.25`` against the unrestricted ``-342.84``. This is not a likelihood-ratio comparison — the restricted run stops at the 100-iteration cap and is still climbing — but it shows the block partition is not fighting the data. With `blocks=nothing` (the default) all variables load on a single global block.

!!! note "Factors actually extracted"
    The state space supports ``n_f = \min(r \cdot B, N)`` factors. Requesting more block-factors than there are series truncates silently to ``N``, and the loadings of the dropped factors stay zero.

---

## Forecasting

`forecast` iterates the state forward with the estimated transition matrix, ``z_{T+h} = A^h z_T``, and maps each projected state back through the observation equation. The horizon counts **months**, not quarters, so a quarterly target needs `h=3` for one quarter ahead and `h=6` for two.

```@example nc_dfm
forecast(dfm, 6; target_var=N)
```

The path starts at 0.0398, rises to 0.0479 by the second month and then settles near 0.046: with a stationary factor VAR the state decays toward its unconditional mean, so the forecast converges on the sample mean of the target and longer horizons add no information. Omitting `target_var` returns the full ``h \times N`` panel instead of a single column. Both forms return a `NowcastForecast`, which prints as a horizon table and indexes like the underlying array, so `fc[3]` and `fc[3, 2]` work directly.

`nowcast(model)` is the convenience wrapper for the single number the model exists to produce. It reports the last smoothed value of the target as the current-quarter estimate and projects the state three months ahead for the next quarter.

```@example nc_dfm
result = nowcast(dfm)
(nowcast=round(result.nowcast, digits=4), forecast=round(result.forecast, digits=4),
 method=result.method)
```

---

## Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `r` | `Int` | `2` | Number of factors per block |
| `p` | `Int` | `1` | VAR lags in the factor dynamics |
| `idio` | `Symbol` | `:ar1` | Idiosyncratic dynamics, `:ar1` or `:iid` |
| `blocks` | `Matrix{Int}` or `nothing` | `nothing` | ``N \times B`` binary block-membership matrix |
| `max_iter` | `Int` | `100` | Maximum EM iterations |
| `thresh` | `Real` | ``10^{-4}`` | Convergence threshold on the relative log-likelihood change |

---

## NowcastDFM Return Values

| Field | Type | Description |
|-------|------|-------------|
| `X_sm` | `Matrix{T}` | Smoothed panel in original units, observed entries preserved and every NaN filled |
| `F` | `Matrix{T}` | Smoothed factors, ``T_{\text{obs}} \times \min(rp, \dim z)`` |
| `C` | `Matrix{T}` | Observation loadings, ``N \times \dim z`` |
| `A` | `Matrix{T}` | State transition matrix, ``\dim z \times \dim z`` |
| `Q` | `Matrix{T}` | State innovation covariance |
| `R` | `Matrix{T}` | Observation noise covariance (diagonal) |
| `Mx` | `Vector{T}` | Column means used for standardization |
| `Wx` | `Vector{T}` | Column standard deviations used for standardization |
| `Z_0` | `Vector{T}` | Initial state mean |
| `V_0` | `Matrix{T}` | Initial state covariance |
| `r` | `Int` | Factors per block |
| `p` | `Int` | VAR lags in the factor dynamics |
| `blocks` | `Matrix{Int}` | Block-membership matrix used (all ones when `blocks=nothing`) |
| `loglik` | `T` | Log-likelihood at the final iteration |
| `n_iter` | `Int` | EM iterations used; equals `max_iter` when the run hit the cap |
| `nM` | `Int` | Number of monthly variables |
| `nQ` | `Int` | Number of quarterly variables |
| `idio` | `Symbol` | Idiosyncratic specification used |
| `data` | `Matrix{T}` | Original input panel, NaN included |

`StatsAPI` methods are defined: `loglikelihood(dfm)` returns `loglik`, `predict(dfm)` returns `X_sm`, and `nobs(dfm)` returns the number of periods.

---

## Balancing Panels

`balance_panel` reuses the DFM as a general-purpose imputer for a `TimeSeriesData` or `PanelData` container. Every variable is treated as monthly (``n_Q = 0``), observed values pass through untouched, and only NaN entries are replaced by their smoothed estimates. A container with no missing values is returned unchanged.

```@example nc_dfm
Y_bal = to_matrix(apply_tcode(nc_md))
Y_bal = Y_bal[all.(isfinite, eachrow(Y_bal)), :]
Y_bal = Y_bal[end-59:end, :]
Y_bal[end, 1:3] .= NaN          # three indicators not yet released

ts_ragged = TimeSeriesData(Y_bal; varnames=["INDPRO", "UNRATE", "CPI", "M2", "FEDFUNDS"],
                           frequency=Monthly)
ts_filled = balance_panel(ts_ragged; r=2, p=1)
(missing_before=count(isnan, ts_ragged.data), missing_after=count(isnan, ts_filled.data),
 filled=round.(ts_filled.data[end, 1:3], digits=4))
```

The three unreleased values are reconstructed from the factors that the two remaining series still identify in that month. For a `PanelData` container the routine runs group by group, and the result is marked balanced when every group ends up with the same number of periods.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:dfm` | Fill method; only `:dfm` is implemented |
| `r` | `Int` | `3` | Factors for the imputation DFM, capped at ``\min(N - 1, T - 1)`` |
| `p` | `Int` | `2` | Factor VAR lags, capped at ``T/3`` |

---

## Complete Example

```@example nc_dfm
# === Step 1: Estimate the DFM ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=100, thresh=1e-4)
report(dfm)
```

```@example nc_dfm
# === Step 2: Current quarter and next quarter ===
result = nowcast(dfm)
report(result)
```

```@example nc_dfm
# === Step 3: Monthly forecast path for the target ===
forecast(dfm, 6; target_var=N)
```

```@example nc_dfm
# === Step 4: Attribute a revision to the three latest monthly releases ===
X_old = copy(Y)
X_old[end, 1:3] .= NaN          # vintage before INDPRO, UNRATE and CPI were published
news = nowcast_news(Y, X_old, dfm, T_obs; target_var=N)
report(news)
```

**Interpretation.** Two factors summarize the four monthly FRED-MD indicators and, through the Mariano-Murasawa weights, the quarterly `FEDFUNDS` target; EM converges in 55 iterations at a log-likelihood of ``-342.84``. The current-quarter estimate is 0.0375 and the three-month-ahead projection 0.0472, so the model reads the quarter as running slightly below where the factor dynamics point next. Adding the three withheld monthly releases moves the nowcast from 0.0368 to 0.0375, a revision of 0.0007 to which the CPI release contributes the largest single piece (0.0006) while the unemployment release pulls the other way. Because the DFM parameters are held fixed across the two vintages, the news terms account for the revision to machine precision; [News Decomposition](@ref nowcast_news_page) covers the weighting scheme.

For the `NowcastResult` visualizations — the smoothed target with its nowcast extension, the ragged-edge z-score heatmap, and the factor-contribution decomposition — see the Visualization section of [Nowcasting](@ref nowcast_page).

---

## Common Pitfalls

1. **Column ordering is part of the contract.** The first `nM` columns must be the monthly variables and the last `nQ` the quarterly ones. Nothing validates the ordering: reversing it applies the ``[1,2,3,2,1]`` aggregation to a monthly series and treats a quarterly series as monthly.

2. **The quarterly mask must sit at `mod(t, 3) == 0`.** The temporal aggregation assumes a quarterly observation at month ``t`` summarizes months ``t-2, t-1, t``. Masking at a different phase misaligns the weights against the data and biases the loadings without raising an error.

3. **Too many factors defeats the purpose.** As ``r`` approaches ``N`` the factors reproduce the panel instead of compressing it. Keep ``r \leq N/3`` and use `ic_criteria(X, r_max)` from the factor-model module to pick ``r`` by information criterion. With block structure the binding count is ``r \cdot B``, truncated at ``N``.

4. **Check `n_iter` against `max_iter` before trusting the estimates.** A run that hits the cap has not converged. Raise `max_iter` first; if the likelihood is still climbing after that, the factor structure is weak or the sample too short, and no tolerance setting will fix it.

5. **`:ar1` costs one state per monthly variable.** On a wide panel it inflates the state vector by ``n_M`` and the Kalman recursions scale cubically in the state dimension. Switch to `:iid` for large panels, accepting that idiosyncratic serial correlation then has to be absorbed by the factors.

6. **Transform before estimating.** The DFM standardizes each column but does not difference or deflate. Feed it stationary series — `apply_tcode()` applies the FRED-MD transformation codes — since a trending panel drives the factors into a common trend and leaves nothing to nowcast with.

---

## References

- Bańbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2011. "Nowcasting." In *The Oxford Handbook of Economic Forecasting*, 193--224. Oxford: Oxford University Press. [https://doi.org/10.1093/oxfordhb/9780195398649.013.0008](https://doi.org/10.1093/oxfordhb/9780195398649.013.0008)
- Giannone, Domenico, Lucrezia Reichlin, and David Small. 2008. "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55 (4): 665--676. [https://doi.org/10.1016/j.jmoneco.2008.05.010](https://doi.org/10.1016/j.jmoneco.2008.05.010)
- Mariano, Roberto S., and Yasutomo Murasawa. 2003. "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18 (4): 427--443. [https://doi.org/10.1002/jae.695](https://doi.org/10.1002/jae.695)
