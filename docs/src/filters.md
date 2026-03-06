# [Time Series Filters](@id filters_page)

**MacroEconometricModels.jl** provides five standard filters for decomposing macroeconomic time series into trend and cyclical components. Each filter embodies a different assumption about what constitutes the "trend," making the choice of filter an economic modeling decision.

- **Hodrick-Prescott**: Penalized least squares smoother (Hodrick & Prescott 1997) with frequency-dependent ``\lambda``
- **Hamilton**: OLS regression-based filter (Hamilton 2018) that avoids spurious cyclicality and endpoint bias
- **Beveridge-Nelson**: ARIMA-based or state-space decomposition (Beveridge & Nelson 1981) into permanent and transitory components
- **Baxter-King**: Symmetric band-pass filter (Baxter & King 1999) isolating fluctuations in a specified frequency band
- **Boosted HP**: Iterated HP with data-driven stopping (Phillips & Shi 2021) that removes residual unit root behavior from the cycle

All results support unified `trend()` and `cycle()` accessors, `report()` for tabular output, and `plot_result()` for interactive D3.js visualization.

## Quick Start

**Recipe 1: HP filter on monthly data**

```julia
using MacroEconometricModels

# Log industrial production — a trending I(1) monthly series
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

hp = hp_filter(y; lambda=129600.0)
report(hp)
```

**Recipe 2: Hamilton regression filter**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly parameters: 2-year horizon (h=24), 12 monthly lags
ham = hamilton_filter(y; h=24, p=12)
report(ham)
```

**Recipe 3: Beveridge-Nelson decomposition**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

bn = beveridge_nelson(y)
report(bn)
```

**Recipe 4: Baxter-King band-pass filter**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly business cycle band: 18–96 months (1.5–8 years), K=36
bk = baxter_king(y; pl=18, pu=96, K=36)
report(bk)
```

**Recipe 5: Boosted HP with BIC stopping**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)
report(bhp)
```

**Recipe 6: Visualize any filter result**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

hp = hp_filter(y; lambda=129600.0)
p = plot_result(hp)
save_plot(p, "hp_filter.html")
```

---

## Hodrick-Prescott Filter

The HP filter (Hodrick & Prescott 1997) decomposes a time series ``y_t`` into a smooth trend ``\tau_t`` and a cyclical component ``c_t = y_t - \tau_t`` by solving the penalized least squares problem:

```math
\min_{\tau} \sum_{t=1}^T (y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1} (\tau_{t+1} - 2\tau_t + \tau_{t-1})^2
```

where:
- ``y_t`` is the observed time series at time ``t``
- ``\tau_t`` is the trend component
- ``\lambda`` is the smoothing parameter controlling trend curvature
- ``T`` is the sample size

The first term penalizes deviations of the trend from the data; the second penalizes curvature (second differences) in the trend. As ``\lambda \to 0`` the trend converges to the data; as ``\lambda \to \infty`` the trend converges to a linear time trend.

!!! note "Technical Note"
    The closed-form solution is ``\tau = (I + \lambda D'D)^{-1} y`` where ``D`` is the ``(T-2) \times T`` second-difference matrix. The implementation builds a sparse pentadiagonal system and solves via Cholesky factorization, giving ``O(T)`` computational cost.

### Choosing ``\lambda``

The smoothing parameter must match the data frequency. Ravn and Uhlig (2002) provide a frequency-based justification for scaling ``\lambda`` by the fourth power of the frequency ratio relative to the quarterly benchmark:

| Data Frequency | Recommended ``\lambda`` |
|----------------|------------------------|
| Annual | 6.25 |
| Quarterly | 1,600 |
| Monthly | 129,600 |

```julia
using MacroEconometricModels

# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly smoothing parameter
hp = hp_filter(y; lambda=129600.0)
report(hp)

# Visualize trend and cycle
p = plot_result(hp)
```

```@raw html
<iframe src="../assets/plots/filter_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The HP trend tracks the low-frequency movements in log industrial production, while the cycle component captures business cycle fluctuations. The cycle standard deviation indicates the amplitude of the extracted fluctuations relative to trend.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lambda` | `Real` | `1600.0` | Smoothing parameter (6.25 annual, 1600 quarterly, 129600 monthly) |

### Return Value (`HPFilterResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `trend` | `Vector{T}` | Estimated trend component (length ``T``) |
| `cycle` | `Vector{T}` | Cyclical component ``y - \tau`` (length ``T``) |
| `lambda` | `T` | Smoothing parameter used |
| `T_obs` | `Int` | Number of observations |

---

## Hamilton Filter

Hamilton (2018) proposes a regression-based alternative to the HP filter that avoids spurious cyclicality, endpoint bias, and spurious dynamic relations between filtered series. The filter regresses the future value ``y_{t+h}`` on a constant and ``p`` lags:

```math
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_t
```

where:
- ``y_{t+h}`` is the dependent variable (``h``-period-ahead value)
- ``\beta_0`` is the intercept
- ``\beta_1, \ldots, \beta_p`` are OLS coefficients on lagged values
- ``v_t`` is the residual (cyclical component)
- ``h`` is the forecast horizon
- ``p`` is the number of lags

The fitted values ``\hat{y}_{t+h}`` form the trend and the OLS residuals ``v_t`` form the cycle. The default parameters ``h = 8``, ``p = 4`` correspond to a 2-year ahead projection using 4 quarterly lags.

!!! warning "Observation loss"
    The Hamilton filter loses ``h + p - 1`` observations at the start of the sample. For monthly data with ``h=24``, ``p=12``, this is 35 observations. Plan accordingly with short samples.

```julia
using MacroEconometricModels

# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly parameters: 2-year horizon, 12 monthly lags
ham = hamilton_filter(y; h=24, p=12)
report(ham)

# OLS coefficients from the predictive regression
ham.beta

# Visualize (pass original series for overlay on shortened output)
p = plot_result(ham; original=y)
```

```@raw html
<iframe src="../assets/plots/filter_hamilton.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The Hamilton cycle avoids the spurious cyclicality that plagues the HP filter at sample endpoints. Hamilton (2018) demonstrates that this filter is robust to unit roots and structural breaks, making it the preferred choice when endpoint behavior matters.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `h` | `Int` | `8` | Forecast horizon (8 for quarterly = 2 years, 24 for monthly) |
| `p` | `Int` | `4` | Number of lags in the regression (4 for quarterly, 12 for monthly) |

### Return Value (`HamiltonFilterResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `trend` | `Vector{T}` | Fitted values (length ``T - h - p + 1``) |
| `cycle` | `Vector{T}` | OLS residuals (length ``T - h - p + 1``) |
| `beta` | `Vector{T}` | OLS coefficients ``[\beta_0, \beta_1, \ldots, \beta_p]`` |
| `h` | `Int` | Forecast horizon used |
| `p` | `Int` | Number of lags used |
| `T_obs` | `Int` | Original series length |
| `valid_range` | `UnitRange{Int}` | Indices into original series where results are valid |

---

## Beveridge-Nelson Decomposition

The Beveridge-Nelson (1981) decomposition separates an I(1) process into a permanent (random walk with drift) component and a stationary transitory component. It exploits the Wold representation of the first-differenced series:

```math
\Delta y_t = \mu + \psi(L) \varepsilon_t = \mu + \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}
```

where:
- ``\Delta y_t = y_t - y_{t-1}`` is the first difference
- ``\mu`` is the drift (mean growth rate)
- ``\psi(L) = \sum_{j=0}^{\infty} \psi_j L^j`` is the lag polynomial with ``\psi_0 = 1``
- ``\varepsilon_t`` is a white noise innovation

The long-run multiplier ``\psi(1) = 1 + \sum_{j=1}^{\infty} \psi_j`` determines the permanent impact of shocks. The decomposition is:

```math
y_t = \tau_t + c_t
```

where:
- ``\tau_t`` is the permanent component (random walk with drift ``\mu \cdot \psi(1)``)
- ``c_t`` is the transitory component (mean-zero stationary process)

!!! note "Technical Note"
    Two methods are available. The classic `:arima` method fits an ARMA model to ``\Delta y_t``, computes the ``\psi``-weights from the MA(``\infty``) representation, and constructs the transitory component. The `:statespace` method estimates the correlated unobserved components (UC) model of Morley, Nelson & Zivot (2003) via MLE and Kalman smoother, allowing correlation between permanent and transitory innovations.

```julia
using MacroEconometricModels

# Log industrial production — an I(1) series suitable for BN decomposition
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Automatic ARMA order selection for Δy
bn = beveridge_nelson(y)
report(bn)

# Manual ARMA order specification
bn2 = beveridge_nelson(y; p=2, q=1)

# Correlated UC model (Morley, Nelson & Zivot 2003)
bn_ss = beveridge_nelson(y; method=:statespace, cycle_order=2)

# Visualize permanent and transitory components
p = plot_result(bn)
```

```@raw html
<iframe src="../assets/plots/filter_bn.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The permanent component tracks the stochastic trend in log industrial production, while the transitory component captures stationary deviations from trend. The long-run multiplier ``\psi(1)`` quantifies how much of each unit innovation becomes permanent --- values above 1 indicate that transitory dynamics amplify the long-run effect of shocks.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:arima` | Decomposition method (`:arima` or `:statespace`) |
| `p` | `Int` or `Symbol` | `:auto` | AR order for ARMA model of ``\Delta y`` (`:auto` uses `auto_arima`) |
| `q` | `Int` or `Symbol` | `:auto` | MA order for ARMA model of ``\Delta y`` (`:auto` uses `auto_arima`) |
| `max_terms` | `Int` | `500` | Maximum ``\psi``-weights for MA(``\infty``) truncation |
| `cycle_order` | `Int` | `2` | AR order for cyclical component (`:statespace` method, 1 or 2) |

### Return Value (`BeveridgeNelsonResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `permanent` | `Vector{T}` | Permanent (trend) component |
| `transitory` | `Vector{T}` | Transitory (cycle) component |
| `drift` | `T` | Estimated drift ``\mu`` |
| `long_run_multiplier` | `T` | Long-run multiplier ``\psi(1)`` |
| `arima_order` | `Tuple{Int,Int,Int}` | ``(p, d, q)`` order used |
| `T_obs` | `Int` | Number of observations |

---

## Baxter-King Band-Pass Filter

The Baxter-King (1999) filter isolates cyclical fluctuations in a specified frequency band ``[\omega_L, \omega_H]`` using a symmetric finite moving average approximation to the ideal band-pass filter. The ideal (infinite) band-pass filter has weights:

```math
B_0 = \frac{\omega_H - \omega_L}{\pi}, \quad B_j = \frac{\sin(\omega_H j) - \sin(\omega_L j)}{\pi j} \quad \text{for } j \geq 1
```

where:
- ``\omega_H = 2\pi / p_l`` is the high-frequency cutoff (short-period boundary)
- ``\omega_L = 2\pi / p_u`` is the low-frequency cutoff (long-period boundary)
- ``p_l`` and ``p_u`` are the minimum and maximum periods of oscillation to pass

The ideal filter is truncated at lag ``K`` and adjusted to ensure the weights sum to zero, eliminating stochastic trends:

```math
a_j = B_j + \theta, \quad \theta = -\frac{B_0 + 2\sum_{j=1}^K B_j}{2K + 1}
```

where:
- ``a_j`` is the adjusted filter weight at lag ``j``
- ``\theta`` is the correction ensuring ``a_0 + 2\sum_{j=1}^K a_j = 0``
- ``K`` is the truncation length

The filtered series is:

```math
c_t = a_0 y_t + \sum_{j=1}^K a_j (y_{t-j} + y_{t+j})
```

where:
- ``c_t`` is the band-pass filtered (cyclical) component
- ``y_t`` is the observed time series

!!! warning "Endpoint truncation"
    The BK filter loses ``K`` observations at each end (``2K`` total). With ``K = 36`` and monthly data, this is 6 years of data at the boundaries.

```julia
using MacroEconometricModels

# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly business cycle band: 18–96 months (1.5–8 years), K=36
bk = baxter_king(y; pl=18, pu=96, K=36)
report(bk)

# Verify weights sum to zero by construction
w = bk.weights
total = w[1] + 2 * sum(w[2:end])  # ≈ 0

# Visualize (pass original series for overlay on shortened output)
p = plot_result(bk; original=y)
```

```@raw html
<iframe src="../assets/plots/filter_bk.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The BK filter extracts fluctuations in the 1.5--8 year range, corresponding to the NBER business cycle definition. The zero-sum weight constraint ensures that unit root processes pass through the filter as stationary series, making it appropriate for trending data without prior differencing.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `pl` | `Int` | `6` | Minimum period of oscillation to pass (quarterly: 6 = 1.5 years) |
| `pu` | `Int` | `32` | Maximum period of oscillation to pass (quarterly: 32 = 8 years) |
| `K` | `Int` | `12` | Truncation length (observations lost at each end) |

### Return Value (`BaxterKingResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `cycle` | `Vector{T}` | Band-pass filtered component (length ``T - 2K``) |
| `trend` | `Vector{T}` | Residual (low + high frequency, length ``T - 2K``) |
| `weights` | `Vector{T}` | Symmetric filter weights ``[a_0, a_1, \ldots, a_K]`` |
| `pl` | `Int` | Lower period bound |
| `pu` | `Int` | Upper period bound |
| `K` | `Int` | Truncation length |
| `T_obs` | `Int` | Original series length |
| `valid_range` | `UnitRange{Int}` | Indices into original series where results are valid |

---

## Boosted HP Filter

Phillips and Shi (2021) propose iterating the HP filter on the cyclical component to improve trend estimation when the data contains stochastic trends. A single HP pass leaves unit root behavior in the cycle; re-filtering removes it. The algorithm proceeds as:

```math
\hat{c}^{(m)} = (I - S) \hat{c}^{(m-1)}, \quad \hat{\tau}^{(m^*)} = y - \hat{c}^{(m^*)}
```

where:
- ``S = (I + \lambda D'D)^{-1}`` is the HP smoother matrix
- ``\hat{c}^{(m)}`` is the cyclical component at iteration ``m``
- ``\hat{\tau}^{(m^*)}`` is the final trend estimate at stopping iteration ``m^*``
- ``I - S`` is the HP cycle extraction operator

### Stopping Criteria

Three stopping rules determine the optimal number of iterations ``m^*``:

| Criterion | Symbol | Behavior |
|-----------|--------|----------|
| **Phillips-Shi IC** | `:BIC` | Fit AR(1) to cycle at each iteration; stop when the information criterion increases |
| **ADF test** | `:ADF` | Run ADF test on cycle; stop when unit root null is rejected at level `sig_p` |
| **Fixed** | `:fixed` | Run exactly `max_iter` iterations |

!!! note "Technical Note"
    The Phillips-Shi information criterion balances variance reduction against effective degrees of freedom: ``\text{IC}(m) = \text{Var}(c_m) / \text{Var}(c_1) + \log(T) \cdot \text{tr}(B_m) / \text{tr}(I - S)`` where ``B_m = I - (I - S)^m``. The eigenvalues of ``(I - S)`` are computed once and reused across iterations.

```julia
using MacroEconometricModels

# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# BIC stopping (default) with monthly lambda
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)
report(bhp)

# ADF stopping — ensures the cycle is stationary
bhp_adf = boosted_hp(y; lambda=129600.0, stopping=:ADF, sig_p=0.05)

# Fixed iterations for comparison
bhp_fixed = boosted_hp(y; lambda=129600.0, stopping=:fixed, max_iter=5)

# Visualize
p = plot_result(bhp)
```

```@raw html
<iframe src="../assets/plots/filter_boosted_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The boosted HP trend is sharper than the standard HP trend, tracking structural shifts more closely. The number of iterations indicates how many re-filterings were needed to remove unit root behavior from the cycle --- more iterations imply stronger trend persistence in the original data. Mei, Phillips & Shi (2024) show that the boosted HP filter encompasses the standard HP filter as the special case ``m^* = 1``.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lambda` | `Real` | `1600.0` | HP smoothing parameter (same frequency rules as standard HP) |
| `stopping` | `Symbol` | `:BIC` | Stopping criterion (`:BIC`, `:ADF`, or `:fixed`) |
| `max_iter` | `Int` | `100` | Maximum number of boosting iterations |
| `sig_p` | `Real` | `0.05` | Significance level for ADF stopping criterion |

### Return Value (`BoostedHPResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `trend` | `Vector{T}` | Final boosted trend estimate (length ``T``) |
| `cycle` | `Vector{T}` | Final cyclical component (length ``T``) |
| `lambda` | `T` | Smoothing parameter used |
| `iterations` | `Int` | Number of boosting iterations performed |
| `stopping` | `Symbol` | Stopping criterion used (`:ADF`, `:BIC`, or `:fixed`) |
| `bic_path` | `Vector{T}` | Phillips-Shi IC value at each iteration |
| `adf_pvalues` | `Vector{T}` | ADF p-values at each iteration |
| `T_obs` | `Int` | Number of observations |

---

## Unified Accessors

All filter results inherit from `AbstractFilterResult` and support the `trend()` and `cycle()` accessors for uniform access to decomposition components:

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

hp  = hp_filter(y; lambda=129600.0)
ham = hamilton_filter(y; h=24, p=12)
bn  = beveridge_nelson(y)
bk  = baxter_king(y; pl=18, pu=96, K=36)
bhp = boosted_hp(y; lambda=129600.0)

# Uniform interface across all filter types
for r in [hp, ham, bn, bk, bhp]
    t = trend(r)   # trend component
    c = cycle(r)   # cyclical component
end
```

For `BeveridgeNelsonResult`, `trend()` returns the permanent component and `cycle()` returns the transitory component.

---

## Complete Example

This example applies all five filters to log industrial production from FRED-MD and compares the extracted business cycles:

```julia
using MacroEconometricModels
using Statistics

# Log industrial production (monthly, FRED-MD)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Apply all five filters with monthly parameters
hp  = hp_filter(y; lambda=129600.0)
ham = hamilton_filter(y; h=24, p=12)
bn  = beveridge_nelson(y)
bk  = baxter_king(y; pl=18, pu=96, K=36)
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)

# Report each filter
report(hp)
report(ham)
report(bn)
report(bk)
report(bhp)

# Compare cycle amplitudes
round(std(cycle(hp)), digits=4)
round(std(cycle(ham)), digits=4)
round(std(cycle(bn)), digits=4)
round(std(cycle(bk)), digits=4)
round(std(cycle(bhp)), digits=4)
```

The HP, Hamilton, and boosted HP cycles have full sample length, while the Hamilton and Baxter-King cycles are shorter due to observation loss. Cycle standard deviations differ across filters because each isolates a different frequency range: the BK filter targets a specific band (18--96 months), the HP filter penalizes curvature globally, and the Hamilton filter captures predictable variation over a 2-year horizon. The boosted HP cycle is typically smaller in amplitude than the standard HP cycle because additional iterations remove residual trend contamination.

---

## Common Pitfalls

1. **Wrong ``\lambda`` for data frequency.** Using ``\lambda = 1600`` (the quarterly default) on monthly data produces an excessively smooth trend that misses business cycle turning points. Scale ``\lambda`` by the fourth power of the frequency ratio: 6.25 for annual, 1600 for quarterly, 129600 for monthly (Ravn & Uhlig 2002).

2. **Hamilton filter observation loss.** The Hamilton filter loses ``h + p - 1`` observations at the start. For monthly data with ``h = 24``, ``p = 12``, the first 35 observations are unavailable. With a short sample this can eliminate a substantial fraction of the data.

3. **Baxter-King endpoint truncation.** The BK filter loses ``K`` observations at each end (``2K`` total). With ``K = 36`` on monthly data, 6 years are trimmed from the boundaries. Choosing a smaller ``K`` reduces data loss but worsens the approximation to the ideal band-pass filter.

4. **Beveridge-Nelson on I(0) data.** The BN decomposition assumes the series is I(1). Applying it to a stationary series produces a degenerate decomposition where the permanent component absorbs nearly all variation. Verify the unit root assumption with `adf_test` or `kpss_test` before using.

5. **Boosted HP stopping criterion choice.** The `:BIC` criterion balances parsimony and fit but may stop too early on series with strong trend persistence. The `:ADF` criterion ensures cycle stationarity but may over-iterate on near-unit-root processes. Use `:fixed` with a known iteration count for replication studies.

6. **HP filter endpoint bias.** The HP filter exhibits spurious cyclicality at sample endpoints (Hamilton 2018). Real-time analysis that depends on the most recent observations should prefer the Hamilton filter or boosted HP, which are more robust at the boundary.

---

## References

- Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. Business Cycles: An Empirical Investigation.
  *Journal of Money, Credit and Banking*, 29(1), 1--16. [DOI](https://doi.org/10.2307/2953682)

- Ravn, M. O., & Uhlig, H. (2002). On Adjusting the Hodrick-Prescott Filter for the Frequency of Observations.
  *Review of Economics and Statistics*, 84(2), 371--376. [DOI](https://doi.org/10.1162/003465302317411604)

- Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
  *Review of Economics and Statistics*, 100(5), 831--843. [DOI](https://doi.org/10.1162/rest_a_00706)

- Beveridge, S., & Nelson, C. R. (1981). A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'.
  *Journal of Monetary Economics*, 7(2), 151--174. [DOI](https://doi.org/10.1016/0304-3932(81)90040-4)

- Morley, J. C., Nelson, C. R., & Zivot, E. (2003). Why Are the Beveridge-Nelson and Unobserved-Components Decompositions of GDP So Different?
  *Review of Economics and Statistics*, 85(2), 235--243. [DOI](https://doi.org/10.1162/003465303765299774)

- Baxter, M., & King, R. G. (1999). Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series.
  *Review of Economics and Statistics*, 81(4), 575--593. [DOI](https://doi.org/10.1162/003465399558454)

- Phillips, P. C. B., & Shi, Z. (2021). Boosting: Why You Can Use the HP Filter.
  *International Economic Review*, 62(2), 521--570. [DOI](https://doi.org/10.1111/iere.12495)

- Mei, Z., Phillips, P. C. B., & Shi, Z. (2024). The Boosted HP Filter Is More General Than You Might Think.
  *Journal of Applied Econometrics*, 39(7), 1260--1281. [DOI](https://doi.org/10.1002/jae.3086)
