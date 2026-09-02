# [Time Series Filters](@id filters_page)

**MacroEconometricModels.jl** provides five standard filters for decomposing macroeconomic time series into trend and cyclical components. Each filter embodies a different assumption about what constitutes the "trend," making the choice of filter an economic modeling decision.

- **Hodrick-Prescott**: Penalized least squares smoother (Hodrick & Prescott 1997) with frequency-dependent ``\lambda``
- **Hamilton**: OLS regression-based filter (Hamilton 2018) that avoids spurious cyclicality and endpoint bias
- **Beveridge-Nelson**: ARIMA-based or state-space decomposition (Beveridge & Nelson 1981) into permanent and transitory components
- **Baxter-King**: Symmetric band-pass filter (Baxter & King 1999) isolating fluctuations in a specified frequency band
- **Boosted HP**: Iterated HP with data-driven stopping (Phillips & Shi 2021) that removes residual unit root behavior from the cycle

All results support unified `trend()` and `cycle()` accessors, `report()` for tabular output, and `plot_result()` for interactive D3.js visualization.

Two neighboring pages cover related decompositions. [State-Space Models](@ref statespace_page) estimates trend and cycle as latent states of an explicit model, and is the natural next step when the decomposition needs a likelihood. [X-13ARIMA-SEATS](@ref x13_page) removes the *seasonal* component rather than the trend. [Spectral Analysis](@ref spectral_page) supplies `transfer_function`, which plots exactly which frequencies each filter on this page passes.

```@setup filters
using MacroEconometricModels, Statistics
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))
```

## Quick Start

The examples below filter the log of U.S. industrial production (`INDPRO` from FRED-MD, 804 monthly observations).

**Recipe 1: HP filter on monthly data**

```@example filters
hp = hp_filter(y; lambda=129600.0)
report(hp)
```

**Recipe 2: Hamilton regression filter**

```@example filters
# Monthly parameters: 2-year horizon (h=24), 12 monthly lags
ham = hamilton_filter(y; h=24, p=12)
report(ham)
```

**Recipe 3: Beveridge-Nelson decomposition**

```@example filters
bn = beveridge_nelson(y)
report(bn)
```

**Recipe 4: Baxter-King band-pass filter**

```@example filters
# Monthly business cycle band: 18–96 months (1.5–8 years), K=36
bk = baxter_king(y; pl=18, pu=96, K=36)
report(bk)
```

**Recipe 5: Boosted HP with BIC stopping**

```@example filters
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)
report(bhp)
```

**Recipe 6: Compare cycle amplitudes through the unified accessors**

```@example filters
(hp = round(std(cycle(hp)), digits=4),
 hamilton = round(std(cycle(ham)), digits=4),
 bn = round(std(cycle(bn)), digits=4),
 bk = round(std(cycle(bk)), digits=4),
 boosted = round(std(cycle(bhp)), digits=4))
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
    The closed-form solution is ``\tau = (I + \lambda D'D)^{-1} y`` where ``D`` is the ``(T-2) \times T`` second-difference matrix. The implementation builds a sparse pentadiagonal system and solves via Cholesky factorization, giving ``O(T)`` computational cost. Setting `lambda=0` short-circuits the solve and returns ``\tau = y`` with a zero cycle.

### Choosing ``\lambda``

The smoothing parameter must match the data frequency. Ravn and Uhlig (2002) provide a frequency-based justification for scaling ``\lambda`` by the fourth power of the frequency ratio relative to the quarterly benchmark:

| Data Frequency | Recommended ``\lambda`` |
|----------------|------------------------|
| Annual | 6.25 |
| Quarterly | 1,600 |
| Monthly | 129,600 |

```@example filters
# Monthly smoothing parameter
hp = hp_filter(y; lambda=129600.0)
report(hp)
```

```julia
# Visualize trend and cycle
p = plot_result(hp)
```

```@raw html
<iframe src="../assets/plots/filter_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The HP trend tracks the low-frequency movements in log industrial production; its mean of ``4.1288`` corresponds to an index level of ``e^{4.1288} \approx 62`` averaged over the sample. The cycle standard deviation is ``0.0314``, so a typical HP business-cycle deviation is about 3.1 percent of trend output. Because the cycle is defined residually as ``y - \tau``, all variation the penalty declines to absorb — including high-frequency measurement noise — lands in the cycle.

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

```@example filters
# Monthly parameters: 2-year horizon, 12 monthly lags
ham = hamilton_filter(y; h=24, p=12)
report(ham)
```

```@example filters
# OLS coefficients from the predictive regression
round.(ham.beta, digits=4)
```

```julia
# Visualize (pass original series for overlay on shortened output)
p = plot_result(ham; original=y)
```

```@raw html
<iframe src="../assets/plots/filter_hamilton.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Of the 804 original observations, 769 survive the ``h + p - 1 = 35``-observation loss, leaving a valid range of ``36{:}804``. The regression loads on the current level (``1.2187``), the first lag (``-0.4626``), and the two lags closest to the one-year mark (``-0.3310`` and ``0.5673``); every intervening coefficient is below ``0.1`` in absolute value. The slope coefficients sum to ``0.92``, so the trend is a slightly shrunk projection of the level. The resulting cycle standard deviation, ``0.0609``, is nearly double the HP cycle: what the Hamilton filter calls "cycle" is everything unpredictable at a 24-month horizon, which includes medium-run variation the HP penalty assigns to trend. Hamilton (2018) shows this construction is robust to unit roots and structural breaks, making it the preferred choice when endpoint behavior matters.

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
    The classic `:arima` method fits an ARMA model to ``\Delta y_t`` (order chosen by `auto_arima` over ``p, q \le 6`` when either is `:auto`), computes the ``\psi``-weights from the MA(``\infty``) representation, and constructs the transitory component. When the selected order is ARMA(0,0) the series is a random walk with drift, the transitory component is identically zero, and `report()` says so.

```@example filters
# Automatic ARMA order selection for Δy
bn = beveridge_nelson(y)
report(bn)
```

```@example filters
# Manual ARMA order specification
bn2 = beveridge_nelson(y; p=2, q=1)
report(bn2)
```

```julia
# Visualize permanent and transitory components
p = plot_result(bn)
```

```@raw html
<iframe src="../assets/plots/filter_bn.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Automatic selection lands on ARIMA(0,1,1) for log industrial production, with drift ``0.0019`` — average monthly growth of 0.19 percent, or 2.3 percent annualized. The long-run multiplier ``\psi(1) = 1.3070`` says a one-unit innovation ultimately shifts the level by 1.31 units, so the transitory dynamics *amplify* rather than damp the permanent effect. Forcing the richer ARMA(2,1) specification barely moves either quantity (``\psi(1) = 1.3388``), which is the usual sign that the MA(1) term already captures the short-run dynamics. The transitory standard deviation, ``0.0029``, is an order of magnitude below every other cycle on this page: under the BN definition almost all movement in industrial production is permanent.

### State-Space Decomposition

The `:statespace` method estimates the correlated unobserved-components model of Morley, Nelson & Zivot (2003) by maximum likelihood and extracts the components with a Kalman smoother. Unlike the ARIMA route it allows the permanent and transitory innovations to be correlated, which is what makes the UC cycle differ from the BN cycle in the first place.

!!! warning "Different field semantics under `:statespace`"
    The `:statespace` path reuses the `BeveridgeNelsonResult` container but not all of its fields carry the ARIMA meaning. `long_run_multiplier` is fixed at ``1.0`` (the UC model has no ARIMA long-run multiplier) and `arima_order` reports ``(\text{cycle\_order}, 0, 0)``, the AR order of the cycle, not an ARIMA specification. Estimation runs Nelder-Mead to 5000 iterations, takes several seconds on a full monthly sample, and may report near-singular matrix warnings from the diffuse-trend Kalman recursion.

```@example filters
# Correlated UC model (Morley, Nelson & Zivot 2003)
bn_ss = beveridge_nelson(y; method=:statespace, cycle_order=2)
report(bn_ss)
```

The UC cycle has standard deviation ``0.0103``, roughly three and a half times the ARIMA-based BN cycle, and the two correlate at ``0.70``. The gap is the point of Morley, Nelson & Zivot (2003): once the permanent and transitory innovations are allowed to correlate, the likelihood prefers a materially larger cycle than the BN identity delivers. The drift estimate, ``0.0019``, is essentially identical across the two methods, since both must reproduce the same average growth rate.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:arima` | Decomposition method (`:arima` or `:statespace`) |
| `p` | `Int` or `Symbol` | `:auto` | AR order for ARMA model of ``\Delta y`` (`:auto` uses `auto_arima`) |
| `q` | `Int` or `Symbol` | `:auto` | MA order for ARMA model of ``\Delta y`` (`:auto` uses `auto_arima`) |
| `max_terms` | `Int` | `500` | Maximum ``\psi``-weights for MA(``\infty``) truncation (`:arima` only) |
| `cycle_order` | `Int` | `2` | AR order for cyclical component (`:statespace` only, 1 or 2) |

### Return Value (`BeveridgeNelsonResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `permanent` | `Vector{T}` | Permanent (trend) component |
| `transitory` | `Vector{T}` | Transitory (cycle) component |
| `drift` | `T` | Estimated drift ``\mu`` |
| `long_run_multiplier` | `T` | Long-run multiplier ``\psi(1)``; fixed at ``1.0`` under `:statespace` |
| `arima_order` | `Tuple{Int,Int,Int}` | ``(p, d, q)`` order used; ``(\text{cycle\_order}, 0, 0)`` under `:statespace` |
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

```@example filters
# Monthly business cycle band: 18–96 months (1.5–8 years), K=36
bk = baxter_king(y; pl=18, pu=96, K=36)
report(bk)
```

```@example filters
# Weights sum to zero by construction
w = bk.weights
w[1] + 2 * sum(w[2:end])
```

```julia
# Visualize (pass original series for overlay on shortened output)
p = plot_result(bk; original=y)
```

```@raw html
<iframe src="../assets/plots/filter_bk.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The filter returns 732 of the 804 observations — ``2K = 72`` months, six years, are trimmed from the ends — over a valid range of ``37{:}768``. The weights sum to ``-4.2 \times 10^{-17}``, zero to machine precision, which is what allows a unit root process to pass through the filter as a stationary series without prior differencing. The extracted cycle has standard deviation ``0.0294``, just under the HP cycle's ``0.0314``: restricting attention to the 18--96 month band discards the high-frequency noise the HP residual retains, but the two agree closely on the business-cycle content itself.

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

Three stopping rules determine the number of iterations ``m^*``:

| Criterion | Symbol | Behavior |
|-----------|--------|----------|
| **Phillips-Shi IC** | `:BIC` | Fit the IC at each iteration; stop at the last iteration before it increases |
| **ADF test** | `:ADF` | Run an ADF test on the cycle; stop when the unit root null is rejected at level `sig_p` |
| **Fixed** | `:fixed` | Run exactly `max_iter` iterations |

!!! note "Technical Note"
    The Phillips-Shi information criterion balances variance reduction against effective degrees of freedom: ``\text{IC}(m) = \text{Var}(c_m) / \text{Var}(c_1) + \log(T) \cdot \text{tr}(B_m) / \text{tr}(I - S)`` where ``B_m = I - (I - S)^m``. The eigenvalues of ``(I - S)`` are computed once and reused across iterations.

```@example filters
# BIC stopping (default) with monthly lambda
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)
report(bhp)
```

```julia
# Visualize
p = plot_result(bhp)
```

```@raw html
<iframe src="../assets/plots/filter_boosted_hp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

BIC stopping runs 51 iterations before the information criterion turns up, driving the cycle standard deviation down to ``0.0151``, less than half the single-pass HP value of ``0.0314``. Each pass moves persistent variation out of the cycle and into the trend, so the boosted trend tracks the data more closely than the single-pass HP trend and bends to follow level shifts instead of smoothing through them. Mei, Phillips & Shi (2024) show the boosted HP filter encompasses the standard HP filter as the special case ``m^* = 1``.

!!! warning "The ADF rule can run to `max_iter`"
    On a strongly trending series the ADF test may never reject: on log industrial production the p-value stalls near ``0.99`` and `:ADF` exhausts `max_iter` without stopping, returning whatever iteration the budget allows. At high iteration counts the ADF regression design becomes ill-conditioned and `adf_test` reports near-singular matrix warnings. Check `iterations` against `max_iter` before trusting an `:ADF` result.

```@example filters
# ADF stopping — never rejects here, so it runs the full budget
bhp_adf = boosted_hp(y; lambda=129600.0, stopping=:ADF, sig_p=0.05, max_iter=20)

# Fixed iterations for replication
bhp_fixed = boosted_hp(y; lambda=129600.0, stopping=:fixed, max_iter=5)

(adf_iterations = bhp_adf.iterations,
 adf_last_pvalue = round(bhp_adf.adf_pvalues[end], digits=4),
 fixed_cycle_std = round(std(cycle(bhp_fixed)), digits=4))
```

The ADF variant stops only because it hits the 20-iteration budget, and its final p-value of ``0.9977`` confirms the cycle is nowhere near rejecting a unit root. The 5-iteration fixed run leaves a cycle standard deviation of ``0.0239``, between the single HP pass and the 51-iteration BIC solution — a direct illustration that the iteration count, not the criterion, is what sets the amplitude.

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
| `bic_path` | `Vector{T}` | Phillips-Shi IC value at each iteration (empty unless `stopping=:BIC`) |
| `adf_pvalues` | `Vector{T}` | ADF p-values at each iteration (empty unless `stopping=:ADF`) |
| `T_obs` | `Int` | Number of observations |

---

## Unified Accessors

All filter results inherit from `AbstractFilterResult` and support the `trend()` and `cycle()` accessors, so code that consumes a decomposition never needs to know which filter produced it:

```@example filters
results = ["HP" => hp, "Hamilton" => ham, "BN" => bn, "BK" => bk, "Boosted HP" => bhp]

[(name, length(trend(r)), round(std(cycle(r)), digits=4)) for (name, r) in results]
```

For `BeveridgeNelsonResult`, `trend()` returns the permanent component and `cycle()` returns the transitory component. `HamiltonFilterResult` and `BaxterKingResult` return shortened vectors — 769 and 732 elements against the 804 the other three return — so align them with the original series through their `valid_range` field before comparing dates.

---

## Complete Example

This example applies all five filters to log industrial production from FRED-MD and compares the extracted business cycles:

```@example filters
hp  = hp_filter(y; lambda=129600.0)
ham = hamilton_filter(y; h=24, p=12)
bn  = beveridge_nelson(y)
bk  = baxter_king(y; pl=18, pu=96, K=36)
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)

report(bk)
```

```@example filters
# Cycle amplitude, in log points, for each decomposition
(hamilton = round(std(cycle(ham)), digits=4),
 hp = round(std(cycle(hp)), digits=4),
 bk = round(std(cycle(bk)), digits=4),
 boosted_hp = round(std(cycle(bhp)), digits=4),
 bn = round(std(cycle(bn)), digits=4))
```

Cycle standard deviations span a factor of twenty across the five filters, because each isolates a different frequency range. The Hamilton cycle is largest at ``0.0609`` — it treats everything unforecastable two years ahead as cyclical. The HP (``0.0314``) and Baxter-King (``0.0294``) cycles agree closely, which is reassuring: one penalizes curvature globally and the other targets the 18--96 month band explicitly, yet they recover nearly the same business-cycle amplitude. The boosted HP cycle (``0.0151``) is smaller because 51 iterations move persistent variation into the trend, and the Beveridge-Nelson cycle (``0.0029``) is smallest because the BN identity attributes nearly all movement in industrial production to the permanent component. Report the filter alongside any cycle statistic: the number means nothing without it.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example filters
path = joinpath(mktempdir(), "hp.jld2")
save_model(hp, path)
hp2 = load_model(path)
typeof(hp2)
```

---

## Common Pitfalls

1. **Wrong ``\lambda`` for data frequency.** Using ``\lambda = 1600`` (the quarterly default) on monthly data produces an excessively smooth trend that misses business cycle turning points. Scale ``\lambda`` by the fourth power of the frequency ratio: 6.25 for annual, 1600 for quarterly, 129600 for monthly (Ravn & Uhlig 2002).

2. **Hamilton filter observation loss.** The Hamilton filter loses ``h + p - 1`` observations at the start. For monthly data with ``h = 24``, ``p = 12``, the first 35 observations are unavailable. With a short sample this can eliminate a substantial fraction of the data.

3. **Baxter-King endpoint truncation.** The BK filter loses ``K`` observations at each end (``2K`` total). With ``K = 36`` on monthly data, 6 years are trimmed from the boundaries. Choosing a smaller ``K`` reduces data loss but worsens the approximation to the ideal band-pass filter.

4. **Beveridge-Nelson on I(0) data.** The BN decomposition assumes the series is I(1). Applying it to a stationary series produces a degenerate decomposition where the permanent component absorbs nearly all variation. Verify the unit root assumption with `adf_test` or `kpss_test` before using.

5. **Reading `:statespace` fields as ARIMA quantities.** Under `method=:statespace` the `long_run_multiplier` field is a hard-coded ``1.0`` and `arima_order` reports the cycle AR order as ``(p, 0, 0)``. Neither is an estimate. Use the `:arima` method when the long-run multiplier is the quantity of interest.

6. **Boosted HP stopping criterion choice.** The `:BIC` criterion balances parsimony and fit. The `:ADF` criterion targets cycle stationarity but frequently exhausts `max_iter` on near-unit-root processes — always compare `iterations` against `max_iter`. Use `:fixed` with a known iteration count for replication studies.

7. **HP filter endpoint bias.** The HP filter exhibits spurious cyclicality at sample endpoints (Hamilton 2018). Real-time analysis that depends on the most recent observations should prefer the Hamilton filter or boosted HP, which are more robust at the boundary.

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
  *Review of Economics and Statistics*, 85(2), 235--243. [DOI](https://doi.org/10.1162/003465303765299765)

- Baxter, M., & King, R. G. (1999). Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series.
  *Review of Economics and Statistics*, 81(4), 575--593. [DOI](https://doi.org/10.1162/003465399558454)

- Phillips, P. C. B., & Shi, Z. (2021). Boosting: Why You Can Use the HP Filter.
  *International Economic Review*, 62(2), 521--570. [DOI](https://doi.org/10.1111/iere.12495)

- Mei, Z., Phillips, P. C. B., & Shi, Z. (2024). The Boosted Hodrick-Prescott Filter Is More General Than You Might Think.
  *Journal of Applied Econometrics*, 39(7), 1260--1281. [DOI](https://doi.org/10.1002/jae.3086)
