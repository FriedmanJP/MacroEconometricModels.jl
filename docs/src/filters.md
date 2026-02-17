# Time Series Filters

Macroeconomic time series are often decomposed into trend and cyclical components. This module provides five standard filters used in applied macroeconomics for trend-cycle decomposition:

| Filter | Key Idea | Observations Lost |
|--------|----------|-------------------|
| **Hodrick-Prescott** | Penalized least squares (smoothing spline) | None |
| **Hamilton** | OLS regression on lagged values | `h + p - 1` at start |
| **Beveridge-Nelson** | ARIMA-based permanent/transitory decomposition | None |
| **Baxter-King** | Symmetric band-pass moving average | `2K` (K at each end) |
| **Boosted HP** | Iterated HP with data-driven stopping | None |

## Quick Start

```julia
using MacroEconometricModels

# Log industrial production — a trending I(1) monthly series
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Hodrick-Prescott filter (monthly data)
hp = hp_filter(y; lambda=129600.0)

# Hamilton (2018) regression filter (monthly: h=24, p=12)
ham = hamilton_filter(y; h=24, p=12)

# Beveridge-Nelson decomposition
bn = beveridge_nelson(y)

# Baxter-King band-pass filter (monthly: 18–96 month band, K=36)
bk = baxter_king(y; pl=18, pu=96, K=36)

# Boosted HP filter
bhp = boosted_hp(y; lambda=129600.0, stopping=:BIC)

# Unified accessors
trend(hp)   # trend component
cycle(hp)   # cyclical component
```

---

## Hodrick-Prescott Filter

The HP filter (Hodrick & Prescott 1997) decomposes a time series ``y_t`` into trend ``\tau_t`` and cycle ``c_t = y_t - \tau_t`` by solving:

```math
\min_{\tau} \sum_{t=1}^T (y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1} (\tau_{t+1} - 2\tau_t + \tau_{t-1})^2
```

where ``\lambda`` controls the smoothness of the trend. The first term penalizes deviations of the trend from the data, while the second penalizes curvature in the trend.

### Choosing ``\lambda``

| Data Frequency | Recommended ``\lambda`` |
|----------------|------------------------|
| Annual | 6.25 |
| Quarterly | 1,600 |
| Monthly | 129,600 |

The quarterly value of 1,600 proposed by Hodrick and Prescott has become standard, though Ravn and Uhlig (2002) provide a frequency-based justification.

### Implementation

The solution is ``\tau = (I + \lambda D'D)^{-1} y`` where ``D`` is the ``(T-2) \times T`` second-difference matrix. The implementation builds a sparse pentadiagonal system and solves via Cholesky factorization, giving ``O(T)`` computational cost.

```julia
# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly filter (lambda=129600)
hp = hp_filter(y; lambda=129600.0)
hp.trend   # smooth trend
hp.cycle   # cyclical deviations — the HP cycle captures business cycle fluctuations

# Annual data would use lambda=6.25
hp_annual = hp_filter(y; lambda=6.25)
```

!!! note "Technical Note"
    The HP filter has known issues with spurious cyclicality at sample endpoints and can induce spurious dynamic relations. Hamilton (2018) provides a detailed critique and proposes an alternative (see below).

---

## Hamilton Filter

Hamilton (2018) proposes a regression-based alternative to the HP filter. Instead of imposing smoothness, it regresses the future value ``y_{t+h}`` on current and lagged values:

```math
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_t
```

The fitted values form the trend and the residuals form the cycle. The default parameters ``h = 8, p = 4`` correspond to a 2-year ahead projection using 4 quarterly lags.

### Advantages over HP

- Does not require choosing a smoothing parameter
- Avoids spurious cyclicality
- Does not induce spurious dynamic relations between filtered series
- Robust to unit roots and structural breaks

### Implementation

```julia
# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly parameters (h=24, p=12): 2-year horizon, 12 monthly lags
ham = hamilton_filter(y; h=24, p=12)
ham.trend       # fitted values (length T - h - p + 1)
ham.cycle       # residuals — the Hamilton filter avoids endpoint bias
ham.beta        # OLS coefficients
ham.valid_range # indices into original series
```

!!! warning
    The Hamilton filter loses `h + p - 1` observations at the start of the sample. For monthly data with `h=24, p=12`, this is 35 observations. Plan accordingly with short samples.

---

## Beveridge-Nelson Decomposition

The Beveridge-Nelson (1981) decomposition separates an I(1) process into a permanent (random walk with drift) component and a stationary transitory component. It exploits the Wold representation of the first-differenced series:

```math
\Delta y_t = \mu + \psi(L) \varepsilon_t = \mu + \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}
```

where ``\psi_0 = 1``. The long-run multiplier is ``\psi(1) = 1 + \sum_{j=1}^{\infty} \psi_j``, and the decomposition is:

```math
y_t = \tau_t + c_t
```

where ``\tau_t`` is a random walk with drift ``\mu \cdot \psi(1)`` (permanent component) and ``c_t`` is a mean-zero stationary process (transitory component).

### Implementation

The function fits an ARMA model to ``\Delta y_t``, computes the ``\psi``-weights from the MA(``\infty``) representation, and constructs the transitory component.

```julia
# Log industrial production — an I(1) series suitable for BN decomposition
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Automatic ARMA order selection for Δy
bn = beveridge_nelson(y)
bn.permanent    # random walk + drift (trend)
bn.transitory   # stationary cycle
bn.drift        # estimated drift
bn.long_run_multiplier  # ψ(1)
bn.arima_order  # (p, 1, q) used

# Manual ARMA order
bn2 = beveridge_nelson(y; p=2, q=1)
```

!!! note "Technical Note"
    The BN decomposition assumes the series is I(1). Apply unit root tests (`adf_test`, `kpss_test`) to verify this assumption before using. For I(0) series, the decomposition is degenerate.

---

## Baxter-King Band-Pass Filter

The Baxter-King (1999) filter isolates cyclical fluctuations in a specified frequency band ``[2\pi/p_u, 2\pi/p_l]`` using a symmetric finite moving average approximation to the ideal band-pass filter.

### Ideal Band-Pass Weights

The ideal (infinite) band-pass filter has weights:
```math
B_0 = \frac{\omega_H - \omega_L}{\pi}, \quad B_j = \frac{\sin(\omega_H j) - \sin(\omega_L j)}{\pi j}
```

where ``\omega_H = 2\pi/p_l`` and ``\omega_L = 2\pi/p_u``.

### Truncation and Adjustment

The ideal filter is truncated at lag ``K`` and adjusted to ensure the weights sum to zero (eliminating stochastic trends):
```math
a_j = B_j + \theta, \quad \theta = -\frac{B_0 + 2\sum_{j=1}^K B_j}{2K + 1}
```

The filtered series is:
```math
c_t = a_0 y_t + \sum_{j=1}^K a_j (y_{t-j} + y_{t+j})
```

### Implementation

```julia
# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# Monthly business cycle band: 18–96 months (1.5–8 years), K=36
bk = baxter_king(y; pl=18, pu=96, K=36)
bk.cycle       # band-pass filtered (business cycle component)
bk.trend       # residual (low + high frequency)
bk.weights     # [a_0, a_1, ..., a_K]
bk.valid_range # K+1 : T-K
```

The filter weights sum to zero by construction:
```julia
w = bk.weights
total = w[1] + 2 * sum(w[2:end])  # ≈ 0
```

!!! warning
    The BK filter loses ``K`` observations at each end (``2K`` total). With ``K = 36`` and monthly data, this is 6 years of data at the boundaries.

---

## Boosted HP Filter

Phillips and Shi (2021) propose iterating the HP filter on the cyclical component to improve trend estimation when the data contains stochastic trends. The key insight is that a single HP pass may leave unit root behavior in the cycle; re-filtering removes it.

### Algorithm

1. Apply HP filter: ``\hat{\tau}^{(1)} = S \cdot y``, ``\hat{c}^{(1)} = (I - S) \cdot y``
2. Re-filter the cycle: ``\hat{c}^{(m)} = (I - S) \hat{c}^{(m-1)}``
3. Stop when a data-driven criterion is met
4. Final trend: ``\hat{\tau} = y - \hat{c}^{(m^*)}``

where ``S = (I + \lambda D'D)^{-1}``.

### Stopping Criteria

Three stopping rules are available:

- **`:BIC`** (default) — Fit AR(1) to the cycle at each iteration; stop when BIC increases. Selects the iteration that best balances parsimony and fit.
- **`:ADF`** — Run the ADF test on the cycle; stop when the null of a unit root is rejected at significance level `sig_p`. Ensures the cycle is stationary.
- **`:fixed`** — Run exactly `max_iter` iterations. Useful for comparison or when the other criteria are too conservative.

### Implementation

```julia
# Log industrial production (monthly)
fred = load_example(:fred_md)
y = filter(isfinite, log.(fred[:, "INDPRO"]))

# BIC stopping (default), monthly lambda
bhp = boosted_hp(y; lambda=129600.0)
bhp.trend       # boosted trend
bhp.cycle       # boosted cycle
bhp.iterations  # number of iterations used
bhp.bic_path    # BIC at each iteration

# ADF stopping
bhp_adf = boosted_hp(y; lambda=129600.0, stopping=:ADF, sig_p=0.05)
bhp_adf.adf_pvalues  # p-values at each iteration

# Fixed iterations
bhp_fixed = boosted_hp(y; lambda=129600.0, stopping=:fixed, max_iter=5)
```

---

## Unified Accessors

All filter results support the `trend()` and `cycle()` accessors:

```julia
# Works for any AbstractFilterResult
for r in [hp, ham, bn, bk, bhp]
    t = trend(r)  # trend component
    c = cycle(r)  # cyclical component
end
```

For `BeveridgeNelsonResult`, `trend()` returns the permanent component and `cycle()` returns the transitory component.

---

## Comparison Example

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

# Compare cycle standard deviations
println("Cycle standard deviations:")
println("  HP:       ", round(std(cycle(hp)), digits=4))
println("  Hamilton: ", round(std(cycle(ham)), digits=4))
println("  BN:       ", round(std(cycle(bn)), digits=4))
println("  BK:       ", round(std(cycle(bk)), digits=4))
println("  bHP:      ", round(std(cycle(bhp)), digits=4))
```

## References

- Hodrick, Robert J., and Edward C. Prescott. 1997. "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking* 29 (1): 1--16. [https://doi.org/10.2307/2953682](https://doi.org/10.2307/2953682)
- Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics* 100 (5): 831--843. [https://doi.org/10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Beveridge, Stephen, and Charles R. Nelson. 1981. "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151--174. [https://doi.org/10.1016/0304-3932(81)90040-4](https://doi.org/10.1016/0304-3932(81)90040-4)
- Baxter, Marianne, and Robert G. King. 1999. "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics* 81 (4): 575--593. [https://doi.org/10.1162/003465399558454](https://doi.org/10.1162/003465399558454)
- Phillips, Peter C. B., and Zhentao Shi. 2021. "Boosting: Why You Can Use the HP Filter." *International Economic Review* 62 (2): 521--570. [https://doi.org/10.1111/iere.12495](https://doi.org/10.1111/iere.12495)
- Mei, Ziwei, Peter C. B. Phillips, and Zhentao Shi. 2024. "The boosted HP filter is more general than you might think." *Journal of Applied Econometrics* 39 (7): 1260--1281. [https://doi.org/10.1002/jae.3086](https://doi.org/10.1002/jae.3086)
