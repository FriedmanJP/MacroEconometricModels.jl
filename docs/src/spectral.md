# [Spectral Analysis](@id spectral_page)

**MacroEconometricModels.jl** provides a complete spectral analysis toolkit for univariate and bivariate time series. The module covers autocorrelation analysis, nonparametric and parametric spectral density estimation, cross-spectral analysis, frequency-domain filtering, and diagnostic tests for white noise and hidden periodicities.

- **ACF/PACF/CCF**: Sample autocorrelation, partial autocorrelation (Levinson-Durbin or OLS), and cross-correlation with cumulative Ljung-Box Q-statistics
- **Spectral density**: Raw periodogram, Welch's averaged periodogram, kernel-smoothed periodogram, and AR parametric spectrum (Burg's method)
- **Cross-spectrum**: Coherence, phase, and gain from Welch-based cross-spectral density
- **Filtering**: Ideal bandpass filter and transfer function evaluation for the HP, Baxter-King, and Hamilton filters
- **Diagnostics**: Fisher's test for hidden periodicities, Bartlett's cumulative periodogram test, plus the portmanteau tests (Ljung-Box, Box-Pierce, Durbin-Watson)

The three portmanteau tests live here rather than on [Model Diagnostics](@ref tests_diagnostics_page) because they are the time-domain counterpart of the frequency-domain white-noise tests in this module: Ljung-Box and Bartlett's test answer the same question from opposite sides of the Fourier transform. Model Diagnostics covers the residual tests that require a fitted model — ARCH effects, normality, Granger causality, BDS independence.

The filters whose transfer functions this page evaluates are documented on [Time Series Filters](@ref filters_page), and the ACF/PACF correlogram is the standard input to order selection on [ARIMA Models](@ref arima_page).

All results support `show()` for publication-quality tabular output and `plot_result()` for interactive D3.js visualization.

```@setup spectral
using MacroEconometricModels, Random, Statistics
Random.seed!(42)
fred = load_example(:fred_md)
y = filter(isfinite, to_vector(apply_tcode(fred[:, ["INDPRO"]])))
y = y[end-99:end]
y_cpi = filter(isfinite, to_vector(apply_tcode(fred[:, ["CPIAUCSL"]])))
y_cpi = y_cpi[end-99:end]
n = min(length(y), length(y_cpi))
```

## Quick Start

The examples use the last 100 months of two FRED-MD series, each in its `apply_tcode` transformation: `INDPRO` at `tcode=5` is the log difference of industrial production, that is monthly output growth; `CPIAUCSL` at `tcode=6` is the *second* log difference of the price index, that is the monthly change in inflation.

**Recipe 1: ACF/PACF correlogram**

```@example spectral
result = acf_pacf(y; lags=24)
show(stdout, result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/spectral_acf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 2: Spectral density (Welch)**

```@example spectral
sd = spectral_density(y; method=:welch)
show(stdout, sd)
```

```julia
plot_result(sd)
```

```@raw html
<iframe src="../assets/plots/spectral_density.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 3: Cross-spectrum coherence**

```@example spectral
cs = cross_spectrum(y[1:n], y_cpi[1:n])
show(stdout, cs)
```

```julia
plot_result(cs)
```

```@raw html
<iframe src="../assets/plots/spectral_cross.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 4: Ideal bandpass at business-cycle frequencies**

```@example spectral
# Business cycle: 18–96 months (1.5–8 years)
y_bc = ideal_bandpass(y, 2π/96, 2π/18)
round(var(y_bc) / var(y), digits=4)
```

**Recipe 5: Fisher test for hidden periodicities**

```@example spectral
result = fisher_test(y)
show(stdout, result)
```

**Recipe 6: HP filter frequency response**

```@example spectral
tf_hp = transfer_function(:hp; lambda=1600)
show(stdout, tf_hp)
```

```julia
plot_result(tf_hp)
```

```@raw html
<iframe src="../assets/plots/spectral_transfer.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Autocorrelation Functions

The **autocorrelation function** (ACF) measures the linear dependence between ``y_t`` and ``y_{t-k}``:

```math
\hat{\rho}_k = \frac{\sum_{t=k+1}^{n} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n} (y_t - \bar{y})^2}
```

where ``\bar{y}`` is the sample mean. The biased estimator (dividing by ``n``, not ``n-k``) guarantees a positive semi-definite autocovariance matrix.

The **partial autocorrelation function** (PACF) measures the correlation between ``y_t`` and ``y_{t-k}`` after removing the linear effects of ``y_{t-1}, \ldots, y_{t-k+1}``. Two estimation methods are available:

- **Levinson-Durbin** (default): Recursive algorithm using the ACF values. Efficient at ``O(k^2)`` and numerically stable (Brockwell & Davis 1991).
- **OLS**: Regresses ``y_t`` on ``y_{t-1}, \ldots, y_{t-k}`` and extracts the last coefficient. Conceptually transparent but slower at ``O(k^3)`` per lag.

The **cross-correlation function** (CCF) measures the correlation between ``x_{t+k}`` and ``y_t``:

```math
\hat{\rho}_{xy}(k) = \frac{\sum_{t=1}^{n-|k|} (x_{t+k} - \bar{x})(y_t - \bar{y})}{\sqrt{\sum (x_t - \bar{x})^2 \sum (y_t - \bar{y})^2}}
```

Positive lags indicate ``x`` leads ``y``; negative lags indicate ``y`` leads ``x``. `ccf` returns lags from ``-k`` to ``+k``, so its `lags` field has ``2k+1`` entries where the others have ``k``.

!!! note "Ljung-Box Q-Statistics in the Correlogram"
    `acf` and `acf_pacf` compute cumulative Ljung-Box Q-statistics at each lag. The ``k``-th Q-statistic tests ``H_0: \rho_1 = \rho_2 = \cdots = \rho_k = 0`` against the alternative that at least one autocorrelation is non-zero, and `show` displays them in a Stata/EViews-style correlogram. `pacf` and `ccf` return zero-filled `q_stats` and unit `q_pvalues` — they compute no Q-statistics.

```@example spectral
result = acf_pacf(y; lags=24)
show(stdout, result)
```

The correlogram shows a positive first-order autocorrelation of ``0.1714`` followed by a sharp negative second-order value of ``-0.2708``, both outside the ``\pm 0.196`` white-noise band. The PACF cuts off after lag 2 (``0.1714``, then ``-0.3092``, then ``-0.0754``), the signature of a low-order AR process rather than a long memory. The cumulative Q-statistic reaches ``10.66`` at lag 2 with p-value ``0.0048``, so the first two lags carry real information — but by lag 20 the statistic has grown only to ``22.43`` against 20 degrees of freedom (p = ``0.318``). Monthly industrial production growth has a short, sharp dependence structure and nothing beyond it.

### Functions

```julia
result = acf(y; lags=24)           # ACF only
result = pacf(y; lags=24)          # PACF only
result = acf_pacf(y; lags=24)      # Both (efficient single pass)
result = ccf(x, y; lags=24)        # Cross-correlation
```

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `0` | Maximum lag (0 selects ``\min(n-1, \text{round}(10\log_{10} n))``) |
| `method` | `Symbol` | `:levinson` | PACF algorithm: `:levinson` or `:ols`; accepted by `pacf` and `acf_pacf` only |
| `conf_level` | `Real` | `0.95` | Confidence level for the white-noise band |

Passing `method` to `acf` or `ccf` raises a `MethodError` — neither function computes a PACF, so neither accepts the keyword.

### Return Value

**`ACFResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `lags` | `Vector{Int}` | Lag indices (``1{:}k``, or ``-k{:}k`` for `ccf`) |
| `acf` | `Vector{T}` | Autocorrelation values (zeros from `pacf` and `ccf`) |
| `pacf` | `Vector{T}` | Partial autocorrelation values (zeros from `acf` and `ccf`) |
| `ci` | `T` | Confidence interval half-width ``z_{\alpha/2}/\sqrt{n}`` |
| `ccf` | `Union{Nothing,Vector{T}}` | Cross-correlation values (`nothing` except from `ccf`) |
| `q_stats` | `Vector{T}` | Cumulative Ljung-Box Q-statistics |
| `q_pvalues` | `Vector{T}` | P-values of the Q-statistics (``\chi^2(k)``) |
| `nobs` | `Int` | Number of observations |

---

## Spectral Density Estimation

The **spectral density** ``S(\omega)`` decomposes a stationary process into contributions from different frequencies ``\omega \in [0, \pi]``:

```math
S(\omega) = \frac{1}{2\pi} \sum_{k=-\infty}^{\infty} \gamma_k e^{-i\omega k}
```

where ``\gamma_k = \text{Cov}(y_t, y_{t-k})`` is the autocovariance at lag ``k``. The integral of ``S(\omega)`` over ``[-\pi, \pi]`` equals the variance of the process.

Four estimators are available. The raw periodogram has its own entry point, `periodogram`; the three consistent estimators are reached through `spectral_density`.

### Periodogram

The **raw periodogram** is the sample analog of the spectral density:

```math
I(\omega_j) = \frac{1}{2\pi n U} \left| \sum_{t=1}^{n} w_t y_t e^{-i\omega_j t} \right|^2
```

where ``w_t`` is a data window and ``U = n^{-1}\sum w_t^2`` is the window energy normalization. The periodogram is computed via the FFT in ``O(n \log n)`` time.

```@example spectral
I = periodogram(y; window=:hann)
show(stdout, I)
```

The raw periodogram resolves 51 Fourier frequencies from 100 observations, the finest grid any of these estimators offers, and places its peak at ``\omega = 1.1938`` — a period of ``2\pi/1.1938 \approx 5.3`` months. Because each ordinate is a ``\chi^2_2`` variable regardless of ``n``, that peak comes with a 95 percent confidence interval running from ``0.27`` to ``39.5`` times the point estimate, a span of a factor of 146. A single periodogram spike is never evidence on its own.

### Welch's Method

**Welch's averaged modified periodogram** (Welch 1967) reduces variance by averaging periodograms computed from overlapping data segments:

1. Divide the series into ``K`` segments of length ``L`` with overlap fraction ``\alpha``
2. Apply a data window (Hann, Hamming, Blackman, etc.) to each segment
3. Compute the periodogram of each windowed segment
4. Average across all ``K`` periodograms

The variance reduction comes at the cost of reduced frequency resolution. The equivalent degrees of freedom are ``2K``.

```@example spectral
sd_default = spectral_density(y; method=:welch)
sd_long    = spectral_density(y; method=:welch, window=:hann, segment_length=64, overlap=0.5)

(default_frequencies = length(sd_default.freq),
 long_frequencies = length(sd_long.freq),
 default_peak = round(sd_default.freq[argmax(sd_default.density)], digits=4),
 long_peak = round(sd_long.freq[argmax(sd_long.density)], digits=4))
```

The trade-off is visible in one line. The default segment length of ``n/4 = 25`` yields 7 segments and only 13 frequencies, but 14 equivalent degrees of freedom; lengthening segments to 64 buys 33 frequencies at the cost of far fewer segments to average over. Both locate the peak near ``\omega \approx 1.2``–``1.3`` (a 5-month period), so the coarse default is not hiding structure here — but on a series with a narrow spectral peak the default would smear it.

### Kernel-Smoothed Periodogram

The **smoothed periodogram** applies a Daniell kernel to the raw periodogram:

```math
\hat{S}(\omega_k) = \frac{1}{2m+1} \sum_{j=-m}^{m} I(\omega_{k+j})
```

where ``m`` is the kernel half-width (bandwidth). Larger bandwidth reduces variance but increases bias. The equivalent degrees of freedom are ``2(2m+1)``.

```@example spectral
sd_smooth = spectral_density(y; method=:smoothed, bandwidth=7)
show(stdout, sd_smooth)
```

Smoothing preserves all 51 raw frequencies while averaging each ordinate over its 15 nearest neighbours, giving 30 equivalent degrees of freedom against the periodogram's 2. Note the default window for `:smoothed` is `:rectangular`, not the `:hann` that `:welch` uses: the smoothing already controls variance, so no taper is applied unless one is requested.

### AR Parametric Spectrum

The **AR parametric spectrum** (Burg 1968) fits an autoregressive model and evaluates its theoretical spectral density:

```math
S(\omega) = \frac{\hat{\sigma}^2}{2\pi \left| 1 + \sum_{j=1}^{p} \hat{a}_j e^{-i\omega j} \right|^2}
```

The AR order ``p`` is selected by AIC over ``1 \le p \le \min(\text{round}(10\log_{10} n),\, n/3)`` unless `order` is given. Burg's algorithm produces stable AR coefficient estimates.

```@example spectral
sd_ar12 = spectral_density(y; method=:ar, order=12)
sd_auto = spectral_density(y; method=:ar)

# Effective bandwidth is 2πp/n, so it reveals the selected order
(order_12_bandwidth = round(sd_ar12.bandwidth, digits=4),
 auto_bandwidth = round(sd_auto.bandwidth, digits=4),
 auto_peak = round(sd_auto.freq[argmax(sd_auto.density)], digits=4))
```

AIC selects ``p = 1``: the effective bandwidth of ``0.0628 = 2\pi/100`` corresponds to a single estimated coefficient, against ``0.7540 = 24\pi/100`` for the imposed AR(12). The AR(1) spectrum peaks at ``\omega = 0``, monotonically declining thereafter, because the fitted first-order coefficient is positive. This is where the parametric approach earns its keep and where it misleads: an AR(1) is a two-parameter summary of the whole spectrum, so it produces a clean curve, but it cannot represent the negative second-order dependence the PACF found, and it therefore misses the mid-frequency peak every nonparametric estimator reports.

### Method Comparison

| Method | Entry point | Variance | Resolution | Best for |
|--------|-------------|----------|------------|----------|
| Periodogram | `periodogram` | High | Highest | Exploratory work, long series |
| Welch | `spectral_density(:welch)` | Low | Moderate | General-purpose default |
| Smoothed | `spectral_density(:smoothed)` | Low | Moderate | Smooth spectral shape |
| AR | `spectral_density(:ar)` | Lowest | Highest | Sharp peaks, short series |

### Keywords

`periodogram` accepts `window` and `conf_level` only. `spectral_density` accepts `method`, `conf_level`, and the keywords its chosen method uses; keywords belonging to other methods are ignored.

| Keyword | Type | Default | Applies to | Description |
|---------|------|---------|-----------|-------------|
| `method` | `Symbol` | `:welch` | `spectral_density` | `:welch`, `:smoothed`, or `:ar` |
| `window` | `Symbol` | `:hann` (`:rectangular` for `periodogram` and `:smoothed`) | all but `:ar` | Data window (`:rectangular`, `:bartlett`, `:hann`, `:hamming`, `:blackman`, `:tukey`, `:flat_top`) |
| `segment_length` | `Int` | `0` (selects ``\max(16, n/4)``) | `:welch` | Segment length for Welch averaging |
| `overlap` | `Real` | `0.5` | `:welch` | Overlap fraction, ``\in [0, 1)`` |
| `bandwidth` | `Int` | `0` (selects ``\text{round}(\sqrt{n})``) | `:smoothed` | Daniell kernel half-width |
| `order` | `Int` | `0` (selects by AIC) | `:ar` | AR order |
| `n_freq` | `Int` | `256` | `:ar` | Frequency grid points |
| `conf_level` | `Real` | `0.95` | all | Confidence level for the ``\chi^2`` spectral bounds |

!!! warning "`:periodogram` is not a `spectral_density` method"
    `spectral_density(y; method=:periodogram)` throws an `ArgumentError`. The raw periodogram is a separate function, `periodogram(y)`, because it is inconsistent — its variance does not shrink with ``n`` — and `spectral_density` is the entry point for the consistent estimators.

### Return Value

**`SpectralDensityResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `freq` | `Vector{T}` | Frequency grid in ``[0, \pi]`` |
| `density` | `Vector{T}` | Estimated spectral density |
| `ci_lower` | `Vector{T}` | Lower confidence bound |
| `ci_upper` | `Vector{T}` | Upper confidence bound |
| `method` | `Symbol` | Estimation method used |
| `bandwidth` | `T` | Effective bandwidth |
| `nobs` | `Int` | Number of observations |

---

## Cross-Spectral Analysis

The **cross-spectral density** between two stationary processes ``x_t`` and ``y_t`` decomposes their linear association by frequency:

```math
S_{xy}(\omega) = C_{xy}(\omega) - i Q_{xy}(\omega)
```

where:

- ``C_{xy}(\omega)`` is the **co-spectrum** (real part) --- the in-phase association at frequency ``\omega``
- ``Q_{xy}(\omega)`` is the **quadrature spectrum** (negative imaginary part) --- the out-of-phase association

Three derived quantities summarize the relationship:

- **Squared coherence**: ``\kappa^2_{xy}(\omega) = |S_{xy}(\omega)|^2 / (S_{xx}(\omega) S_{yy}(\omega)) \in [0, 1]`` --- the frequency-domain analog of ``R^2``
- **Phase**: ``\phi_{xy}(\omega) = \text{atan2}(Q_{xy}, C_{xy})`` --- the lead-lag relationship in radians
- **Gain**: ``G_{xy}(\omega) = |S_{xy}(\omega)| / S_{xx}(\omega)`` --- the amplitude ratio

Estimation uses the same Welch segmentation as `spectral_density`, so all three quantities inherit its segment-length trade-off.

!!! warning "Coherence needs more than one segment"
    With a single segment there is nothing to average over and squared coherence is identically ``1`` at every frequency. `cross_spectrum` warns when this happens. Reduce `segment_length` or raise `overlap` so at least a handful of segments contribute.

```@example spectral
cs = cross_spectrum(y[1:n], y_cpi[1:n]; window=:hann)
show(stdout, cs)
```

```julia
plot_result(cs)
```

```@raw html
<iframe src="../assets/plots/spectral_cross.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# Convenience accessors: each refits the cross-spectrum and returns (freq, values)
freq, coh = coherence(y[1:n], y_cpi[1:n])
freq, ph  = phase(y[1:n], y_cpi[1:n])
freq, g   = gain(y[1:n], y_cpi[1:n])

(peak_coherence = round(maximum(coh), digits=4),
 peak_frequency = round(freq[argmax(coh)], digits=4),
 peak_period_months = round(2π / freq[argmax(coh)], digits=1),
 phase_at_peak = round(ph[argmax(coh)], digits=4))
```

Squared coherence between output growth and the change in inflation peaks at ``0.744`` at ``\omega = 0.7854``, exactly an 8-month period — at that frequency roughly three quarters of the variation in one series is linearly explained by the other. The phase there is ``-1.0934`` radians; since `cross_spectrum(x, y)` returns ``-\omega d`` when ``x`` leads ``y`` by ``d`` periods, dividing by the frequency gives ``1.0934/0.7854 \approx 1.4`` months of output growth leading price acceleration — the ordering a short-run Phillips curve implies. Coherence at the lowest frequencies is much weaker (``0.162`` at ``\omega = 0``), so this is a business-cycle-horizon relationship, not a long-run one.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `window` | `Symbol` | `:hann` | Data window |
| `segment_length` | `Int` | `0` (selects ``\max(16, n/4)``) | Segment length for Welch averaging |
| `overlap` | `Real` | `0.5` | Overlap fraction |

### Return Value

**`CrossSpectrumResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `freq` | `Vector{T}` | Frequency grid in ``[0, \pi]`` |
| `co_spectrum` | `Vector{T}` | Co-spectrum ``C_{xy}(\omega)`` |
| `quad_spectrum` | `Vector{T}` | Quadrature spectrum ``Q_{xy}(\omega)`` |
| `coherence` | `Vector{T}` | Squared coherence ``\kappa^2_{xy}(\omega)`` |
| `phase` | `Vector{T}` | Phase spectrum (radians) |
| `gain` | `Vector{T}` | Gain (amplitude ratio) |
| `nobs` | `Int` | Number of observations |

---

## Spectral Diagnostics

### Fisher's Test for Hidden Periodicities

Fisher's exact test (Fisher 1929) detects a single dominant periodic component in a time series:

```math
g = \frac{\max_j I(\omega_j)}{\sum_j I(\omega_j)}
```

where ``I(\omega_j)`` is the periodogram at Fourier frequency ``\omega_j``. Under ``H_0`` (white noise), ``g`` has an exact distribution. A large ``g`` indicates a hidden periodicity at the peak frequency.

```@example spectral
result = fisher_test(y)
show(stdout, result)
```

Fisher's ``g`` is ``0.0592`` with p-value ``0.9787``: the largest of the 49 periodogram ordinates accounts for under 6 percent of total power, which is unremarkable when the expected share under white noise is ``1/49 \approx 2`` percent and the maximum of 49 exponential variables routinely runs several times its mean. There is no deterministic cycle in monthly industrial production growth — the peak frequency of ``1.1310`` (a 5.6-month period) is a sampling artifact, not a business cycle.

### Bartlett's Cumulative Periodogram Test

Bartlett's test (Bartlett 1955) checks whether the cumulative normalized periodogram follows the uniform distribution expected under white noise:

```math
D = \max_j \left| \frac{\sum_{k=1}^{j} I(\omega_k)}{\sum_{k=1}^{m} I(\omega_k)} - \frac{j}{m} \right|
```

The Kolmogorov-Smirnov statistic ``D`` measures the maximum departure from uniformity. Rejection indicates the series is not white noise.

```@example spectral
result = bartlett_white_noise_test(y)
show(stdout, result)
```

Bartlett's ``D = 0.1862`` carries a p-value of ``0.0669`` — rejection at the 10 percent level but not at 5 percent. This is exactly what the correlogram predicted: the spectrum is tilted away from flat by the AR(2)-like dependence at the first two lags, but the departure is too modest for 100 observations to resolve decisively. Fisher's test looks for one dominant frequency and finds nothing; Bartlett's test looks for any systematic tilt across all frequencies and finds a hint. The two tests are complementary, not redundant.

### Band Power

`band_power` integrates the estimated spectral density over a frequency band by the trapezoidal rule, interpolating the density linearly at the band edges:

```@example spectral
sd = spectral_density(y; method=:welch)

total = band_power(sd, 0.0, π)
business_cycle = band_power(sd, 2π/96, 2π/18)   # 18–96 months
high_frequency = band_power(sd, 2π/6, π)        # under 6 months

(business_cycle_share = round(business_cycle / total, digits=4),
 high_frequency_share = round(high_frequency / total, digits=4))
```

Only ``5.0`` percent of the variance of monthly output growth sits in the 18--96 month business-cycle band, against ``65.7`` percent at periods shorter than 6 months. This is a property of growth rates, not of the business cycle: differencing multiplies the spectrum by ``|1 - e^{-i\omega}|^2 = 2(1 - \cos\omega)``, which is near zero at low frequencies and maximal at ``\omega = \pi``. Run the same decomposition on the *level* of log industrial production, or filter first, before concluding anything about cyclical importance.

---

## Portmanteau Tests

Three classical serial correlation tests complement the spectral diagnostics. Where Fisher's and Bartlett's tests work on the periodogram, these work directly on the sample autocorrelations.

### Ljung-Box Q Test

The Ljung-Box test (Ljung & Box 1978) checks for autocorrelation up to lag ``h``:

```math
Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(h - p)
```

where ``p`` is the number of fitted AR/MA parameters (set via `fitdf`).

```@example spectral
result = ljung_box_test(y; lags=20, fitdf=0)
show(stdout, result)
```

### Box-Pierce Q Test

The original Box-Pierce test (Box & Pierce 1970) uses the simpler statistic ``Q_0 = n \sum \hat{\rho}_k^2``. The Ljung-Box modification is preferred for small samples.

```@example spectral
result = box_pierce_test(y; lags=20)
show(stdout, result)
```

Ljung-Box gives ``Q = 22.43`` (p = ``0.318``) and Box-Pierce ``Q_0 = 20.82`` (p = ``0.408``) on the same 20 lags. The gap between the two statistics is the ``n(n+2)/(n-k)`` correction, worth about 8 percent here and growing with ``h/n``; at 20 lags on 100 observations it is already large enough to matter, which is why the Ljung-Box form is the one to report. Both fail to reject, because spreading two genuinely informative lags across 20 degrees of freedom dilutes the evidence — the cumulative Q-statistic at lag 2 in the correlogram above rejects at the 1 percent level.

### Durbin-Watson Test

The Durbin-Watson test (Durbin & Watson 1950) detects first-order autocorrelation in regression residuals:

```math
DW = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2} \approx 2(1 - \hat{\rho}_1)
```

Values near 2 indicate no autocorrelation; values near 0 indicate positive autocorrelation; values near 4 indicate negative autocorrelation.

```@example spectral
dw_result = durbin_watson_test(y)
show(stdout, dw_result)
```

``DW = 1.6564`` implies ``\hat{\rho}_1 \approx 1 - DW/2 = 0.172``, within ``0.0005`` of the ``0.1714`` first-order autocorrelation the correlogram reports. The p-value of ``0.0858`` marks it as borderline. Focusing on lag 1 alone recovers the significance the 20-lag portmanteau tests dilute away — the cost is blindness to everything at longer lags.

---

## Frequency-Domain Filtering

### Ideal Bandpass Filter

The **ideal bandpass filter** retains frequency components in ``[\omega_l, \omega_h]`` by zeroing all other Fourier coefficients, then inverting the transform. The series is demeaned first, so the output is mean-zero regardless of the band.

!!! warning "Gibbs Phenomenon"
    The ideal bandpass filter applies a sharp cutoff in the frequency domain, which produces ringing artifacts (Gibbs phenomenon) in the time domain. For applied work, the Baxter-King or HP filters provide smoother alternatives. The ideal bandpass is useful for quick exploratory analysis or when exact frequency isolation is needed.

```@example spectral
y_bc = ideal_bandpass(y, 2π/96, 2π/18)   # business cycle: 18–96 months
y_hf = ideal_bandpass(y, 2π/18, π)       # high frequency: under 18 months

(business_cycle_share = round(var(y_bc) / var(y), digits=4),
 high_frequency_share = round(var(y_hf) / var(y), digits=4))
```

The two bands partition the variance almost exactly: ``4.4`` percent in the business-cycle band and ``95.5`` percent above it, summing to ``99.8`` percent with the remainder in the sub-96-month tail. The business-cycle share here (``0.0438``) is close to the ``0.0502`` that `band_power` reported from the Welch spectrum, and the difference is instructive — `band_power` integrates a smoothed 13-point estimate while `ideal_bandpass` partitions the raw Fourier coefficients exactly.

### Transfer Function

`transfer_function` computes the frequency response (gain and phase) of three standard macroeconomic filters. Each is evaluated from its analytical or weight-based representation, not from filtering data.

```@example spectral
tf_hp  = transfer_function(:hp; lambda=1600)
tf_bk  = transfer_function(:bk; K=12)
tf_ham = transfer_function(:hamilton; h=8)

(hp_max_gain = round(maximum(tf_hp.gain), digits=4),
 bk_max_gain = round(maximum(tf_bk.gain), digits=4),
 hamilton_max_gain = round(maximum(tf_ham.gain), digits=4))
```

```julia
plot_result(tf_hp)
```

```@raw html
<iframe src="../assets/plots/spectral_transfer.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The three maxima say what each filter does. The HP cyclical gain approaches ``1.0`` from below and never exceeds it — it is a genuine high-pass filter that only ever attenuates. The Baxter-King gain overshoots to ``1.0964`` inside its passband, the ripple that finite truncation at ``K = 12`` leaves behind. The Hamilton response reaches ``2.0``, meaning it *amplifies* frequencies whose period divides ``2h``: its transfer is ``1 - e^{-i\omega h}``, which doubles a component exactly out of phase over ``h`` periods. A filter with gain above one is not extracting a component of the data; it is manufacturing amplitude.

The HP penalty falls on the **second** difference of the trend, whose squared transfer modulus is ``16\sin^4(\omega/2)``, so the cyclical (high-pass) gain is a ``\sin^4`` --- not ``\sin^2`` --- response:

```math
G(\omega) = \frac{16\lambda \sin^4(\omega/2)}{1 + 16\lambda \sin^4(\omega/2)}
         = \frac{4\lambda (1 - \cos\omega)^2}{1 + 4\lambda (1 - \cos\omega)^2}
```

At ``\lambda = 1600`` the half-power point (``G = 0.5``) occurs at ``\omega^* = 2\arcsin\big((16\lambda)^{-1/4}\big)``, a period of ``\approx 39.7`` quarters (``\approx 9.8`` quarters at ``\lambda = 6.25``).

!!! note "Fixed Baxter-King band and simplified Hamilton response"
    The `:bk` response is evaluated at the standard 6--32 quarter business-cycle band; only `K` is adjustable, so it will not match a `baxter_king` call made with different `pl` and `pu`. The `:hamilton` response uses the random-walk benchmark ``1 - e^{-i\omega h}`` rather than the transfer implied by the fitted ``p``-lag regression, so it depends on `h` alone.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lambda` | `Real` | `1600` | HP smoothing parameter (`:hp` only) |
| `K` | `Int` | `12` | Baxter-King half-window length (`:bk` only) |
| `h` | `Int` | `8` | Hamilton regression horizon (`:hamilton` only) |
| `n_freq` | `Int` | `256` | Number of frequency grid points |

### Return Value

**`TransferFunctionResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `freq` | `Vector{T}` | Frequency grid in ``[0, \pi]`` |
| `gain` | `Vector{T}` | Gain (amplitude) at each frequency |
| `phase` | `Vector{T}` | Phase shift (radians) at each frequency; identically zero for `:hp` |
| `filter` | `Symbol` | Filter type (`:hp`, `:bk`, `:hamilton`) |

---

## Complete Example

A full spectral analysis of U.S. industrial production growth, from the correlogram through the frequency-domain decomposition to the diagnostic tests.

```@example spectral
# 1. Correlogram: ACF + PACF with Ljung-Box Q-stats
corr = acf_pacf(y; lags=24)
show(stdout, corr)
```

```julia
plot_result(corr)
```

```@raw html
<iframe src="../assets/plots/spectral_acf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# 2. Spectral density: Welch's method with a Hann window
sd = spectral_density(y; method=:welch, window=:hann)
show(stdout, sd)
```

```julia
plot_result(sd)
```

```@raw html
<iframe src="../assets/plots/spectral_density.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# 3. Where the variance sits, by frequency band
total = band_power(sd, 0.0, π)

(business_cycle = round(band_power(sd, 2π/96, 2π/18) / total, digits=4),
 medium_run = round(band_power(sd, 2π/18, 2π/6) / total, digits=4),
 high_frequency = round(band_power(sd, 2π/6, π) / total, digits=4))
```

```@example spectral
# 4. Cross-spectrum: output growth against the change in inflation
cs = cross_spectrum(y[1:n], y_cpi[1:n])

(peak_coherence = round(maximum(cs.coherence), digits=4),
 peak_period_months = round(2π / cs.freq[argmax(cs.coherence)], digits=1))
```

```@example spectral
# 5. White-noise diagnostics from both domains
(fisher_p = round(fisher_test(y).pvalue, digits=4),
 bartlett_p = round(bartlett_white_noise_test(y).pvalue, digits=4),
 ljung_box_p = round(ljung_box_test(y; lags=20).pvalue, digits=4))
```

The five steps tell one story. The correlogram finds significant dependence concentrated at the first two lags and nothing beyond. The Welch spectrum places ``5.0`` percent of variance in the business-cycle band, ``28.5`` percent between 6 and 18 months, and ``65.7`` percent below 6 months — the tilt toward high frequencies that differencing guarantees. Coherence with the change in inflation peaks at ``0.744`` at an 8-month period, so whatever links output and prices operates at business-cycle rather than long-run horizons. All three white-noise tests fail to reject at the 5 percent level (``0.9787``, ``0.0669``, ``0.3178``), which is the correct reading: monthly output growth is close to, but not quite, unforecastable.

---

## Common Pitfalls

1. **Non-stationary input.** Spectral density estimation assumes stationarity. Apply `diff()` or a filter to trending series before computing the periodogram; the ACF of a non-stationary series decays slowly, producing a spectral density concentrated near ``\omega = 0``.

2. **Reading band shares off differenced data.** Differencing multiplies the spectrum by ``2(1 - \cos\omega)``, which annihilates low frequencies. The 5 percent business-cycle share found above is a property of the growth rate, not evidence that business cycles are unimportant. Compute band shares on levels, or undo the transfer function of the difference operator.

3. **Periodogram variance.** The raw periodogram is inconsistent — its variance does not decrease with sample size, and `spectral_density(y; method=:periodogram)` deliberately throws rather than let it be used as one. Use `:welch` or `:smoothed`.

4. **Window choice affects resolution.** Data windows (Hann, Hamming, Blackman) trade main-lobe width for sidelobe suppression. Use `:rectangular` for maximum frequency resolution, `:blackman` for maximum sidelobe suppression. Note that `periodogram` and `:smoothed` default to `:rectangular` while `:welch` and `cross_spectrum` default to `:hann`.

5. **Frequency units.** All frequencies are in radians per observation, ``\omega \in [0, \pi]``. To convert to period: ``T = 2\pi/\omega``. For monthly data, ``\omega = 2\pi/12`` corresponds to a 12-month (annual) cycle.

6. **Ljung-Box `fitdf`.** When testing residuals from an ARMA(p,q) model, set `fitdf=p+q` to adjust the degrees of freedom. Omitting this inflates the test size.

7. **Too many lags dilute the portmanteau tests.** Q-statistics spread the evidence over ``h`` degrees of freedom. Two informative lags out of 20 will not reject even when each is individually significant — read the cumulative Q column of the correlogram rather than a single high-``h`` test.

---

## References

- Bartlett, M. S. (1955). *An Introduction to Stochastic Processes*. Cambridge: Cambridge University Press.

- Box, G. E. P., & Pierce, D. A. (1970). Distribution of Residual Autocorrelations in Autoregressive-Integrated Moving Average Time Series Models. *Journal of the American Statistical Association*, 65(332), 1509--1526. [DOI](https://doi.org/10.1080/01621459.1970.10481180)

- Brockwell, P. J., & Davis, R. A. (1991). *Time Series: Theory and Methods* (2nd ed.). New York: Springer. ISBN 978-1-4419-0319-8.

- Brillinger, D. R. (1981). *Time Series: Data Analysis and Theory*. Expanded ed. San Francisco: Holden-Day. ISBN 978-0-8162-1150-0.

- Burg, J. P. (1968). A New Analysis Technique for Time Series Data. *NATO Advanced Study Institute on Signal Processing*, Enschede, Netherlands.

- Durbin, J., & Watson, G. S. (1950). Testing for Serial Correlation in Least Squares Regression. I. *Biometrika*, 37(3/4), 409--428. [DOI](https://doi.org/10.2307/2332391)

- Fisher, R. A. (1929). Tests of Significance in Harmonic Analysis. *Proceedings of the Royal Society of London A*, 125(796), 54--59. [DOI](https://doi.org/10.1098/rspa.1929.0151)

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Ljung, G. M., & Box, G. E. P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297--303. [DOI](https://doi.org/10.1093/biomet/65.2.297)

- Priestley, M. B. (1981). *Spectral Analysis and Time Series*. London: Academic Press. ISBN 978-0-12-564922-3.

- Welch, P. D. (1967). The Use of Fast Fourier Transform for the Estimation of Power Spectra: A Method Based on Time Averaging Over Short, Modified Periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70--73. [DOI](https://doi.org/10.1109/TAU.1967.1161901)
