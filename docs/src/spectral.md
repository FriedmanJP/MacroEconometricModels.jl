# [Spectral Analysis](@id spectral_page)

**MacroEconometricModels.jl** provides a complete spectral analysis toolkit for univariate and bivariate time series. The module covers autocorrelation analysis, nonparametric and parametric spectral density estimation, cross-spectral analysis, frequency-domain filtering, and diagnostic tests for white noise and hidden periodicities.

- **ACF/PACF/CCF**: Sample autocorrelation, partial autocorrelation (Levinson-Durbin or OLS), and cross-correlation with cumulative Ljung-Box Q-statistics
- **Spectral Density**: Periodogram, Welch's averaged periodogram, kernel-smoothed periodogram, and AR parametric spectrum (Burg's method)
- **Cross-Spectrum**: Coherence, phase, and gain from Welch-based cross-spectral density
- **Filtering**: Ideal bandpass filter and transfer function evaluation for HP, Baxter-King, and Hamilton filters
- **Diagnostics**: Fisher's test for hidden periodicities, Bartlett's cumulative periodogram test, plus portmanteau tests (Ljung-Box, Box-Pierce, Durbin-Watson)

All results support `report()` for publication-quality tabular output and `plot_result()` for interactive D3.js visualization.

```@setup spectral
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
y = filter(isfinite, to_vector(apply_tcode(fred[:, "INDPRO"])))
y = y[end-99:end]
```

## Quick Start

**Recipe 1: ACF/PACF correlogram**

```@example spectral
result = acf_pacf(y; lags=24)
report(result)
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
report(sd)
```

```julia
plot_result(sd)
```

```@raw html
<iframe src="../assets/plots/spectral_density.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 3: Cross-spectrum coherence**

```@example spectral
y_cpi = filter(isfinite, to_vector(apply_tcode(fred[:, "CPIAUCSL"])))
y_cpi = y_cpi[end-99:end]
n = min(length(y), length(y_cpi))
cs = cross_spectrum(y[1:n], y_cpi[1:n])
report(cs)
```

```julia
plot_result(cs)
```

```@raw html
<iframe src="../assets/plots/spectral_cross.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 4: Ideal bandpass (business-cycle frequencies)**

```@example spectral
# Business cycle: 18–96 months (1.5–8 years)
y_bc = ideal_bandpass(y, 2π/96, 2π/18)
nothing  # hide
```

**Recipe 5: Fisher test for hidden periodicities**

```@example spectral
y_wn = randn(200)
result = fisher_test(y_wn)
report(result)
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

Positive lags indicate ``x`` leads ``y``; negative lags indicate ``y`` leads ``x``.

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
| `lags` | `Int` | `0` | Maximum lag (0 = ``\min(n-1, \lfloor 10\log_{10}(n)\rfloor)``) |
| `method` | `Symbol` | `:levinson` | PACF method: `:levinson` or `:ols` (ignored for `acf` and `ccf`) |
| `conf_level` | `Real` | `0.95` | Confidence level for white-noise band |

### Return Value

**`ACFResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `lags` | `Vector{Int}` | Lag indices |
| `acf` | `Vector{T}` | Autocorrelation values |
| `pacf` | `Vector{T}` | Partial autocorrelation values |
| `ci` | `T` | Confidence interval half-width (``\pm z_{\alpha/2}/\sqrt{n}``) |
| `ccf` | `Union{Nothing,Vector{T}}` | Cross-correlation values (non-null for `ccf()`) |
| `q_stats` | `Vector{T}` | Cumulative Ljung-Box Q-statistics |
| `q_pvalues` | `Vector{T}` | P-values of Q-statistics (``\chi^2`` distribution) |
| `nobs` | `Int` | Number of observations |

!!! note "Ljung-Box Q-Statistics in Correlogram"
    The `acf` and `acf_pacf` functions compute cumulative Ljung-Box Q-statistics at each lag. The ``k``-th Q-statistic tests ``H_0: \rho_1 = \rho_2 = \cdots = \rho_k = 0`` against the alternative that at least one autocorrelation is non-zero. The `report()` output displays these in a Stata/EViews-style correlogram table.

---

## Spectral Density Estimation

The **spectral density** ``S(\omega)`` decomposes a stationary process into contributions from different frequencies ``\omega \in [0, \pi]``:

```math
S(\omega) = \frac{1}{2\pi} \sum_{k=-\infty}^{\infty} \gamma_k e^{-i\omega k}
```

where ``\gamma_k = \text{Cov}(y_t, y_{t-k})`` is the autocovariance at lag ``k``. The integral of ``S(\omega)`` over ``[0, \pi]`` equals the variance of the process.

### Periodogram

The **raw periodogram** is the sample analog of the spectral density:

```math
I(\omega_j) = \frac{1}{2\pi n U} \left| \sum_{t=1}^{n} w_t y_t e^{-i\omega_j t} \right|^2
```

where ``w_t`` is a data window and ``U = n^{-1}\sum w_t^2`` is the window energy normalization. The periodogram is computed via the FFT in ``O(n \log n)`` time.

```@example spectral
I = periodogram(y; window=:hann)
report(I)
```

### Welch's Method

**Welch's averaged modified periodogram** (Welch 1967) reduces variance by averaging periodograms computed from overlapping data segments:

1. Divide the series into ``K`` segments of length ``L`` with overlap fraction ``\alpha``
2. Apply a data window (Hann, Hamming, Blackman, etc.) to each segment
3. Compute the periodogram of each windowed segment
4. Average across all ``K`` periodograms

The variance reduction comes at the cost of reduced frequency resolution. The equivalent degrees of freedom are ``2K``.

```@example spectral
sd = spectral_density(y; method=:welch, window=:hann, segment_length=64, overlap=0.5)
report(sd)
```

### Kernel-Smoothed Periodogram

The **smoothed periodogram** applies a Daniell kernel to the raw periodogram:

```math
\hat{S}(\omega_k) = \frac{1}{2m+1} \sum_{j=-m}^{m} I(\omega_{k+j})
```

where ``m`` is the kernel half-width (bandwidth). Larger bandwidth reduces variance but increases bias.

```@example spectral
sd = spectral_density(y; method=:smoothed, bandwidth=7)
nothing  # hide
```

### AR Parametric Spectrum

The **AR parametric spectrum** (Burg 1968) fits an autoregressive model and evaluates its theoretical spectral density:

```math
S(\omega) = \frac{\hat{\sigma}^2}{2\pi \left| 1 + \sum_{j=1}^{p} \hat{a}_j e^{-i\omega j} \right|^2}
```

The AR order ``p`` is selected by AIC (default) or specified directly. Burg's algorithm produces stable AR coefficient estimates.

```@example spectral
sd = spectral_density(y; method=:ar, order=12)
nothing  # hide
```

### Method Comparison

| Method | Variance | Resolution | Assumptions | Best For |
|--------|----------|------------|-------------|----------|
| Periodogram | High | Highest | None | Exploratory, long series |
| Welch | Low | Moderate | Stationarity within segments | General-purpose default |
| Smoothed | Low | Moderate | Stationarity | Smooth spectral shape |
| AR | Lowest | Highest | AR process | Sharp peaks, short series |

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:welch` | `:periodogram`, `:welch`, `:smoothed`, or `:ar` |
| `window` | `Symbol` | `:hann` | Data window (`:rectangular`, `:bartlett`, `:hann`, `:hamming`, `:blackman`, `:tukey`, `:flat_top`) |
| `segment_length` | `Int` | ``n/4`` | Segment length for Welch (minimum 16) |
| `overlap` | `Real` | `0.5` | Overlap fraction for Welch, ``\in [0, 1)`` |
| `bandwidth` | `Int` | ``\lfloor\sqrt{n}\rfloor`` | Daniell kernel half-width for `:smoothed` |
| `order` | `Int` | AIC | AR order for `:ar` method |
| `n_freq` | `Int` | `256` | Frequency grid points for `:ar` |
| `conf_level` | `Real` | `0.95` | Confidence level for spectral bounds |

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
- **Phase**: ``\phi_{xy}(\omega) = \arctan(Q_{xy} / C_{xy})`` --- the lead-lag relationship in radians
- **Gain**: ``G_{xy}(\omega) = |S_{xy}(\omega)| / S_{xx}(\omega)`` --- the amplitude ratio

```@example spectral
cs = cross_spectrum(y[1:n], y_cpi[1:n]; window=:hann)
report(cs)
```

```julia
plot_result(cs)
```

```@raw html
<iframe src="../assets/plots/spectral_cross.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# Convenience accessors
freq, coh = coherence(y[1:n], y_cpi[1:n])
freq, ph  = phase(y[1:n], y_cpi[1:n])
freq, g   = gain(y[1:n], y_cpi[1:n])
nothing  # hide
```

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `window` | `Symbol` | `:hann` | Data window |
| `segment_length` | `Int` | ``n/4`` | Segment length for Welch averaging |
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
report(result)
```

### Bartlett's Cumulative Periodogram Test

Bartlett's test (Bartlett 1955) checks whether the cumulative normalized periodogram follows the uniform distribution expected under white noise:

```math
D = \max_j \left| \frac{\sum_{k=1}^{j} I(\omega_k)}{\sum_{k=1}^{m} I(\omega_k)} - \frac{j}{m} \right|
```

The Kolmogorov-Smirnov statistic ``D`` measures the maximum departure from uniformity. Rejection indicates the series is not white noise.

```@example spectral
result = bartlett_white_noise_test(y)
report(result)
```

### Band Power

`band_power` computes the integrated spectral density in a frequency band:

```@example spectral
sd = spectral_density(y; method=:welch)
power = band_power(sd, 2π/32, 2π/6)  # business-cycle band (6–32 quarters)
```

---

## Portmanteau Tests

Three classical serial correlation tests complement the spectral diagnostics.

### Ljung-Box Q Test

The Ljung-Box test (Ljung & Box 1978) checks for autocorrelation up to lag ``h``:

```math
Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(h - p)
```

where ``p`` is the number of fitted AR/MA parameters (set via `fitdf`).

```@example spectral
result = ljung_box_test(y; lags=20, fitdf=0)
report(result)
```

### Box-Pierce Q Test

The original Box-Pierce test (Box & Pierce 1970) uses the simpler statistic ``Q_0 = n \sum \hat{\rho}_k^2``. The Ljung-Box modification is preferred for small samples.

```@example spectral
result = box_pierce_test(y; lags=20)
report(result)
```

### Durbin-Watson Test

The Durbin-Watson test (Durbin & Watson 1950, 1951) detects first-order autocorrelation in regression residuals:

```math
DW = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2} \approx 2(1 - \hat{\rho}_1)
```

Values near 2 indicate no autocorrelation; values near 0 indicate positive autocorrelation; values near 4 indicate negative autocorrelation.

```@example spectral
dw_result = durbin_watson_test(y)
report(dw_result)
```

---

## Frequency-Domain Filtering

### Ideal Bandpass Filter

The **ideal bandpass filter** retains frequency components in ``[\omega_l, \omega_h]`` by zeroing out all other Fourier coefficients:

```@example spectral
# Business cycle: 18–96 months
y_bc = ideal_bandpass(y, 2π/96, 2π/18)

# High-frequency: < 18 months
y_hf = ideal_bandpass(y, 2π/18, π)
nothing  # hide
```

!!! warning "Gibbs Phenomenon"
    The ideal bandpass filter applies a sharp cutoff in the frequency domain, which produces ringing artifacts (Gibbs phenomenon) in the time domain. For applied work, the Baxter-King or HP filters provide smoother alternatives. The ideal bandpass is useful for quick exploratory analysis or when exact frequency isolation is needed.

### Transfer Function

`transfer_function` computes the frequency response (gain and phase) of three standard macroeconomic filters:

```@example spectral
# HP filter frequency response
tf_hp = transfer_function(:hp; lambda=1600)
report(tf_hp)
```

```julia
plot_result(tf_hp)
```

```@raw html
<iframe src="../assets/plots/spectral_transfer.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# Baxter-King frequency response
tf_bk = transfer_function(:bk; K=12)

# Hamilton filter frequency response
tf_ham = transfer_function(:hamilton; h=8)
nothing  # hide
```

The HP transfer function has the closed-form gain:

```math
G(\omega) = \frac{4\lambda \sin^2(\omega/2)}{1 + 4\lambda \sin^2(\omega/2)}
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lambda` | `Real` | `1600` | HP smoothing parameter |
| `K` | `Int` | `12` | Baxter-King half-window length |
| `h` | `Int` | `8` | Hamilton regression horizon |
| `n_freq` | `Int` | `256` | Number of frequency grid points |

### Return Value

**`TransferFunctionResult{T}`:**

| Field | Type | Description |
|-------|------|-------------|
| `freq` | `Vector{T}` | Frequency grid in ``[0, \pi]`` |
| `gain` | `Vector{T}` | Gain (amplitude) at each frequency |
| `phase` | `Vector{T}` | Phase shift (radians) at each frequency |
| `filter` | `Symbol` | Filter type (`:hp`, `:bk`, `:hamilton`) |

---

## Complete Example

This example demonstrates a full spectral analysis workflow on U.S. industrial production growth:

```@example spectral
# 1. Correlogram: ACF + PACF with Ljung-Box Q-stats
corr = acf_pacf(y; lags=24)
report(corr)
```

```julia
plot_result(corr)
```

```@raw html
<iframe src="../assets/plots/spectral_acf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# 2. Spectral density: Welch's method with Hann window
sd = spectral_density(y; method=:welch, window=:hann)
report(sd)
```

```julia
plot_result(sd)
```

```@raw html
<iframe src="../assets/plots/spectral_density.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# 3. Business-cycle power as fraction of total variance
total_power = band_power(sd, 0.0, π)
bc_power = band_power(sd, 2π/96, 2π/18)  # 18–96 months
```

```@example spectral
# 4. AR parametric spectrum for comparison
sd_ar = spectral_density(y; method=:ar)
report(sd_ar)
```

```@example spectral
# 5. Cross-spectrum: industrial production vs. CPI inflation
cs = cross_spectrum(y[1:n], y_cpi[1:n])
report(cs)
```

```julia
plot_result(cs)
```

```@raw html
<iframe src="../assets/plots/spectral_density.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example spectral
# 6. Diagnostic tests
fisher = fisher_test(y)
bartlett = bartlett_white_noise_test(y)
lb = ljung_box_test(y; lags=20)
report(fisher)
```

```@example spectral
report(bartlett)
```

```@example spectral
report(lb)
```

The correlogram reveals significant autocorrelation in industrial production growth, consistent with the business-cycle peak visible in the spectral density around ``\omega \approx 2\pi/48`` (4-year cycle). The coherence between output growth and inflation indicates the frequencies at which these series co-move.

---

## Common Pitfalls

1. **Non-stationary input**: Spectral density estimation assumes stationarity. Apply `diff()` or a filter to trending series before computing the periodogram. The ACF of a non-stationary series decays slowly, producing a spectral density concentrated near ``\omega = 0``.

2. **Periodogram variance**: The raw periodogram is an inconsistent estimator --- its variance does not decrease with sample size. Use `spectral_density(y; method=:welch)` or `method=:smoothed` for consistent estimation.

3. **Window choice affects resolution**: Data windows (Hann, Hamming, Blackman) trade main-lobe width for sidelobe suppression. Use `:rectangular` for maximum frequency resolution, `:blackman` for maximum sidelobe suppression.

4. **Frequency units**: All frequencies are in radians per observation, ``\omega \in [0, \pi]``. To convert to period: ``T = 2\pi/\omega``. For monthly data, ``\omega = 2\pi/12`` corresponds to a 12-month (annual) cycle.

5. **Ljung-Box `fitdf` parameter**: When testing residuals from an ARMA(p,q) model, set `fitdf=p+q` to adjust the degrees of freedom. Omitting this inflates the test size.

---

## References

- Bartlett, M. S. (1955). An Introduction to Stochastic Processes. Cambridge University Press.

- Box, G. E. P., & Pierce, D. A. (1970). Distribution of Residual Autocorrelations in Autoregressive-Integrated Moving Average Time Series Models. *Journal of the American Statistical Association*, 65(332), 1509--1526. [DOI](https://doi.org/10.1080/01621459.1970.10481180)

- Brockwell, P. J., & Davis, R. A. (1991). *Time Series: Theory and Methods* (2nd ed.). Springer.

- Brillinger, D. R. (1981). *Time Series: Data Analysis and Theory*. Holden-Day.

- Burg, J. P. (1968). A New Analysis Technique for Time Series Data. NATO Advanced Study Institute on Signal Processing, Enschede, Netherlands.

- Durbin, J., & Watson, G. S. (1950). Testing for Serial Correlation in Least Squares Regression. I. *Biometrika*, 37(3/4), 409--428. [DOI](https://doi.org/10.2307/2332391)

- Fisher, R. A. (1929). Tests of Significance in Harmonic Analysis. *Proceedings of the Royal Society of London A*, 125(796), 54--59. [DOI](https://doi.org/10.1098/rspa.1929.0151)

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

- Ljung, G. M., & Box, G. E. P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297--303. [DOI](https://doi.org/10.1093/biomet/65.2.297)

- Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.

- Welch, P. D. (1967). The Use of Fast Fourier Transform for the Estimation of Power Spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70--73. [DOI](https://doi.org/10.1109/TAU.1967.1161901)
