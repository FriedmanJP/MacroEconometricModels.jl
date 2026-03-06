# Design: Spectral Analysis, ACF/PACF, and FFTW Fix

**Issues:** #67, #68, #72
**Date:** 2026-03-07
**Approach:** Monolithic — all three implemented together in one pass

---

## 1. FFTW Dependency Fix (#72)

Move FFTW from `[weakdeps]` to `[deps]` in `Project.toml`. Remove extension entry and delete `ext/MacroEconometricModelsFFTWExt.jl`. Replace `_FFT_IMPL[]`/`_IFFT_IMPL[]`/`_check_fftw()` indirection in `src/factor/generalized.jl` with direct `FFTW.fft`/`FFTW.ifft` calls. No API change.

## 2. Module Structure

New directory `src/spectral/` with 7 files:

| File | Contents |
|------|----------|
| `types.jl` | `ACFResult{T}`, `SpectralDensityResult{T}`, `CrossSpectrumResult{T}`, `TransferFunctionResult{T}` |
| `acf.jl` | `acf()`, `pacf()`, `acf_pacf()`, `ccf()` |
| `estimation.jl` | `periodogram()`, `spectral_density()` (Welch, smoothed Daniell, AR with auto/manual order) |
| `cross.jl` | `cross_spectrum()`, `coherence()`, `phase()`, `gain()` |
| `diagnostics.jl` | `fisher_test()`, `bartlett_white_noise_test()`, `band_power()` |
| `filtering.jl` | `ideal_bandpass()`, `transfer_function()` |
| `windows.jl` | Bartlett, Hann, Hamming, Blackman, Tukey, flat-top window functions |

New test types added to `src/teststat/`: `LjungBoxResult`, `BoxPierceResult`, `DurbinWatsonResult`, `FisherTestResult`, `BartlettWhiteNoiseResult` — all `<: StatsAPI.HypothesisTest`.

## 3. Result Types

All immutable, parametric on `{T<:AbstractFloat}`:

```julia
struct ACFResult{T}
    lags::Vector{Int}
    acf::Vector{T}
    pacf::Vector{T}
    ci::T                          # ±1.96/sqrt(T)
    ccf::Union{Nothing,Vector{T}}
    q_stats::Vector{T}            # cumulative Ljung-Box Q at each lag
    q_pvalues::Vector{T}          # p-values for Q-stats
    nobs::Int
end

struct SpectralDensityResult{T}
    freq::Vector{T}
    density::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    method::Symbol
    bandwidth::T
    nobs::Int
end

struct CrossSpectrumResult{T}
    freq::Vector{T}
    co_spectrum::Vector{T}
    quad_spectrum::Vector{T}
    coherence::Vector{T}
    phase::Vector{T}
    gain::Vector{T}
    nobs::Int
end

struct TransferFunctionResult{T}
    freq::Vector{T}
    gain::Vector{T}
    phase::Vector{T}
    filter::Symbol
end
```

Test result types follow existing pattern: `statistic::T`, `pvalue::T`, `nobs::Int`, plus test-specific fields (`lags::Int` for portmanteau tests).

## 4. API

### ACF/PACF
```julia
acf(y, maxlag=20)
pacf(y, maxlag=20; method=:levinson)  # or :ols
acf_pacf(y, maxlag=20)
ccf(y1, y2, maxlag=20)

# TimeSeriesData/PanelData dispatch
acf(ts_data, 20; var=:GDP)
acf(panel_data, 20; var=:GDP)  # returns Dict{group => ACFResult}
```

### Spectral Density
```julia
periodogram(y)
spectral_density(y; method=:welch, window=:hann, segments=8)
spectral_density(y; method=:smoothed, kernel=:daniell, bandwidth=5)
spectral_density(y; method=:ar)          # auto order via AIC
spectral_density(y; method=:ar, order=4)  # manual order

# TimeSeriesData/PanelData dispatch
spectral_density(ts_data; var=:GDP, method=:welch)
spectral_density(panel_data; var=:GDP)   # per-group
```

### Cross-Spectral
```julia
csp = cross_spectrum(y1, y2; method=:welch)
coherence(csp)   # accessor, returns Vector{T}
phase(csp)       # accessor
gain(csp)        # accessor
```

### Statistical Tests (in src/teststat/)
```julia
ljung_box_test(y, lags=10)
box_pierce_test(y, lags=10)
durbin_watson_test(resid, X)
fisher_test(y)
bartlett_white_noise_test(y)
```

### Diagnostics & Filtering (in src/spectral/)
```julia
band_power(sp::SpectralDensityResult, (lo, hi))  # variance share
ideal_bandpass(y, (lo, hi))                       # FFT exact filter
transfer_function(:hp; lambda=1600)
transfer_function(:bk; pl=6, pu=32)
transfer_function(:hamilton; h=8, p=4)
```

## 5. Plotting & Display

### plot_result() dispatches (4 new)

| Type | Plot |
|------|------|
| `SpectralDensityResult` | Log-spectral density line with shaded CI band |
| `CrossSpectrumResult` | 2-panel: coherence + phase vs frequency |
| `ACFResult` | Side-by-side ACF + PACF bar charts with dashed CI lines |
| `TransferFunctionResult` | 2-panel: gain + phase vs frequency |

Frequency axis uses period labels when `Frequency` metadata available.

### Base.show() tables (Stata-style)

- `ACFResult`: Lag | ACF | PACF | Q-Stat | Prob (Stata `corrgram` style)
- `SpectralDensityResult`: header + Freq | Period | Density | CI_low | CI_high
- `CrossSpectrumResult`: Freq | Coherence | Phase | Gain
- `TransferFunctionResult`: Freq | Gain | Phase
- Test results: single-row Statistic | p-value | Lags | N

## 6. Integration Points

- `src/MacroEconometricModels.jl`: include spectral files, add all exports
- `src/data/convert.jl`: TimeSeriesData/PanelData dispatch wrappers
- `src/plotting/models.jl`: 4 new plot_result methods
- `src/factor/generalized.jl`: direct FFTW calls (remove indirection)

## 7. Testing

Single new file `test/spectral_tests.jl` added to parallel test groups.

**Test categories:**
- ACF/PACF: white noise near-zero, AR(1) ACF(1) ≈ ρ, PACF cutoff, Levinson vs OLS, CCF symmetry
- Periodogram: flat for white noise, spike for sinusoid, Parseval's theorem
- Spectral density: Welch variance reduction, AR(1) analytical match, auto order recovery
- Cross-spectrum: perfect coherence, zero phase, known phase shift
- Windows: length, symmetry, peak, [0,1] bounds
- Tests: Ljung-Box/Box-Pierce on white noise vs AR(1), Fisher sinusoid detection, Bartlett white noise, DW ≈ 2
- Filtering: bandpass preserves in-band, kills out-of-band, HP transfer function analytical match
- Integration: TimeSeriesData dispatch consistency, PanelData per-group Dict, frequency labels
- FFTW fix: estimate_gdfm() works without using FFTW

Deterministic test data (known processes, planted sinusoids). `Random.MersenneTwister(seed)` for stochastic tests with generous tolerances.

## 8. Deferred (not in scope)

- Partial coherence (controlling for third variable)
- Spectral interpolation for missing observations
- Spectral density ratio test (structural break in frequency domain)
- Robust ACF (rank-based, median-based)

These can be added later without API changes.
