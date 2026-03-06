# Spectral Analysis, ACF/PACF & FFTW Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement issues #67 (spectral analysis), #68 (ACF/PACF), and #72 (FFTW dependency fix) as a monolithic change.

**Architecture:** New `src/spectral/` module (7 files) for ACF/PACF and spectral analysis. New test types in `src/teststat/`. FFTW moved from weak dep to regular dep. All functions dispatch on `TimeSeriesData`/`PanelData` via `src/data/convert.jl`. Plotting via 4 new `plot_result()` dispatches.

**Tech Stack:** Julia, FFTW.jl (direct), Distributions.jl (chi-squared p-values), StatsAPI.jl, D3.js (inline plots)

---

## Task 1: FFTW Dependency Fix (#72)

**Files:**
- Modify: `Project.toml` (lines 18-28)
- Delete: `ext/MacroEconometricModelsFFTWExt.jl`
- Modify: `src/factor/generalized.jl` (lines 23-38)
- Modify: `test/runtests.jl` (lines 8, 134)

**Step 1: Update Project.toml**

Move FFTW from `[weakdeps]` to `[deps]` and remove extension entry:

```toml
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsAPI = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[weakdeps]
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PATHSolver = "f5f7c340-0bb3-5c69-969a-41884d311d1b"

[extensions]
MacroEconometricModelsJuMPExt = ["JuMP", "Ipopt"]
MacroEconometricModelsPATHExt = ["JuMP", "PATHSolver"]
```

Also remove FFTW from `[extras]` and `[targets].test` (it's now a regular dep, loaded automatically):

```toml
[extras]
Aqua = "4c88cf16-eb10-579e-8560-4a9242c79595"
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PATHSolver = "f5f7c340-0bb3-5c69-969a-41884d311d1b"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Aqua", "Test", "Documenter", "JuMP", "Ipopt", "PATHSolver"]
```

**Step 2: Replace FFT indirection in generalized.jl**

Replace lines 23-38 of `src/factor/generalized.jl`:

```julia
using LinearAlgebra, Statistics, StatsAPI
using FFTW
```

Remove entirely:
- `const _FFT_IMPL = Ref{Any}(nothing)` (line 29)
- `const _IFFT_IMPL = Ref{Any}(nothing)` (line 30)
- `function _check_fftw()` (lines 32-35)
- `_fft(X, dims) = ...` (line 37)
- `_ifft(X, dims) = ...` (line 38)

Then search-and-replace in the same file:
- `_fft(` → `FFTW.fft(`
- `_ifft(` → `FFTW.ifft(`

There are 5 call sites: lines 185, 253, 262, 269, 277.

**Step 3: Delete extension file**

```bash
rm ext/MacroEconometricModelsFFTWExt.jl
```

If `ext/` directory is now empty, delete it too. Check for other extension files first (`MacroEconometricModelsJuMPExt.jl`, `MacroEconometricModelsPATHExt.jl`).

**Step 4: Add `using FFTW` to main module**

In `src/MacroEconometricModels.jl`, add `using FFTW` after line 68 (after `using SparseArrays`).

**Step 5: Remove `using FFTW` from test/runtests.jl**

Line 8: `using FFTW  # activate FFTW extension for GDFM tests` — delete this line.
Line 134: `using Test, MacroEconometricModels, FFTW` → `using Test, MacroEconometricModels`

**Step 6: Verify GDFM tests still pass**

```bash
cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/polymorphic-tickling-raven
julia --project=. -e 'using Test, MacroEconometricModels; @testset "GDFM" begin include("test/factor/test_gdfm.jl") end'
```

Expected: All GDFM tests pass without `using FFTW`.

**Step 7: Commit**

```bash
git add Project.toml src/factor/generalized.jl src/MacroEconometricModels.jl test/runtests.jl
git rm ext/MacroEconometricModelsFFTWExt.jl
git commit -m "fix: make FFTW a regular dependency, remove weak dep extension (#72)"
```

---

## Task 2: Spectral Result Types & Window Functions

**Files:**
- Create: `src/spectral/types.jl`
- Create: `src/spectral/windows.jl`

**Step 1: Create `src/spectral/types.jl`**

```julia
# MacroEconometricModels.jl — Spectral Analysis Types

using StatsAPI

"""
    ACFResult{T} <: AbstractAnalysisResult

Autocorrelation and partial autocorrelation function result.

Fields:
- `lags`: Lag indices 1:maxlag
- `acf`: Sample autocorrelation at each lag
- `pacf`: Partial autocorrelation at each lag
- `ci`: Confidence band width (±1.96/√n by default)
- `ccf`: Cross-correlation (nothing for univariate)
- `q_stats`: Cumulative Ljung-Box Q statistics
- `q_pvalues`: P-values for Q statistics
- `nobs`: Number of observations
"""
struct ACFResult{T<:AbstractFloat} <: AbstractAnalysisResult
    lags::Vector{Int}
    acf::Vector{T}
    pacf::Vector{T}
    ci::T
    ccf::Union{Nothing,Vector{T}}
    q_stats::Vector{T}
    q_pvalues::Vector{T}
    nobs::Int
end

"""
    SpectralDensityResult{T} <: AbstractAnalysisResult

Spectral density estimate with confidence intervals.

Fields:
- `freq`: Frequencies (0 to π)
- `density`: Estimated spectral density
- `ci_lower`: Lower confidence bound
- `ci_upper`: Upper confidence bound
- `method`: Estimation method (:periodogram, :welch, :smoothed, :ar)
- `bandwidth`: Smoothing bandwidth (0 for periodogram)
- `nobs`: Number of observations
"""
struct SpectralDensityResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    density::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    method::Symbol
    bandwidth::T
    nobs::Int
end

"""
    CrossSpectrumResult{T} <: AbstractAnalysisResult

Cross-spectral analysis result.

Fields:
- `freq`: Frequencies (0 to π)
- `co_spectrum`: Co-spectrum (real part of cross-spectral density)
- `quad_spectrum`: Quadrature spectrum (imaginary part)
- `coherence`: Squared coherency at each frequency
- `phase`: Phase spectrum (radians)
- `gain`: Gain (amplitude ratio)
- `nobs`: Number of observations
"""
struct CrossSpectrumResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    co_spectrum::Vector{T}
    quad_spectrum::Vector{T}
    coherence::Vector{T}
    phase::Vector{T}
    gain::Vector{T}
    nobs::Int
end

"""
    TransferFunctionResult{T} <: AbstractAnalysisResult

Filter transfer function (gain and phase).

Fields:
- `freq`: Frequencies (0 to π)
- `gain`: Filter gain at each frequency
- `phase`: Filter phase at each frequency
- `filter`: Filter type (:hp, :bk, :hamilton)
"""
struct TransferFunctionResult{T<:AbstractFloat} <: AbstractAnalysisResult
    freq::Vector{T}
    gain::Vector{T}
    phase::Vector{T}
    filter::Symbol
end
```

**Step 2: Create `src/spectral/windows.jl`**

```julia
# MacroEconometricModels.jl — Window Functions for Spectral Analysis

"""
    _spectral_window(n::Int, wtype::Symbol) -> Vector

Compute a window function of length `n`.

Supported windows: :rectangular, :bartlett, :hann, :hamming, :blackman, :tukey, :flat_top
"""
function _spectral_window(n::Int, wtype::Symbol)
    n < 1 && throw(ArgumentError("Window length must be ≥ 1"))
    n == 1 && return [1.0]

    w = zeros(Float64, n)
    nm1 = n - 1

    if wtype == :rectangular
        fill!(w, 1.0)
    elseif wtype == :bartlett
        for i in 0:nm1
            w[i+1] = 1.0 - abs(2i/nm1 - 1.0)
        end
    elseif wtype == :hann || wtype == :hanning
        for i in 0:nm1
            w[i+1] = 0.5 * (1.0 - cos(2π * i / nm1))
        end
    elseif wtype == :hamming
        for i in 0:nm1
            w[i+1] = 0.54 - 0.46 * cos(2π * i / nm1)
        end
    elseif wtype == :blackman
        for i in 0:nm1
            w[i+1] = 0.42 - 0.5 * cos(2π * i / nm1) + 0.08 * cos(4π * i / nm1)
        end
    elseif wtype == :tukey
        # Tukey window with α=0.5 (Tukey-Hanning)
        α = 0.5
        for i in 0:nm1
            t = i / nm1
            if t < α/2
                w[i+1] = 0.5 * (1.0 + cos(2π/α * (t - α/2)))
            elseif t > 1.0 - α/2
                w[i+1] = 0.5 * (1.0 + cos(2π/α * (t - 1.0 + α/2)))
            else
                w[i+1] = 1.0
            end
        end
    elseif wtype == :flat_top
        for i in 0:nm1
            w[i+1] = 1.0 - 1.93*cos(2π*i/nm1) + 1.29*cos(4π*i/nm1) -
                     0.388*cos(6π*i/nm1) + 0.0322*cos(8π*i/nm1)
        end
    else
        throw(ArgumentError("Unknown window type: $wtype. Supported: :rectangular, :bartlett, :hann, :hamming, :blackman, :tukey, :flat_top"))
    end
    w
end
```

**Step 3: Verify types compile**

```bash
julia --project=. -e 'using MacroEconometricModels'
```

This will fail until we add the includes — that's OK at this stage. We just check syntax:

```bash
julia --project=. -e 'include("src/spectral/types.jl"); include("src/spectral/windows.jl"); println("OK")'
```

This will also fail because `AbstractAnalysisResult` isn't in scope. Don't worry — the includes will be wired in Task 8.

**Step 4: Commit**

```bash
git add src/spectral/types.jl src/spectral/windows.jl
git commit -m "feat(spectral): add result types and window functions"
```

---

## Task 3: ACF/PACF Implementation

**Files:**
- Create: `src/spectral/acf.jl`

**Step 1: Create `src/spectral/acf.jl`**

```julia
# MacroEconometricModels.jl — ACF, PACF, CCF

using Statistics, Distributions

"""
    acf(y::AbstractVector, maxlag::Int=20) -> ACFResult

Compute sample autocorrelation function up to `maxlag`.

Returns ACFResult with `acf` field populated, `pacf` as zeros.
"""
function acf(y::AbstractVector{T}, maxlag::Int=20) where {T<:AbstractFloat}
    n = length(y)
    maxlag < 1 && throw(ArgumentError("maxlag must be ≥ 1"))
    maxlag >= n && throw(ArgumentError("maxlag must be < length(y) = $n"))

    ȳ = mean(y)
    γ0 = sum((yi - ȳ)^2 for yi in y) / n
    γ0 < eps(T) && return _zero_acf_result(T, maxlag, n)

    acf_vals = zeros(T, maxlag)
    for k in 1:maxlag
        acf_vals[k] = sum((y[t] - ȳ) * (y[t-k] - ȳ) for t in k+1:n) / (n * γ0)
    end

    ci = T(1.96) / sqrt(T(n))
    q_stats, q_pvals = _ljung_box_cumulative(acf_vals, n)

    ACFResult{T}(collect(1:maxlag), acf_vals, zeros(T, maxlag), ci, nothing, q_stats, q_pvals, n)
end

acf(y::AbstractVector, maxlag::Int=20) = acf(Float64.(y), maxlag)

"""
    pacf(y::AbstractVector, maxlag::Int=20; method::Symbol=:levinson) -> ACFResult

Compute partial autocorrelation function up to `maxlag`.

Methods:
- `:levinson` (default): Levinson-Durbin recursion
- `:ols`: OLS regression-based
"""
function pacf(y::AbstractVector{T}, maxlag::Int=20; method::Symbol=:levinson) where {T<:AbstractFloat}
    n = length(y)
    maxlag < 1 && throw(ArgumentError("maxlag must be ≥ 1"))
    maxlag >= n && throw(ArgumentError("maxlag must be < length(y) = $n"))

    pacf_vals = if method == :levinson
        _pacf_levinson(y, maxlag)
    elseif method == :ols
        _pacf_ols(y, maxlag)
    else
        throw(ArgumentError("method must be :levinson or :ols"))
    end

    ci = T(1.96) / sqrt(T(n))
    ACFResult{T}(collect(1:maxlag), zeros(T, maxlag), pacf_vals, ci, nothing,
                 zeros(T, maxlag), zeros(T, maxlag), n)
end

pacf(y::AbstractVector, maxlag::Int=20; kwargs...) = pacf(Float64.(y), maxlag; kwargs...)

"""
    acf_pacf(y::AbstractVector, maxlag::Int=20; method::Symbol=:levinson) -> ACFResult

Compute both ACF and PACF in one call, with cumulative Ljung-Box Q-stats.
"""
function acf_pacf(y::AbstractVector{T}, maxlag::Int=20; method::Symbol=:levinson) where {T<:AbstractFloat}
    n = length(y)
    maxlag < 1 && throw(ArgumentError("maxlag must be ≥ 1"))
    maxlag >= n && throw(ArgumentError("maxlag must be < length(y) = $n"))

    ȳ = mean(y)
    γ0 = sum((yi - ȳ)^2 for yi in y) / n
    γ0 < eps(T) && return _zero_acf_result(T, maxlag, n)

    acf_vals = zeros(T, maxlag)
    for k in 1:maxlag
        acf_vals[k] = sum((y[t] - ȳ) * (y[t-k] - ȳ) for t in k+1:n) / (n * γ0)
    end

    pacf_vals = method == :levinson ? _pacf_levinson(y, maxlag) : _pacf_ols(y, maxlag)
    ci = T(1.96) / sqrt(T(n))
    q_stats, q_pvals = _ljung_box_cumulative(acf_vals, n)

    ACFResult{T}(collect(1:maxlag), acf_vals, pacf_vals, ci, nothing, q_stats, q_pvals, n)
end

acf_pacf(y::AbstractVector, maxlag::Int=20; kwargs...) = acf_pacf(Float64.(y), maxlag; kwargs...)

"""
    ccf(y1::AbstractVector, y2::AbstractVector, maxlag::Int=20) -> ACFResult

Compute cross-correlation function between `y1` and `y2` at lags -maxlag:maxlag.

The CCF vector has length 2*maxlag+1, with index maxlag+1 corresponding to lag 0.
"""
function ccf(y1::AbstractVector{T}, y2::AbstractVector{T}, maxlag::Int=20) where {T<:AbstractFloat}
    n = length(y1)
    length(y2) == n || throw(DimensionMismatch("y1 and y2 must have same length"))
    maxlag < 1 && throw(ArgumentError("maxlag must be ≥ 1"))
    maxlag >= n && throw(ArgumentError("maxlag must be < length(y) = $n"))

    ȳ1, ȳ2 = mean(y1), mean(y2)
    σ1 = sqrt(sum((y1i - ȳ1)^2 for y1i in y1) / n)
    σ2 = sqrt(sum((y2i - ȳ2)^2 for y2i in y2) / n)
    denom = n * σ1 * σ2

    # Lags from -maxlag to +maxlag
    ccf_vals = zeros(T, 2 * maxlag + 1)
    for k in -maxlag:maxlag
        s = zero(T)
        if k >= 0
            for t in k+1:n
                s += (y1[t] - ȳ1) * (y2[t-k] - ȳ2)
            end
        else
            for t in 1:n+k
                s += (y1[t] - ȳ1) * (y2[t-k] - ȳ2)
            end
        end
        ccf_vals[k + maxlag + 1] = s / denom
    end

    ci = T(1.96) / sqrt(T(n))
    lags_full = collect(-maxlag:maxlag)
    nl = length(lags_full)

    ACFResult{T}(lags_full, zeros(T, nl), zeros(T, nl), ci, ccf_vals,
                 zeros(T, nl), zeros(T, nl), n)
end

ccf(y1::AbstractVector, y2::AbstractVector, maxlag::Int=20) = ccf(Float64.(y1), Float64.(y2), maxlag)

# =============================================================================
# Internal helpers
# =============================================================================

"""Levinson-Durbin recursion for PACF."""
function _pacf_levinson(y::AbstractVector{T}, maxlag::Int) where {T<:AbstractFloat}
    n = length(y)
    ȳ = mean(y)
    # Compute autocovariances
    γ = zeros(T, maxlag + 1)
    for k in 0:maxlag
        γ[k+1] = sum((y[t] - ȳ) * (y[t-k] - ȳ) for t in k+1:n) / n
    end
    γ[1] < eps(T) && return zeros(T, maxlag)

    pacf_vals = zeros(T, maxlag)
    # Levinson-Durbin
    a = zeros(T, maxlag)
    a[1] = γ[2] / γ[1]
    pacf_vals[1] = a[1]
    v = γ[1] * (one(T) - a[1]^2)

    for k in 2:maxlag
        # Forward prediction error
        num = γ[k+1] - sum(a[j] * γ[k-j+1] for j in 1:k-1)
        pacf_vals[k] = num / v
        a_new = zeros(T, k)
        a_new[k] = pacf_vals[k]
        for j in 1:k-1
            a_new[j] = a[j] - pacf_vals[k] * a[k-j]
        end
        a[1:k] = a_new
        v *= (one(T) - pacf_vals[k]^2)
        abs(v) < eps(T) && break
    end
    pacf_vals
end

"""OLS regression-based PACF."""
function _pacf_ols(y::AbstractVector{T}, maxlag::Int) where {T<:AbstractFloat}
    n = length(y)
    ȳ = mean(y)
    y_centered = y .- ȳ
    pacf_vals = zeros(T, maxlag)

    for k in 1:maxlag
        n_eff = n - k
        n_eff < k + 1 && break
        X = ones(T, n_eff, k + 1)
        for j in 1:k
            X[:, j+1] = y_centered[k+1-j:n-j]
        end
        y_reg = y_centered[k+1:n]
        β = X \ y_reg
        pacf_vals[k] = β[k+1]
    end
    pacf_vals
end

"""Cumulative Ljung-Box Q-stats at each lag."""
function _ljung_box_cumulative(acf_vals::Vector{T}, n::Int) where {T<:AbstractFloat}
    maxlag = length(acf_vals)
    q_stats = zeros(T, maxlag)
    q_pvals = zeros(T, maxlag)
    Q = zero(T)
    for k in 1:maxlag
        Q += T(n) * T(n + 2) * acf_vals[k]^2 / T(n - k)
        q_stats[k] = Q
        q_pvals[k] = one(T) - cdf(Chisq(k), Q)
    end
    q_stats, q_pvals
end

"""Return a zero ACF result for constant series."""
function _zero_acf_result(::Type{T}, maxlag::Int, n::Int) where {T}
    ACFResult{T}(collect(1:maxlag), zeros(T, maxlag), zeros(T, maxlag),
                 T(1.96) / sqrt(T(n)), nothing, zeros(T, maxlag), zeros(T, maxlag), n)
end
```

**Step 2: Commit**

```bash
git add src/spectral/acf.jl
git commit -m "feat(spectral): add ACF, PACF, CCF implementation"
```

---

## Task 4: Spectral Density Estimation

**Files:**
- Create: `src/spectral/estimation.jl`

**Step 1: Create `src/spectral/estimation.jl`**

```julia
# MacroEconometricModels.jl — Spectral Density Estimation

using FFTW, Statistics, Distributions, LinearAlgebra

"""
    periodogram(y::AbstractVector) -> SpectralDensityResult

Compute the raw (unsmoothed) periodogram.

The periodogram at frequency ωⱼ = 2πj/T is I(ωⱼ) = |Σₜ yₜ e^{-iωⱼt}|² / (2πT).
"""
function periodogram(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    y_centered = y .- mean(y)
    Y = FFTW.fft(y_centered)
    n_freq = div(n, 2) + 1
    freq = [T(2π * (j - 1) / n) for j in 1:n_freq]

    # I(ω) = |Y(ω)|² / (2π n)
    density = [abs2(Y[j]) / (T(2π) * n) for j in 1:n_freq]

    # Chi-squared CI: periodogram ~ (σ²/2) χ²(2) for each frequency
    # Except freq 0 and Nyquist which are χ²(1)
    ν = T(2)  # degrees of freedom per periodogram ordinate
    ci_lower = density .* (ν / quantile(Chisq(ν), T(0.975)))
    ci_upper = density .* (ν / quantile(Chisq(ν), T(0.025)))

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :periodogram, zero(T), n)
end

periodogram(y::AbstractVector) = periodogram(Float64.(y))

"""
    spectral_density(y::AbstractVector; method=:welch, kwargs...) -> SpectralDensityResult

Estimate the spectral density using the specified method.

# Methods
- `:welch`: Welch's method with overlapping segments
- `:smoothed`: Smoothed periodogram with Daniell kernel
- `:ar`: Autoregressive spectral estimation (Burg's method)

# Keyword Arguments
- `window::Symbol=:hann`: Window function (for :welch)
- `segments::Int=8`: Number of segments (for :welch)
- `overlap::Float64=0.5`: Segment overlap fraction (for :welch)
- `kernel::Symbol=:daniell`: Smoothing kernel (for :smoothed)
- `bandwidth::Int=0`: Kernel bandwidth, 0 = automatic (for :smoothed)
- `order::Int=0`: AR order, 0 = automatic via AIC (for :ar)
- `max_order::Int=0`: Maximum AR order for auto-selection (for :ar)
- `n_freq::Int=512`: Number of frequency points (for :ar)
"""
function spectral_density(y::AbstractVector{T};
    method::Symbol=:welch,
    window::Symbol=:hann,
    segments::Int=8,
    overlap::Real=0.5,
    kernel::Symbol=:daniell,
    bandwidth::Int=0,
    order::Int=0,
    max_order::Int=0,
    n_freq::Int=512,
) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    if method == :welch
        _spectral_welch(y, segments, overlap, window)
    elseif method == :smoothed
        _spectral_smoothed(y, bandwidth, kernel)
    elseif method == :ar
        _spectral_ar(y, order, max_order, n_freq)
    else
        throw(ArgumentError("method must be :welch, :smoothed, or :ar"))
    end
end

spectral_density(y::AbstractVector; kwargs...) = spectral_density(Float64.(y); kwargs...)

# =============================================================================
# Welch's Method
# =============================================================================

function _spectral_welch(y::AbstractVector{T}, segments::Int, overlap::Real, wtype::Symbol) where {T<:AbstractFloat}
    n = length(y)
    # Compute segment length
    seg_len = max(4, round(Int, n / (1 + (segments - 1) * (1 - overlap))))
    step = max(1, round(Int, seg_len * (1 - overlap)))
    n_freq = div(seg_len, 2) + 1
    freq = [T(2π * (j - 1) / seg_len) for j in 1:n_freq]

    win = T.(_spectral_window(seg_len, wtype))
    win_ss = sum(abs2, win)  # window power for normalization

    # Average periodograms over segments
    density = zeros(T, n_freq)
    n_segs = 0
    start = 1
    while start + seg_len - 1 <= n
        segment = (y[start:start+seg_len-1] .- mean(y[start:start+seg_len-1])) .* win
        Y = FFTW.fft(segment)
        for j in 1:n_freq
            density[j] += abs2(Y[j]) / (T(2π) * win_ss)
        end
        n_segs += 1
        start += step
    end
    n_segs < 1 && throw(ArgumentError("Not enough data for $segments segments"))
    density ./= n_segs

    # CI: averaged periodogram has ν = 2*n_segs DOF (approximately)
    ν = T(2 * n_segs)
    ci_lower = density .* (ν / quantile(Chisq(ν), T(0.975)))
    ci_upper = density .* (ν / quantile(Chisq(ν), T(0.025)))

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :welch, T(seg_len), n)
end

# =============================================================================
# Smoothed Periodogram
# =============================================================================

function _spectral_smoothed(y::AbstractVector{T}, bw::Int, kernel::Symbol) where {T<:AbstractFloat}
    n = length(y)
    bw = bw <= 0 ? max(1, round(Int, sqrt(n))) : bw

    # Compute raw periodogram first
    y_centered = y .- mean(y)
    Y = FFTW.fft(y_centered)
    n_freq = div(n, 2) + 1
    freq = [T(2π * (j - 1) / n) for j in 1:n_freq]
    raw = [abs2(Y[j]) / (T(2π) * n) for j in 1:n_freq]

    # Daniell kernel (uniform averaging)
    density = zeros(T, n_freq)
    for j in 1:n_freq
        s = zero(T)
        count = 0
        for k in max(1, j - bw):min(n_freq, j + bw)
            w = if kernel == :daniell
                one(T)
            elseif kernel == :modified_daniell
                # Modified Daniell: downweight endpoints
                abs(k - j) == bw ? T(0.5) : one(T)
            else
                throw(ArgumentError("kernel must be :daniell or :modified_daniell"))
            end
            s += w * raw[k]
            count += 1
        end
        density[j] = s / count
    end

    # DOF for smoothed periodogram: ν ≈ 2 * (2*bw + 1)
    ν = T(2 * min(2 * bw + 1, n_freq))
    ci_lower = density .* (ν / quantile(Chisq(ν), T(0.975)))
    ci_upper = density .* (ν / quantile(Chisq(ν), T(0.025)))

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :smoothed, T(bw), n)
end

# =============================================================================
# AR Spectral Estimation
# =============================================================================

function _spectral_ar(y::AbstractVector{T}, order::Int, max_order::Int, n_freq::Int) where {T<:AbstractFloat}
    n = length(y)
    y_centered = y .- mean(y)

    # Auto order selection via AIC
    if order <= 0
        max_p = max_order > 0 ? max_order : min(round(Int, 10 * log10(n)), n ÷ 4)
        order = _select_ar_order_aic(y_centered, max_p)
    end
    order = min(order, n ÷ 4)

    # Burg's method for AR coefficients
    a, σ2 = _burg_coefficients(y_centered, order)

    # AR spectral density: S(ω) = σ² / (2π |1 - Σ aₖ e^{-iωk}|²)
    freq = [T(π * (j - 1) / (n_freq - 1)) for j in 1:n_freq]
    density = zeros(T, n_freq)
    for j in 1:n_freq
        z = one(Complex{T})
        for k in 1:order
            z -= a[k] * exp(-im * freq[j] * k)
        end
        density[j] = σ2 / (T(2π) * abs2(z))
    end

    # CI: AR spectral estimate has approximately 2n/order DOF
    ν = max(T(2), T(2 * n) / T(order))
    ci_lower = density .* (ν / quantile(Chisq(ν), T(0.975)))
    ci_upper = density .* (ν / quantile(Chisq(ν), T(0.025)))

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :ar, T(order), n)
end

"""Select AR order by minimizing AIC."""
function _select_ar_order_aic(y::AbstractVector{T}, max_p::Int) where {T<:AbstractFloat}
    n = length(y)
    best_aic = T(Inf)
    best_p = 1

    for p in 1:max_p
        n - p < p + 1 && break
        _, σ2 = _burg_coefficients(y, p)
        σ2 <= zero(T) && continue
        aic_val = n * log(σ2) + 2 * p
        if aic_val < best_aic
            best_aic = aic_val
            best_p = p
        end
    end
    best_p
end

"""Burg's method for AR coefficient estimation."""
function _burg_coefficients(y::AbstractVector{T}, order::Int) where {T<:AbstractFloat}
    n = length(y)
    # Initialize forward/backward prediction errors
    ef = copy(y)
    eb = copy(y)
    a = zeros(T, order)
    σ2 = sum(abs2, y) / n

    for k in 1:order
        # Compute reflection coefficient
        num = zero(T)
        den = zero(T)
        for t in k+1:n
            num += T(2) * ef[t] * eb[t-1]
            den += ef[t]^2 + eb[t-1]^2
        end
        den < eps(T) && break
        a_k = num / den

        # Update coefficients
        a_new = zeros(T, k)
        a_new[k] = a_k
        for j in 1:k-1
            a_new[j] = a[j] - a_k * a[k-j]
        end
        a[1:k] = a_new

        # Update prediction errors
        σ2 *= (one(T) - a_k^2)
        ef_new = copy(ef)
        for t in k+1:n
            ef_new[t] = ef[t] - a_k * eb[t-1]
            eb[t] = eb[t-1] - a_k * ef[t]
        end
        ef = ef_new
    end
    a, max(σ2, eps(T))
end
```

**Step 2: Commit**

```bash
git add src/spectral/estimation.jl
git commit -m "feat(spectral): add periodogram, Welch, smoothed, and AR spectral estimation"
```

---

## Task 5: Cross-Spectral Analysis

**Files:**
- Create: `src/spectral/cross.jl`

**Step 1: Create `src/spectral/cross.jl`**

```julia
# MacroEconometricModels.jl — Cross-Spectral Analysis

using FFTW, Statistics

"""
    cross_spectrum(y1::AbstractVector, y2::AbstractVector; method=:welch, kwargs...) -> CrossSpectrumResult

Estimate the cross-spectral density between two series.

Returns coherence, phase, and gain at each frequency.
"""
function cross_spectrum(y1::AbstractVector{T}, y2::AbstractVector{T};
    method::Symbol=:welch,
    window::Symbol=:hann,
    segments::Int=8,
    overlap::Real=0.5,
) where {T<:AbstractFloat}
    n = length(y1)
    length(y2) == n || throw(DimensionMismatch("y1 and y2 must have same length"))
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    if method == :welch
        _cross_spectrum_welch(y1, y2, segments, overlap, window)
    else
        throw(ArgumentError("method must be :welch"))
    end
end

cross_spectrum(y1::AbstractVector, y2::AbstractVector; kwargs...) =
    cross_spectrum(Float64.(y1), Float64.(y2); kwargs...)

"""Accessor: extract coherence vector from CrossSpectrumResult."""
coherence(csp::CrossSpectrumResult) = csp.coherence

"""Accessor: extract phase vector from CrossSpectrumResult."""
phase(csp::CrossSpectrumResult) = csp.phase

"""Accessor: extract gain vector from CrossSpectrumResult."""
gain(csp::CrossSpectrumResult) = csp.gain

# =============================================================================
# Welch Cross-Spectrum
# =============================================================================

function _cross_spectrum_welch(y1::AbstractVector{T}, y2::AbstractVector{T},
    segments::Int, overlap::Real, wtype::Symbol) where {T<:AbstractFloat}
    n = length(y1)
    seg_len = max(4, round(Int, n / (1 + (segments - 1) * (1 - overlap))))
    step = max(1, round(Int, seg_len * (1 - overlap)))
    n_freq = div(seg_len, 2) + 1
    freq = [T(2π * (j - 1) / seg_len) for j in 1:n_freq]

    win = T.(_spectral_window(seg_len, wtype))
    win_ss = sum(abs2, win)

    # Accumulate auto- and cross-spectra
    S11 = zeros(T, n_freq)
    S22 = zeros(T, n_freq)
    S12_real = zeros(T, n_freq)  # co-spectrum
    S12_imag = zeros(T, n_freq)  # quadrature spectrum

    n_segs = 0
    start = 1
    while start + seg_len - 1 <= n
        seg1 = (y1[start:start+seg_len-1] .- mean(y1[start:start+seg_len-1])) .* win
        seg2 = (y2[start:start+seg_len-1] .- mean(y2[start:start+seg_len-1])) .* win
        Y1 = FFTW.fft(seg1)
        Y2 = FFTW.fft(seg2)
        for j in 1:n_freq
            S11[j] += abs2(Y1[j]) / win_ss
            S22[j] += abs2(Y2[j]) / win_ss
            cross = Y1[j] * conj(Y2[j]) / win_ss
            S12_real[j] += real(cross)
            S12_imag[j] += imag(cross)
        end
        n_segs += 1
        start += step
    end
    n_segs < 1 && throw(ArgumentError("Not enough data for $segments segments"))

    S11 ./= n_segs
    S22 ./= n_segs
    S12_real ./= n_segs
    S12_imag ./= n_segs

    # Coherence = |S12|² / (S11 * S22)
    coh = zeros(T, n_freq)
    ph = zeros(T, n_freq)
    gn = zeros(T, n_freq)

    for j in 1:n_freq
        denom = S11[j] * S22[j]
        if denom > eps(T)
            cross_abs2 = S12_real[j]^2 + S12_imag[j]^2
            coh[j] = cross_abs2 / denom
            ph[j] = atan(S12_imag[j], S12_real[j])
            gn[j] = sqrt(cross_abs2) / sqrt(S11[j])
        end
    end

    CrossSpectrumResult{T}(freq, S12_real, S12_imag, coh, ph, gn, n)
end
```

**Step 2: Commit**

```bash
git add src/spectral/cross.jl
git commit -m "feat(spectral): add cross-spectral analysis with coherence, phase, gain"
```

---

## Task 6: Statistical Tests (Ljung-Box, Box-Pierce, Durbin-Watson, Fisher, Bartlett)

**Files:**
- Create: `src/teststat/portmanteau.jl` (Ljung-Box, Box-Pierce, Durbin-Watson)
- Create: `src/spectral/diagnostics.jl` (Fisher, Bartlett white noise, band_power)

**Step 1: Create `src/teststat/portmanteau.jl`**

```julia
# MacroEconometricModels.jl — Portmanteau and Autocorrelation Tests

using Distributions, Statistics

# =============================================================================
# Result Types
# =============================================================================

"""
    LjungBoxResult{T} <: StatsAPI.HypothesisTest

Ljung-Box Q-test for serial autocorrelation.
"""
struct LjungBoxResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    lags::Int
    nobs::Int
end

"""
    BoxPierceResult{T} <: StatsAPI.HypothesisTest

Box-Pierce Q-test for serial autocorrelation.
"""
struct BoxPierceResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    lags::Int
    nobs::Int
end

"""
    DurbinWatsonResult{T} <: StatsAPI.HypothesisTest

Durbin-Watson test for first-order autocorrelation in regression residuals.
"""
struct DurbinWatsonResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    nobs::Int
end

# StatsAPI interface
StatsAPI.nobs(r::LjungBoxResult) = r.nobs
StatsAPI.nobs(r::BoxPierceResult) = r.nobs
StatsAPI.nobs(r::DurbinWatsonResult) = r.nobs
StatsAPI.pvalue(r::LjungBoxResult) = r.pvalue
StatsAPI.pvalue(r::BoxPierceResult) = r.pvalue
StatsAPI.pvalue(r::DurbinWatsonResult) = r.pvalue
StatsAPI.dof(r::LjungBoxResult) = r.lags
StatsAPI.dof(r::BoxPierceResult) = r.lags
StatsAPI.dof(r::DurbinWatsonResult) = 1

# =============================================================================
# Ljung-Box Test
# =============================================================================

"""
    ljung_box_test(y::AbstractVector, lags::Int=10) -> LjungBoxResult

Ljung-Box Q-test for serial autocorrelation.

H₀: No autocorrelation up to lag `lags`
H₁: Autocorrelation present

Q = n(n+2) Σₖ ρ̂²ₖ/(n-k), distributed χ²(lags).
"""
function ljung_box_test(y::AbstractVector{T}, lags::Int=10) where {T<:AbstractFloat}
    n = length(y)
    lags < 1 && throw(ArgumentError("lags must be ≥ 1"))
    n < lags + 2 && throw(ArgumentError("Need at least lags+2 observations"))

    ȳ = mean(y)
    γ0 = sum((yi - ȳ)^2 for yi in y) / n

    Q = zero(T)
    if γ0 > eps(T)
        for k in 1:lags
            ρk = sum((y[t] - ȳ) * (y[t-k] - ȳ) for t in k+1:n) / (n * γ0)
            Q += ρk^2 / T(n - k)
        end
        Q *= T(n) * T(n + 2)
    end

    pval = one(T) - cdf(Chisq(lags), Q)
    LjungBoxResult{T}(Q, pval, lags, n)
end

ljung_box_test(y::AbstractVector, lags::Int=10) = ljung_box_test(Float64.(y), lags)

# =============================================================================
# Box-Pierce Test
# =============================================================================

"""
    box_pierce_test(y::AbstractVector, lags::Int=10) -> BoxPierceResult

Box-Pierce Q-test for serial autocorrelation.

H₀: No autocorrelation up to lag `lags`
H₁: Autocorrelation present

Q = n Σₖ ρ̂²ₖ, distributed χ²(lags) asymptotically.
"""
function box_pierce_test(y::AbstractVector{T}, lags::Int=10) where {T<:AbstractFloat}
    n = length(y)
    lags < 1 && throw(ArgumentError("lags must be ≥ 1"))
    n < lags + 2 && throw(ArgumentError("Need at least lags+2 observations"))

    ȳ = mean(y)
    γ0 = sum((yi - ȳ)^2 for yi in y) / n

    Q = zero(T)
    if γ0 > eps(T)
        for k in 1:lags
            ρk = sum((y[t] - ȳ) * (y[t-k] - ȳ) for t in k+1:n) / (n * γ0)
            Q += ρk^2
        end
        Q *= T(n)
    end

    pval = one(T) - cdf(Chisq(lags), Q)
    BoxPierceResult{T}(Q, pval, lags, n)
end

box_pierce_test(y::AbstractVector, lags::Int=10) = box_pierce_test(Float64.(y), lags)

# =============================================================================
# Durbin-Watson Test
# =============================================================================

"""
    durbin_watson_test(resid::AbstractVector, X::AbstractMatrix) -> DurbinWatsonResult

Durbin-Watson test for first-order autocorrelation in regression residuals.

H₀: No first-order autocorrelation (ρ = 0)

The p-value is computed via the Pan (1968) beta approximation.
"""
function durbin_watson_test(resid::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = length(resid)
    size(X, 1) == n || throw(DimensionMismatch("resid length must match rows of X"))

    # DW statistic: Σ(eₜ - eₜ₋₁)² / Σeₜ²
    num = sum((resid[t] - resid[t-1])^2 for t in 2:n)
    den = sum(abs2, resid)
    den < eps(T) && return DurbinWatsonResult{T}(T(2), one(T), n)
    dw = num / den

    # Beta approximation for p-value (Pan 1968)
    # E[DW] ≈ 2 and Var[DW] depends on X
    # Use Durbin-Watson bound tables via normal approximation
    k = size(X, 2)
    edw = T(2)  # E[d] under H0
    # Approximate variance: Var(d) ≈ 2(n-1)/((n-k)(n-k+2)) * (n² - k*n)/(n²)
    vdw = T(2) * T(n - 1) / (T(n - k) * T(n - k + 2))
    z = (dw - edw) / sqrt(vdw)
    pval = T(2) * (one(T) - cdf(Normal(), abs(z)))

    DurbinWatsonResult{T}(dw, pval, n)
end

durbin_watson_test(resid::AbstractVector, X::AbstractMatrix) =
    durbin_watson_test(Float64.(resid), Float64.(X))
```

**Step 2: Create `src/spectral/diagnostics.jl`**

```julia
# MacroEconometricModels.jl — Spectral Diagnostics

using FFTW, Statistics, Distributions

# =============================================================================
# Result Types (in teststat/ convention: <: StatsAPI.HypothesisTest)
# =============================================================================

"""
    FisherTestResult{T} <: StatsAPI.HypothesisTest

Fisher's test for hidden periodicities.
"""
struct FisherTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    peak_frequency::T
    peak_period::T
    nobs::Int
end

"""
    BartlettWhiteNoiseResult{T} <: StatsAPI.HypothesisTest

Bartlett's test for white noise via cumulative periodogram.
"""
struct BartlettWhiteNoiseResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    nobs::Int
end

# StatsAPI interface
StatsAPI.nobs(r::FisherTestResult) = r.nobs
StatsAPI.nobs(r::BartlettWhiteNoiseResult) = r.nobs
StatsAPI.pvalue(r::FisherTestResult) = r.pvalue
StatsAPI.pvalue(r::BartlettWhiteNoiseResult) = r.pvalue
StatsAPI.dof(r::FisherTestResult) = r.nobs
StatsAPI.dof(r::BartlettWhiteNoiseResult) = r.nobs

# =============================================================================
# Fisher's Test for Hidden Periodicities
# =============================================================================

"""
    fisher_test(y::AbstractVector) -> FisherTestResult

Fisher's exact test for hidden periodicities.

Tests whether the largest periodogram ordinate is significantly larger than
expected under white noise. The test statistic is g = max(I(ωⱼ)) / Σ I(ωⱼ).

Uses the exact distribution: P(g > x) = Σ_{k=1}^{⌊1/x⌋} (-1)^{k+1} C(m,k) (1-kx)^{m-1}
where m = (n-1)/2 (number of periodogram ordinates).
"""
function fisher_test(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 6 && throw(ArgumentError("Need at least 6 observations for Fisher's test"))

    y_centered = y .- mean(y)
    Y = FFTW.fft(y_centered)

    # Use only interior frequencies (exclude 0 and Nyquist)
    n_freq = div(n - 1, 2)
    I_vals = [abs2(Y[j+1]) for j in 1:n_freq]  # periodogram ordinates
    total = sum(I_vals)
    total < eps(T) && return FisherTestResult{T}(zero(T), one(T), zero(T), T(Inf), n)

    g = maximum(I_vals) / total
    peak_idx = argmax(I_vals)
    peak_freq = T(2π * peak_idx / n)
    peak_period = T(n) / peak_idx

    # Exact p-value
    m = n_freq
    pval = zero(T)
    for k in 1:floor(Int, 1 / g)
        pval += T((-1)^(k + 1)) * binomial(m, k) * (1 - k * g)^(m - 1)
    end
    pval = clamp(pval, zero(T), one(T))

    FisherTestResult{T}(g, pval, peak_freq, peak_period, n)
end

fisher_test(y::AbstractVector) = fisher_test(Float64.(y))

# =============================================================================
# Bartlett's White Noise Test
# =============================================================================

"""
    bartlett_white_noise_test(y::AbstractVector) -> BartlettWhiteNoiseResult

Bartlett's test for white noise based on the cumulative periodogram.

Under H₀ (white noise), the cumulative periodogram should follow a uniform line.
The test statistic is the Kolmogorov-Smirnov distance from the theoretical uniform.
"""
function bartlett_white_noise_test(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 6 && throw(ArgumentError("Need at least 6 observations"))

    y_centered = y .- mean(y)
    Y = FFTW.fft(y_centered)

    # Interior frequencies only
    n_freq = div(n - 1, 2)
    I_vals = [abs2(Y[j+1]) for j in 1:n_freq]
    total = sum(I_vals)
    total < eps(T) && return BartlettWhiteNoiseResult{T}(zero(T), one(T), n)

    # Cumulative periodogram
    cum = cumsum(I_vals) ./ total
    theoretical = [T(j) / n_freq for j in 1:n_freq]

    # KS statistic
    D = maximum(abs.(cum .- theoretical))

    # Asymptotic p-value: Kolmogorov distribution
    # P(D > x) ≈ 2 Σ (-1)^{k+1} exp(-2k²n_freq²x²)
    sqrt_m = sqrt(T(n_freq))
    z = (sqrt_m + T(0.12) + T(0.11) / sqrt_m) * D
    pval = zero(T)
    for k in 1:100
        term = T((-1)^(k + 1)) * T(2) * exp(-T(2) * k^2 * z^2)
        pval += term
        abs(term) < eps(T) && break
    end
    pval = clamp(pval, zero(T), one(T))

    BartlettWhiteNoiseResult{T}(D, pval, n)
end

bartlett_white_noise_test(y::AbstractVector) = bartlett_white_noise_test(Float64.(y))

# =============================================================================
# Band Power
# =============================================================================

"""
    band_power(sp::SpectralDensityResult, band::Tuple{Real,Real}) -> Float64

Compute the fraction of total variance in the frequency band (ω_lo, ω_hi).

For business cycle analysis: `band_power(sp, (2π/32, 2π/6))` gives the
share of variance at 6-32 quarter periodicities.

Arguments:
- `sp`: Spectral density result
- `band`: Tuple of (lower_freq, upper_freq) in radians
"""
function band_power(sp::SpectralDensityResult{T}, band::Tuple{Real,Real}) where {T<:AbstractFloat}
    lo, hi = T(band[1]), T(band[2])
    lo < hi || throw(ArgumentError("Lower frequency must be < upper frequency"))

    total = zero(T)
    in_band = zero(T)
    for j in 1:length(sp.freq)
        total += sp.density[j]
        if sp.freq[j] >= lo && sp.freq[j] <= hi
            in_band += sp.density[j]
        end
    end
    total < eps(T) ? zero(T) : in_band / total
end
```

**Step 3: Commit**

```bash
git add src/teststat/portmanteau.jl src/spectral/diagnostics.jl
git commit -m "feat: add Ljung-Box, Box-Pierce, Durbin-Watson, Fisher, Bartlett tests"
```

---

## Task 7: Frequency-Domain Filtering

**Files:**
- Create: `src/spectral/filtering.jl`

**Step 1: Create `src/spectral/filtering.jl`**

```julia
# MacroEconometricModels.jl — Frequency-Domain Filtering

using FFTW, Statistics

"""
    ideal_bandpass(y::AbstractVector, band::Tuple{Real,Real}) -> Vector

Apply an ideal (brick-wall) band-pass filter via FFT.

Zeroes out all frequencies outside the band (ω_lo, ω_hi), then inverse-transforms.

# Arguments
- `y`: Time series vector
- `band`: Tuple of (lower_freq, upper_freq) in radians (0 to π)
"""
function ideal_bandpass(y::AbstractVector{T}, band::Tuple{Real,Real}) where {T<:AbstractFloat}
    n = length(y)
    lo, hi = T(band[1]), T(band[2])
    lo < hi || throw(ArgumentError("Lower frequency must be < upper frequency"))

    y_centered = y .- mean(y)
    Y = FFTW.fft(y_centered)

    # Zero out frequencies outside band
    for j in 1:n
        freq_j = T(2π * (j - 1) / n)
        # Handle both positive and negative frequencies (symmetric)
        freq_j = freq_j > T(π) ? T(2π) - freq_j : freq_j
        if freq_j < lo || freq_j > hi
            Y[j] = zero(Complex{T})
        end
    end

    real.(FFTW.ifft(Y))
end

ideal_bandpass(y::AbstractVector, band::Tuple{Real,Real}) = ideal_bandpass(Float64.(y), band)

"""
    transfer_function(filter::Symbol; kwargs...) -> TransferFunctionResult

Compute the frequency response (gain and phase) of a time-series filter.

# Supported Filters
- `:hp` — Hodrick-Prescott (kwarg: `lambda=1600`)
- `:bk` — Baxter-King band-pass (kwargs: `pl=6, pu=32, K=12`)
- `:hamilton` — Hamilton (2018) (kwargs: `h=8, p=4`)

# Example
```julia
tf = transfer_function(:hp; lambda=1600)
plot_result(tf)
```
"""
function transfer_function(filter::Symbol; n_freq::Int=512, kwargs...)
    freq = [Float64(π * (j - 1) / (n_freq - 1)) for j in 1:n_freq]

    if filter == :hp
        _transfer_hp(freq; kwargs...)
    elseif filter == :bk
        _transfer_bk(freq; kwargs...)
    elseif filter == :hamilton
        _transfer_hamilton(freq; kwargs...)
    else
        throw(ArgumentError("filter must be :hp, :bk, or :hamilton"))
    end
end

# =============================================================================
# HP Filter Transfer Function
# =============================================================================

function _transfer_hp(freq::Vector{Float64}; lambda::Real=1600)
    λ = Float64(lambda)
    n = length(freq)
    gn = zeros(Float64, n)
    ph = zeros(Float64, n)

    for j in 1:n
        ω = freq[j]
        if ω < eps(Float64)
            gn[j] = 0.0  # HP filter removes DC
        else
            # HP gain: 4λ(1-cos(ω))² / (1 + 4λ(1-cos(ω))²)
            x = 4λ * (1 - cos(ω))^2
            gn[j] = x / (1 + x)
        end
        # HP is a symmetric (zero-phase) filter
        ph[j] = 0.0
    end

    TransferFunctionResult{Float64}(freq, gn, ph, :hp)
end

# =============================================================================
# Baxter-King Transfer Function
# =============================================================================

function _transfer_bk(freq::Vector{Float64}; pl::Int=6, pu::Int=32, K::Int=12)
    ωl = 2π / pu  # low cutoff (high period = low freq)
    ωh = 2π / pl  # high cutoff (low period = high freq)
    n = length(freq)
    gn = zeros(Float64, n)
    ph = zeros(Float64, n)

    # BK weights
    bk = zeros(Float64, 2K + 1)
    for j in -K:K
        if j == 0
            bk[j+K+1] = (ωh - ωl) / π
        else
            bk[j+K+1] = (sin(ωh * j) - sin(ωl * j)) / (π * j)
        end
    end
    # Normalize so weights sum to zero (band-pass)
    bk .-= sum(bk) / (2K + 1)

    for j in 1:n
        ω = freq[j]
        # Frequency response: H(ω) = Σ bₖ e^{-iωk}
        H = sum(bk[k+K+1] * exp(-im * ω * k) for k in -K:K)
        gn[j] = abs(H)
        ph[j] = angle(H)
    end

    TransferFunctionResult{Float64}(freq, gn, ph, :bk)
end

# =============================================================================
# Hamilton Transfer Function
# =============================================================================

function _transfer_hamilton(freq::Vector{Float64}; h::Int=8, p::Int=4)
    n = length(freq)
    gn = zeros(Float64, n)
    ph = zeros(Float64, n)

    # Hamilton filter: yₜ = β₀ + β₁yₜ₋ₕ + ... + βₚyₜ₋ₕ₋ₚ₊₁ + eₜ
    # Residual = (1 - projection onto lagged values) → high-pass-like
    # Approximate gain: |1 - H(ω)| where H is the regression transfer function
    for j in 1:n
        ω = freq[j]
        # Simplified: Hamilton filter suppresses low frequencies (periods > 2h)
        # Gain ≈ 1 for business cycle frequencies, ≈ 0 for trend
        cutoff = 2π / (2h)
        if ω < cutoff
            gn[j] = (ω / cutoff)^2  # gradual rolloff
        else
            gn[j] = 1.0
        end
        ph[j] = 0.0  # approximate
    end

    TransferFunctionResult{Float64}(freq, gn, ph, :hamilton)
end
```

**Step 2: Commit**

```bash
git add src/spectral/filtering.jl
git commit -m "feat(spectral): add ideal bandpass filter and transfer function analysis"
```

---

## Task 8: Wire Module Into Main Package + Display Methods

**Files:**
- Modify: `src/MacroEconometricModels.jl` (includes + exports)
- Create: `src/spectral/show.jl` (Base.show methods)

**Step 1: Create `src/spectral/show.jl`**

```julia
# MacroEconometricModels.jl — Display Methods for Spectral Types

function Base.show(io::IO, r::ACFResult{T}) where {T}
    has_acf = any(!iszero, r.acf)
    has_pacf = any(!iszero, r.pacf)
    has_ccf = r.ccf !== nothing

    if has_ccf
        println(io)
        _pretty_table(io, Any["Observations" r.nobs; "Max lag" maximum(abs, r.lags); "95% CI" "±$(round(r.ci, digits=4))"];
            title = "Cross-Correlation Function",
            column_labels = ["Specification", ""],
            alignment = [:l, :r],
        )
        n_show = min(length(r.lags), 41)  # show up to 41 lags
        ccf_data = Matrix{Any}(undef, n_show, 2)
        for i in 1:n_show
            ccf_data[i, 1] = r.lags[i]
            ccf_data[i, 2] = round(r.ccf[i], digits=4)
        end
        _pretty_table(io, ccf_data;
            column_labels = ["Lag", "CCF"],
            alignment = [:r, :r],
        )
        return
    end

    title = if has_acf && has_pacf
        "Correlogram"
    elseif has_acf
        "Autocorrelation Function"
    else
        "Partial Autocorrelation Function"
    end

    println(io)
    _pretty_table(io, Any["Observations" r.nobs; "Lags" length(r.lags); "95% CI" "±$(round(r.ci, digits=4))"];
        title = title,
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    maxlag = length(r.lags)
    n_show = min(maxlag, 36)

    if has_acf && has_pacf
        data = Matrix{Any}(undef, n_show, 5)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = round(r.acf[i], digits=4)
            data[i, 3] = round(r.pacf[i], digits=4)
            data[i, 4] = round(r.q_stats[i], digits=2)
            data[i, 5] = _format_pvalue(r.q_pvalues[i])
        end
        _pretty_table(io, data;
            column_labels = ["Lag", "AC", "PAC", "Q-Stat", "Prob"],
            alignment = [:r, :r, :r, :r, :r],
        )
    elseif has_acf
        data = Matrix{Any}(undef, n_show, 4)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = round(r.acf[i], digits=4)
            data[i, 3] = round(r.q_stats[i], digits=2)
            data[i, 4] = _format_pvalue(r.q_pvalues[i])
        end
        _pretty_table(io, data;
            column_labels = ["Lag", "AC", "Q-Stat", "Prob"],
            alignment = [:r, :r, :r, :r],
        )
    else
        data = Matrix{Any}(undef, n_show, 2)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = round(r.pacf[i], digits=4)
        end
        _pretty_table(io, data;
            column_labels = ["Lag", "PAC"],
            alignment = [:r, :r],
        )
    end
end

function Base.show(io::IO, r::SpectralDensityResult{T}) where {T}
    println(io)
    _pretty_table(io, Any[
        "Method" string(r.method);
        "Bandwidth" round(r.bandwidth, digits=1);
        "Observations" r.nobs;
        "Frequencies" length(r.freq)
    ];
        title = "Spectral Density Estimate",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Show top 20 frequency points (evenly spaced)
    nf = length(r.freq)
    n_show = min(nf, 20)
    indices = round.(Int, range(1, nf, length=n_show))
    data = Matrix{Any}(undef, n_show, 5)
    for (i, j) in enumerate(indices)
        freq_j = r.freq[j]
        period = freq_j > eps(T) ? T(2π) / freq_j : T(Inf)
        data[i, 1] = round(freq_j, digits=4)
        data[i, 2] = period < T(1e6) ? round(period, digits=1) : "Inf"
        data[i, 3] = round(r.density[j], sigdigits=4)
        data[i, 4] = round(r.ci_lower[j], sigdigits=4)
        data[i, 5] = round(r.ci_upper[j], sigdigits=4)
    end
    _pretty_table(io, data;
        column_labels = ["Freq", "Period", "Density", "CI Low", "CI High"],
        alignment = [:r, :r, :r, :r, :r],
    )
end

function Base.show(io::IO, r::CrossSpectrumResult{T}) where {T}
    println(io)
    _pretty_table(io, Any["Observations" r.nobs; "Frequencies" length(r.freq)];
        title = "Cross-Spectral Analysis",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    nf = length(r.freq)
    n_show = min(nf, 20)
    indices = round.(Int, range(1, nf, length=n_show))
    data = Matrix{Any}(undef, n_show, 4)
    for (i, j) in enumerate(indices)
        data[i, 1] = round(r.freq[j], digits=4)
        data[i, 2] = round(r.coherence[j], digits=4)
        data[i, 3] = round(r.phase[j], digits=4)
        data[i, 4] = round(r.gain[j], digits=4)
    end
    _pretty_table(io, data;
        column_labels = ["Freq", "Coherence", "Phase", "Gain"],
        alignment = [:r, :r, :r, :r],
    )
end

function Base.show(io::IO, r::TransferFunctionResult{T}) where {T}
    println(io)
    _pretty_table(io, Any["Filter" string(r.filter); "Frequencies" length(r.freq)];
        title = "Filter Transfer Function",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    nf = length(r.freq)
    n_show = min(nf, 20)
    indices = round.(Int, range(1, nf, length=n_show))
    data = Matrix{Any}(undef, n_show, 3)
    for (i, j) in enumerate(indices)
        data[i, 1] = round(r.freq[j], digits=4)
        data[i, 2] = round(r.gain[j], digits=4)
        data[i, 3] = round(r.phase[j], digits=4)
    end
    _pretty_table(io, data;
        column_labels = ["Freq", "Gain", "Phase"],
        alignment = [:r, :r, :r],
    )
end

# Show methods for test results
function Base.show(io::IO, r::LjungBoxResult)
    println(io)
    _pretty_table(io, Any[
        "H₀" "No serial autocorrelation";
        "H₁" "Autocorrelation present";
        "Lags" r.lags;
        "Observations" r.nobs
    ];
        title = "Ljung-Box Q-Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    _pretty_table(io, Any[
        "Q-statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ];
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::BoxPierceResult)
    println(io)
    _pretty_table(io, Any[
        "H₀" "No serial autocorrelation";
        "H₁" "Autocorrelation present";
        "Lags" r.lags;
        "Observations" r.nobs
    ];
        title = "Box-Pierce Q-Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    _pretty_table(io, Any[
        "Q-statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ];
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::DurbinWatsonResult)
    println(io)
    _pretty_table(io, Any[
        "H₀" "No first-order autocorrelation";
        "Observations" r.nobs
    ];
        title = "Durbin-Watson Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    _pretty_table(io, Any[
        "DW statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ];
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::FisherTestResult)
    println(io)
    _pretty_table(io, Any[
        "H₀" "No hidden periodicities (white noise)";
        "H₁" "Sinusoidal component present";
        "Observations" r.nobs
    ];
        title = "Fisher's Test for Hidden Periodicities",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    _pretty_table(io, Any[
        "g statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue);
        "Peak frequency" round(r.peak_frequency, digits=4);
        "Peak period" round(r.peak_period, digits=1)
    ];
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::BartlettWhiteNoiseResult)
    println(io)
    _pretty_table(io, Any[
        "H₀" "Series is white noise";
        "H₁" "Series has structure (not white noise)";
        "Observations" r.nobs
    ];
        title = "Bartlett's White Noise Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    _pretty_table(io, Any[
        "KS statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ];
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
end
```

**Step 2: Add includes and exports to `src/MacroEconometricModels.jl`**

After line 68 (`using SparseArrays`), add:
```julia
using FFTW
```

After line 135 (`include("teststat/normality.jl")`), add:
```julia
include("teststat/portmanteau.jl")
```

After line 155 (`include("factor/generalized.jl")`), add the spectral module block:
```julia
# Spectral analysis and ACF/PACF
include("spectral/types.jl")
include("spectral/windows.jl")
include("spectral/acf.jl")
include("spectral/estimation.jl")
include("spectral/cross.jl")
include("spectral/diagnostics.jl")
include("spectral/filtering.jl")
include("spectral/show.jl")
```

Add new exports after the "Volatility Models" section (after line 745):
```julia
# =============================================================================
# Exports - Spectral Analysis & ACF/PACF
# =============================================================================

# Result types
export ACFResult, SpectralDensityResult, CrossSpectrumResult, TransferFunctionResult

# ACF/PACF
export acf, pacf, acf_pacf, ccf

# Spectral density
export periodogram, spectral_density

# Cross-spectral
export cross_spectrum, coherence, phase, gain

# Diagnostics
export band_power

# Filtering
export ideal_bandpass, transfer_function

# Test types and functions
export LjungBoxResult, BoxPierceResult, DurbinWatsonResult
export FisherTestResult, BartlettWhiteNoiseResult
export ljung_box_test, box_pierce_test, durbin_watson_test
export fisher_test, bartlett_white_noise_test
```

**Step 3: Verify the package loads**

```bash
julia --project=. -e 'using MacroEconometricModels; println("OK")'
```

Expected: "OK" — no errors.

**Step 4: Commit**

```bash
git add src/spectral/show.jl src/MacroEconometricModels.jl src/teststat/portmanteau.jl
git commit -m "feat: wire spectral module into main package with includes, exports, display"
```

---

## Task 9: TimeSeriesData/PanelData Dispatch Wrappers

**Files:**
- Modify: `src/data/convert.jl` (append at end)

**Step 1: Add dispatch wrappers to `src/data/convert.jl`**

Append the following at the end of the file:

```julia
# =============================================================================
# Spectral / ACF dispatch wrappers (TimeSeriesData → Vector)
# =============================================================================

function acf(d::TimeSeriesData, maxlag::Int=20; var=nothing, kwargs...)
    v = var === nothing ? to_vector(d) : to_vector(d, var isa Symbol ? string(var) : var)
    acf(v, maxlag; kwargs...)
end

function pacf(d::TimeSeriesData, maxlag::Int=20; var=nothing, kwargs...)
    v = var === nothing ? to_vector(d) : to_vector(d, var isa Symbol ? string(var) : var)
    pacf(v, maxlag; kwargs...)
end

function acf_pacf(d::TimeSeriesData, maxlag::Int=20; var=nothing, kwargs...)
    v = var === nothing ? to_vector(d) : to_vector(d, var isa Symbol ? string(var) : var)
    acf_pacf(v, maxlag; kwargs...)
end

function spectral_density(d::TimeSeriesData; var=nothing, kwargs...)
    v = var === nothing ? to_vector(d) : to_vector(d, var isa Symbol ? string(var) : var)
    spectral_density(v; kwargs...)
end

function periodogram(d::TimeSeriesData; var=nothing)
    v = var === nothing ? to_vector(d) : to_vector(d, var isa Symbol ? string(var) : var)
    periodogram(v)
end

# PanelData dispatch: per-group results
function acf(d::PanelData, maxlag::Int=20; var=nothing, kwargs...)
    grps = group_data(d)
    result = Dict{Any,ACFResult}()
    _var = var === nothing ? 1 : (var isa Symbol ? string(var) : var)
    for (gid, gd) in grps
        v = to_vector(gd, _var)
        result[gid] = acf(v, maxlag; kwargs...)
    end
    result
end

function spectral_density(d::PanelData; var=nothing, kwargs...)
    grps = group_data(d)
    result = Dict{Any,SpectralDensityResult}()
    _var = var === nothing ? 1 : (var isa Symbol ? string(var) : var)
    for (gid, gd) in grps
        v = to_vector(gd, _var)
        result[gid] = spectral_density(v; kwargs...)
    end
    result
end
```

**Step 2: Commit**

```bash
git add src/data/convert.jl
git commit -m "feat: add TimeSeriesData/PanelData dispatch for ACF and spectral functions"
```

---

## Task 10: Plotting — ACF, Spectral Density, Cross-Spectrum, Transfer Function

**Files:**
- Create: `src/plotting/spectral.jl`
- Modify: `src/MacroEconometricModels.jl` (add include for plotting/spectral.jl)

**Step 1: Create `src/plotting/spectral.jl`**

```julia
# MacroEconometricModels.jl — Spectral Analysis Plots

# =============================================================================
# ACF/PACF Bar Chart
# =============================================================================

"""
    plot_result(r::ACFResult; title="", save_path=nothing)

Plot ACF and/or PACF as side-by-side bar charts with confidence bands.
For CCF results, plots cross-correlation bars.
"""
function plot_result(r::ACFResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]

    if r.ccf !== nothing
        # CCF plot
        id = _next_plot_id("ccf")
        data_json = _bar_data_json(r.lags, r.ccf, r.ci)
        js = _render_bar_js(id, data_json; xlabel="Lag", ylabel="CCF",
                            ci_val=r.ci)
        push!(panels, _PanelSpec(id, "Cross-Correlation Function", js))
    else
        has_acf = any(!iszero, r.acf)
        has_pacf = any(!iszero, r.pacf)

        if has_acf
            id = _next_plot_id("acf")
            data_json = _bar_data_json(r.lags, r.acf, r.ci)
            js = _render_bar_js(id, data_json; xlabel="Lag", ylabel="ACF",
                                ci_val=r.ci)
            push!(panels, _PanelSpec(id, "Autocorrelation", js))
        end

        if has_pacf
            id = _next_plot_id("pacf")
            data_json = _bar_data_json(r.lags, r.pacf, r.ci)
            js = _render_bar_js(id, data_json; xlabel="Lag", ylabel="PACF",
                                ci_val=r.ci)
            push!(panels, _PanelSpec(id, "Partial Autocorrelation", js))
        end
    end

    isempty(title) && (title = r.ccf !== nothing ? "Cross-Correlation Function" : "Correlogram")
    p = _make_plot(panels; title=title, ncols=min(length(panels), 2))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Spectral Density Plot
# =============================================================================

"""
    plot_result(r::SpectralDensityResult; title="", save_path=nothing, log_scale=true)

Plot log-spectral density with confidence band.
"""
function plot_result(r::SpectralDensityResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing,
                     log_scale::Bool=true) where {T}
    id = _next_plot_id("spec")

    density = log_scale ? max.(log10.(max.(r.density, eps(T))), T(-20)) : r.density
    ci_lo = log_scale ? max.(log10.(max.(r.ci_lower, eps(T))), T(-20)) : r.ci_lower
    ci_hi = log_scale ? max.(log10.(max.(r.ci_upper, eps(T))), T(-20)) : r.ci_upper

    data_json = _spectral_line_data_json(r.freq, density, ci_lo, ci_hi)
    ylabel = log_scale ? "log₁₀(Spectral Density)" : "Spectral Density"
    js = _render_ci_line_js(id, data_json; xlabel="Frequency (radians)", ylabel=ylabel)
    panel = _PanelSpec(id, "Spectral Density ($(r.method))", js)

    isempty(title) && (title = "Spectral Density Estimate ($(r.method))")
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Cross-Spectrum Plot
# =============================================================================

"""
    plot_result(r::CrossSpectrumResult; title="", save_path=nothing)

Plot coherence and phase as 2-panel figure.
"""
function plot_result(r::CrossSpectrumResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Coherence
    id1 = _next_plot_id("coh")
    data1 = _spectral_simple_data_json(r.freq, r.coherence)
    js1 = _render_simple_line_js(id1, data1; xlabel="Frequency", ylabel="Squared Coherency")
    p1 = _PanelSpec(id1, "Coherence", js1)

    # Panel 2: Phase
    id2 = _next_plot_id("phase")
    data2 = _spectral_simple_data_json(r.freq, r.phase)
    js2 = _render_simple_line_js(id2, data2; xlabel="Frequency", ylabel="Phase (radians)")
    p2 = _PanelSpec(id2, "Phase Spectrum", js2)

    isempty(title) && (title = "Cross-Spectral Analysis")
    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Transfer Function Plot
# =============================================================================

"""
    plot_result(r::TransferFunctionResult; title="", save_path=nothing)

Plot filter gain and phase as 2-panel figure.
"""
function plot_result(r::TransferFunctionResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Gain
    id1 = _next_plot_id("tf_gain")
    data1 = _spectral_simple_data_json(r.freq, r.gain)
    # Add ideal band-pass reference line for BK
    refs = r.filter == :bk ? "[{\"value\":1,\"color\":\"#999\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]" : "[]"
    js1 = _render_simple_line_js(id1, data1; xlabel="Frequency", ylabel="Gain",
                                  ref_lines_json=refs)
    p1 = _PanelSpec(id1, "Gain", js1)

    # Panel 2: Phase
    id2 = _next_plot_id("tf_phase")
    data2 = _spectral_simple_data_json(r.freq, r.phase)
    js2 = _render_simple_line_js(id2, data2; xlabel="Frequency", ylabel="Phase (radians)")
    p2 = _PanelSpec(id2, "Phase", js2)

    filter_name = uppercasefirst(string(r.filter))
    isempty(title) && (title = "$filter_name Filter — Transfer Function")
    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Plot Helpers (spectral-specific)
# =============================================================================

"""Build JSON data for bar charts (ACF/PACF)."""
function _bar_data_json(lags::Vector{Int}, values::AbstractVector, ci::Real)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(lags)
        push!(rows, [
            "x" => _json(lags[i]),
            "y" => _json(values[i]),
            "ci_hi" => _json(ci),
            "ci_lo" => _json(-ci),
        ])
    end
    _json_array_of_objects(rows)
end

"""Build JSON data for spectral line with CI band."""
function _spectral_line_data_json(freq, density, ci_lo, ci_hi)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        push!(rows, [
            "x" => _json(freq[i]),
            "y" => _json(density[i]),
            "ci_lo" => _json(ci_lo[i]),
            "ci_hi" => _json(ci_hi[i]),
        ])
    end
    _json_array_of_objects(rows)
end

"""Build JSON data for simple line plot."""
function _spectral_simple_data_json(freq, values)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        push!(rows, [
            "x" => _json(freq[i]),
            "y" => _json(values[i]),
        ])
    end
    _json_array_of_objects(rows)
end

"""Render a bar chart with CI reference lines (for ACF/PACF)."""
function _render_bar_js(id::String, data_json::String;
                        xlabel::String="", ylabel::String="",
                        ci_val::Real=0.0)
    ci_refs = ci_val > 0 ? "[{\"value\":$(ci_val),\"color\":\"#d62728\",\"dash\":\"6,3\"},{\"value\":$(-ci_val),\"color\":\"#d62728\",\"dash\":\"6,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"2,2\"}]" : "[]"
    _render_bar_chart_js(id, data_json;
                          xlabel=xlabel, ylabel=ylabel,
                          ref_lines_json=ci_refs,
                          bar_color=_PLOT_COLORS[1])
end

"""Render a bar chart using D3.js."""
function _render_bar_chart_js(id::String, data_json::String;
                               xlabel::String="", ylabel::String="",
                               ref_lines_json::String="[]",
                               bar_color::String=_PLOT_COLORS[1])
    """
    (function(){
        var data = $data_json;
        var refs = $ref_lines_json;
        var c = document.getElementById("$id");
        var W = c.clientWidth, H = c.clientHeight;
        var m = {t:20,r:30,b:40,l:50}, w = W-m.l-m.r, h = H-m.t-m.b;
        var svg = d3.select(c).append("svg").attr("width",W).attr("height",H)
                    .append("g").attr("transform","translate("+m.l+","+m.t+")");

        var x = d3.scaleBand().domain(data.map(d=>d.x)).range([0,w]).padding(0.3);
        var yExt = d3.extent(data, d=>d.y);
        var yPad = (yExt[1]-yExt[0])*0.1 || 0.2;
        var y = d3.scaleLinear().domain([Math.min(yExt[0]-yPad, -0.15), Math.max(yExt[1]+yPad, 0.15)]).range([h,0]);

        svg.append("g").attr("transform","translate(0,"+h+")").call(d3.axisBottom(x).tickValues(x.domain().filter((d,i)=>i%Math.max(1,Math.floor(data.length/15))===0)));
        svg.append("g").call(d3.axisLeft(y).ticks(6));
        if("$xlabel") svg.append("text").attr("x",w/2).attr("y",h+35).attr("text-anchor","middle").style("font-size","12px").text("$xlabel");
        if("$ylabel") svg.append("text").attr("transform","rotate(-90)").attr("x",-h/2).attr("y",-40).attr("text-anchor","middle").style("font-size","12px").text("$ylabel");

        svg.selectAll(".bar").data(data).enter().append("rect")
            .attr("x", d=>x(d.x)+x.bandwidth()/2-2)
            .attr("y", d=>d.y>=0?y(d.y):y(0))
            .attr("width", 4)
            .attr("height", d=>Math.abs(y(0)-y(d.y)))
            .attr("fill", "$bar_color");

        refs.forEach(function(r){
            svg.append("line").attr("x1",0).attr("x2",w)
                .attr("y1",y(r.value)).attr("y2",y(r.value))
                .attr("stroke",r.color).attr("stroke-dasharray",r.dash).attr("stroke-width",1);
        });
    })();
    """
end

"""Render a simple line chart."""
function _render_simple_line_js(id::String, data_json::String;
                                 xlabel::String="", ylabel::String="",
                                 ref_lines_json::String="[]")
    s = _series_json([""], [_PLOT_COLORS[1]]; keys=["y"])
    _render_line_js(id, data_json, s;
                    ref_lines_json=ref_lines_json,
                    xlabel=xlabel, ylabel=ylabel)
end

"""Render a line chart with CI shading (for spectral density)."""
function _render_ci_line_js(id::String, data_json::String;
                             xlabel::String="", ylabel::String="")
    s = _series_json(["Estimate"], [_PLOT_COLORS[1]]; keys=["y"])
    ci_s = "[{\"lo\":\"ci_lo\",\"hi\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
    _render_line_js(id, data_json, s;
                    ci_bands_json=ci_s,
                    xlabel=xlabel, ylabel=ylabel)
end
```

**Step 2: Add include to `src/MacroEconometricModels.jl`**

After the line `include("plotting/reg.jl")` (line 328), add:
```julia
include("plotting/spectral.jl")
```

**Step 3: Verify package loads**

```bash
julia --project=. -e 'using MacroEconometricModels; println("OK")'
```

**Step 4: Commit**

```bash
git add src/plotting/spectral.jl src/MacroEconometricModels.jl
git commit -m "feat: add plot_result dispatches for ACF, spectral density, cross-spectrum, transfer function"
```

---

## Task 11: Tests

**Files:**
- Create: `test/spectral/test_spectral.jl`
- Modify: `test/runtests.jl` (add to test groups)

**Step 1: Create test directory and file**

```bash
mkdir -p test/spectral
```

**Step 2: Create `test/spectral/test_spectral.jl`**

```julia
using Test, MacroEconometricModels, Random, Statistics

@testset "Spectral Analysis & ACF/PACF" begin
    rng = Random.MersenneTwister(42)

    # =========================================================================
    # ACF
    # =========================================================================
    @testset "ACF" begin
        # White noise: ACF should be near zero for lag > 0
        wn = randn(rng, 1000)
        r = acf(wn, 20)
        @test length(r.acf) == 20
        @test all(abs.(r.acf) .< 3 * r.ci)  # within 3× CI band
        @test r.nobs == 1000

        # AR(1) with ρ=0.8: ACF(1) ≈ 0.8
        y_ar = zeros(500)
        y_ar[1] = randn(rng)
        for t in 2:500
            y_ar[t] = 0.8 * y_ar[t-1] + randn(rng)
        end
        r_ar = acf(y_ar, 10)
        @test abs(r_ar.acf[1] - 0.8) < 0.1
        @test abs(r_ar.acf[2] - 0.64) < 0.15
    end

    # =========================================================================
    # PACF
    # =========================================================================
    @testset "PACF" begin
        rng2 = Random.MersenneTwister(123)
        # AR(2): PACF should cut off after lag 2
        y = zeros(1000)
        y[1:2] = randn(rng2, 2)
        for t in 3:1000
            y[t] = 0.5 * y[t-1] - 0.3 * y[t-2] + randn(rng2)
        end
        r = pacf(y, 10; method=:levinson)
        @test abs(r.pacf[1] - 0.5) < 0.1
        @test abs(r.pacf[2] - (-0.3)) < 0.1
        @test all(abs.(r.pacf[3:end]) .< 3 / sqrt(1000))

        # Levinson vs OLS should agree
        r_ols = pacf(y, 10; method=:ols)
        @test maximum(abs.(r.pacf .- r_ols.pacf)) < 0.05
    end

    # =========================================================================
    # ACF_PACF combined
    # =========================================================================
    @testset "acf_pacf" begin
        rng3 = Random.MersenneTwister(456)
        y = randn(rng3, 200)
        r = acf_pacf(y, 15)
        @test length(r.acf) == 15
        @test length(r.pacf) == 15
        @test length(r.q_stats) == 15
        @test all(r.q_pvalues .> 0.01)  # white noise should pass
    end

    # =========================================================================
    # CCF
    # =========================================================================
    @testset "CCF" begin
        rng4 = Random.MersenneTwister(789)
        x = randn(rng4, 200)
        y = [0.0; x[1:end-1]] + 0.3 * randn(rng4, 200)  # y lags x by 1
        r = ccf(x, y, 10)
        @test length(r.ccf) == 21  # -10:10
        @test r.ccf !== nothing
    end

    # =========================================================================
    # Periodogram
    # =========================================================================
    @testset "Periodogram" begin
        # Sinusoid: peak at planted frequency
        n = 256
        t = 1:n
        freq_planted = 2π * 10 / n  # 10 cycles
        y = sin.(freq_planted .* t)
        sp = periodogram(y)
        peak_idx = argmax(sp.density[2:end]) + 1  # skip DC
        @test abs(sp.freq[peak_idx] - freq_planted) < 2π / n  # within one freq bin

        # Parseval's theorem: Σ I(ωⱼ) ≈ variance * something
        wn = randn(Random.MersenneTwister(111), 128)
        sp_wn = periodogram(wn)
        @test sp_wn.method == :periodogram
        @test length(sp_wn.density) == 65  # n/2 + 1
    end

    # =========================================================================
    # Spectral Density (Welch)
    # =========================================================================
    @testset "Welch" begin
        rng5 = Random.MersenneTwister(222)
        y = randn(rng5, 512)
        sp = spectral_density(y; method=:welch, segments=8, window=:hann)
        @test sp.method == :welch
        @test length(sp.density) > 0
        @test all(sp.ci_lower .<= sp.density)
        @test all(sp.ci_upper .>= sp.density)
    end

    # =========================================================================
    # Spectral Density (AR)
    # =========================================================================
    @testset "AR spectral" begin
        rng6 = Random.MersenneTwister(333)
        # Generate AR(1) with ρ=0.9
        y = zeros(500)
        y[1] = randn(rng6)
        for t in 2:500
            y[t] = 0.9 * y[t-1] + randn(rng6)
        end
        sp = spectral_density(y; method=:ar)
        @test sp.method == :ar
        # AR(1) spectrum should peak at ω=0
        @test argmax(sp.density) <= 3  # near frequency 0
    end

    # =========================================================================
    # Smoothed Periodogram
    # =========================================================================
    @testset "Smoothed" begin
        rng7 = Random.MersenneTwister(444)
        y = randn(rng7, 256)
        sp = spectral_density(y; method=:smoothed, kernel=:daniell, bandwidth=5)
        @test sp.method == :smoothed
        @test length(sp.density) > 0
    end

    # =========================================================================
    # Cross-Spectrum
    # =========================================================================
    @testset "Cross-spectrum" begin
        rng8 = Random.MersenneTwister(555)
        x = randn(rng8, 256)
        y = 2.0 .* x .+ 0.1 .* randn(rng8, 256)  # y ≈ 2x
        csp = cross_spectrum(x, y; method=:welch, segments=4)
        @test all(coherence(csp) .>= 0)
        @test all(coherence(csp) .<= 1.001)  # allow tiny float error
        # High coherence expected
        @test mean(coherence(csp)) > 0.5

        # Accessors return vectors
        @test phase(csp) isa Vector
        @test gain(csp) isa Vector
    end

    # =========================================================================
    # Window Functions
    # =========================================================================
    @testset "Windows" begin
        for wtype in [:rectangular, :bartlett, :hann, :hamming, :blackman, :tukey, :flat_top]
            w = MacroEconometricModels._spectral_window(64, wtype)
            @test length(w) == 64
            # Symmetric
            @test w ≈ reverse(w) atol=1e-12
        end
        # Single element
        w1 = MacroEconometricModels._spectral_window(1, :hann)
        @test w1 == [1.0]
    end

    # =========================================================================
    # Ljung-Box Test
    # =========================================================================
    @testset "Ljung-Box" begin
        rng9 = Random.MersenneTwister(666)
        wn = randn(rng9, 500)
        r = ljung_box_test(wn, 10)
        @test r isa LjungBoxResult
        @test r.pvalue > 0.01  # white noise should pass

        # AR(1): should reject
        y_ar = zeros(500)
        y_ar[1] = randn(rng9)
        for t in 2:500; y_ar[t] = 0.9 * y_ar[t-1] + randn(rng9); end
        r_ar = ljung_box_test(y_ar, 10)
        @test r_ar.pvalue < 0.05
    end

    # =========================================================================
    # Box-Pierce Test
    # =========================================================================
    @testset "Box-Pierce" begin
        rng10 = Random.MersenneTwister(777)
        wn = randn(rng10, 500)
        r = box_pierce_test(wn, 10)
        @test r isa BoxPierceResult
        @test r.pvalue > 0.01
    end

    # =========================================================================
    # Durbin-Watson Test
    # =========================================================================
    @testset "Durbin-Watson" begin
        rng11 = Random.MersenneTwister(888)
        n = 100
        X = [ones(n) randn(rng11, n)]
        β = [1.0, 2.0]
        ε = randn(rng11, n)
        y = X * β + ε
        resid = y - X * (X \ y)
        r = durbin_watson_test(resid, X)
        @test r isa DurbinWatsonResult
        @test abs(r.statistic - 2.0) < 0.5  # should be near 2 for iid errors
    end

    # =========================================================================
    # Fisher's Test
    # =========================================================================
    @testset "Fisher's test" begin
        # Plant a sinusoid
        n = 200
        t = 1:n
        y = sin.(2π * 8 / n .* t) .+ 0.1 .* randn(Random.MersenneTwister(999), n)
        r = fisher_test(y)
        @test r isa FisherTestResult
        @test r.pvalue < 0.05  # should detect the sinusoid

        # White noise
        wn = randn(Random.MersenneTwister(1000), 200)
        r_wn = fisher_test(wn)
        @test r_wn.pvalue > 0.01
    end

    # =========================================================================
    # Bartlett's White Noise Test
    # =========================================================================
    @testset "Bartlett's test" begin
        rng12 = Random.MersenneTwister(1111)
        wn = randn(rng12, 300)
        r = bartlett_white_noise_test(wn)
        @test r isa BartlettWhiteNoiseResult
        @test r.pvalue > 0.01  # white noise should pass

        # AR(1)
        y = zeros(300); y[1] = randn(rng12)
        for t in 2:300; y[t] = 0.8 * y[t-1] + randn(rng12); end
        r_ar = bartlett_white_noise_test(y)
        @test r_ar.pvalue < 0.05  # should reject white noise
    end

    # =========================================================================
    # Band Power
    # =========================================================================
    @testset "band_power" begin
        rng13 = Random.MersenneTwister(1234)
        y = randn(rng13, 256)
        sp = spectral_density(y; method=:welch)
        bp = band_power(sp, (2π/32, 2π/6))
        @test 0.0 <= bp <= 1.0
    end

    # =========================================================================
    # Ideal Bandpass
    # =========================================================================
    @testset "ideal_bandpass" begin
        n = 256
        t = collect(1:n)
        # Low-freq + high-freq
        y_low = sin.(2π * 3 / n .* t)   # 3 cycles (in band)
        y_high = sin.(2π * 50 / n .* t)  # 50 cycles (out of band)
        y = y_low .+ y_high

        # Band-pass: keep 1-10 cycles
        lo = 2π * 1 / n
        hi = 2π * 10 / n
        filtered = ideal_bandpass(y, (lo, hi))
        # Should preserve low-freq, kill high-freq
        @test cor(filtered, y_low) > 0.9
        @test abs(cor(filtered, y_high)) < 0.3
    end

    # =========================================================================
    # Transfer Function
    # =========================================================================
    @testset "transfer_function" begin
        # HP filter
        tf_hp = transfer_function(:hp; lambda=1600)
        @test tf_hp isa TransferFunctionResult
        @test tf_hp.filter == :hp
        @test tf_hp.gain[1] ≈ 0.0 atol=1e-10  # DC removed

        # BK filter
        tf_bk = transfer_function(:bk; pl=6, pu=32)
        @test tf_bk.filter == :bk
        @test length(tf_bk.gain) == 512

        # Hamilton filter
        tf_h = transfer_function(:hamilton; h=8, p=4)
        @test tf_h.filter == :hamilton
    end

    # =========================================================================
    # Display (show) methods
    # =========================================================================
    @testset "Display" begin
        rng14 = Random.MersenneTwister(2222)
        io = IOBuffer()

        r = acf_pacf(randn(rng14, 200), 10)
        show(io, r)
        @test length(take!(io)) > 0

        sp = spectral_density(randn(rng14, 128); method=:welch)
        show(io, sp)
        @test length(take!(io)) > 0

        csp = cross_spectrum(randn(rng14, 128), randn(rng14, 128); segments=4)
        show(io, csp)
        @test length(take!(io)) > 0

        tf = transfer_function(:hp; lambda=1600)
        show(io, tf)
        @test length(take!(io)) > 0

        r_lb = ljung_box_test(randn(rng14, 200), 10)
        show(io, r_lb)
        @test length(take!(io)) > 0

        r_bp = box_pierce_test(randn(rng14, 200), 10)
        show(io, r_bp)
        @test length(take!(io)) > 0

        r_ft = fisher_test(randn(rng14, 100))
        show(io, r_ft)
        @test length(take!(io)) > 0

        r_bt = bartlett_white_noise_test(randn(rng14, 100))
        show(io, r_bt)
        @test length(take!(io)) > 0
    end

    # =========================================================================
    # Plotting
    # =========================================================================
    @testset "Plotting" begin
        rng15 = Random.MersenneTwister(3333)

        r = acf_pacf(randn(rng15, 200), 10)
        p = plot_result(r)
        @test p isa PlotOutput
        @test !isempty(p.html)

        sp = spectral_density(randn(rng15, 128); method=:welch)
        p2 = plot_result(sp)
        @test p2 isa PlotOutput

        csp = cross_spectrum(randn(rng15, 128), randn(rng15, 128); segments=4)
        p3 = plot_result(csp)
        @test p3 isa PlotOutput

        tf = transfer_function(:hp; lambda=1600)
        p4 = plot_result(tf)
        @test p4 isa PlotOutput
    end

    # =========================================================================
    # TimeSeriesData dispatch
    # =========================================================================
    @testset "TimeSeriesData dispatch" begin
        rng16 = Random.MersenneTwister(4444)
        data = randn(rng16, 200, 1)
        ts = TimeSeriesData(data; varnames=["GDP"], frequency=Quarterly)
        r = acf(ts, 10)
        @test r isa ACFResult

        r2 = acf(ts, 10; var="GDP")
        @test r.acf ≈ r2.acf

        sp = spectral_density(ts; method=:welch)
        @test sp isa SpectralDensityResult

        sp2 = periodogram(ts)
        @test sp2 isa SpectralDensityResult
    end

    # =========================================================================
    # Float32 / type stability
    # =========================================================================
    @testset "Float32 fallback" begin
        y32 = Float32.(randn(Random.MersenneTwister(5555), 100))
        r = acf(y32, 10)
        @test r isa ACFResult{Float64}  # should upcast

        sp = periodogram(y32)
        @test sp isa SpectralDensityResult{Float64}
    end

    # =========================================================================
    # FFTW fix: GDFM works without explicit `using FFTW`
    # =========================================================================
    @testset "FFTW direct dep" begin
        rng17 = Random.MersenneTwister(6666)
        X = randn(rng17, 100, 5)
        # Should work without `using FFTW`
        gdfm = estimate_gdfm(X, 2)
        @test gdfm isa GeneralizedDynamicFactorModel
    end
end
```

**Step 3: Add test to runtests.jl**

Add to Group 6 ("Volatility & Filters") in the `TEST_GROUPS` array:
```julia
"spectral/test_spectral.jl",
```

And add the same to the sequential fallback section (inside Group 6's testsets):
```julia
@testset "Spectral Analysis" begin include("spectral/test_spectral.jl") end
```

**Step 4: Run the spectral tests**

```bash
cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/polymorphic-tickling-raven
julia --project=. -e '
    using Test, MacroEconometricModels
    const FAST = false
    include("test/fixtures.jl")
    @testset "Spectral" begin
        include("test/spectral/test_spectral.jl")
    end
'
```

Expected: All tests pass.

**Step 5: Fix any failures**

Debug and fix — this is the main iteration step. Common issues:
- Missing `using` statements in spectral files
- Type mismatches (e.g., `Float64` vs `T`)
- D3.js rendering helpers not found (check `_render_line_js`, `_make_plot` signatures)

**Step 6: Run the full test suite**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl
```

Expected: All ~12400+ tests pass (existing tests unbroken, plus new spectral tests).

**Step 7: Commit**

```bash
git add test/spectral/test_spectral.jl test/runtests.jl
git commit -m "test: add comprehensive spectral analysis and ACF/PACF tests"
```

---

## Task 12: Final Verification & Version Bump

**Files:**
- Modify: `Project.toml` (version bump)

**Step 1: Run full test suite one more time**

```bash
MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl
```

**Step 2: Verify all new exports work**

```bash
julia --project=. -e '
    using MacroEconometricModels
    # ACF/PACF
    @assert acf isa Function
    @assert pacf isa Function
    @assert acf_pacf isa Function
    @assert ccf isa Function
    # Spectral
    @assert periodogram isa Function
    @assert spectral_density isa Function
    @assert cross_spectrum isa Function
    @assert coherence isa Function
    @assert phase isa Function
    @assert gain isa Function
    # Tests
    @assert ljung_box_test isa Function
    @assert box_pierce_test isa Function
    @assert durbin_watson_test isa Function
    @assert fisher_test isa Function
    @assert bartlett_white_noise_test isa Function
    # Filtering
    @assert ideal_bandpass isa Function
    @assert transfer_function isa Function
    @assert band_power isa Function
    # Types
    @assert ACFResult <: AbstractAnalysisResult
    @assert SpectralDensityResult <: AbstractAnalysisResult
    @assert CrossSpectrumResult <: AbstractAnalysisResult
    @assert TransferFunctionResult <: AbstractAnalysisResult
    @assert LjungBoxResult <: StatsAPI.HypothesisTest
    println("All exports verified")
'
```

**Step 3: Bump version**

In `Project.toml`, change version from `"0.3.5"` to `"0.3.6"`.

**Step 4: Commit**

```bash
git add Project.toml
git commit -m "chore: bump version to v0.3.6"
```
