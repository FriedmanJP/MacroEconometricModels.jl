# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Spectral diagnostic tests:
- Fisher's test for periodicity (Fisher 1929)
- Bartlett's white-noise test (Bartlett 1955)
- Band power computation

References:
- Fisher, R. A. (1929). "Tests of significance in harmonic analysis."
- Bartlett, M. S. (1955). An Introduction to Stochastic Processes.
"""

# =============================================================================
# Result Types
# =============================================================================

"""
    FisherTestResult{T} <: StatsAPI.HypothesisTest

Result from Fisher's test for periodicity (hidden periodicities in data).

# Fields
- `statistic::T` — Fisher's g statistic (max periodogram / sum)
- `pvalue::T` — p-value
- `peak_freq::T` — frequency with maximum periodogram value
- `nobs::Int` — number of observations
"""
struct FisherTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    peak_freq::T
    nobs::Int
end

"""
    BartlettWhiteNoiseResult{T} <: StatsAPI.HypothesisTest

Result from Bartlett's cumulative periodogram test for white noise.

# Fields
- `statistic::T` — Kolmogorov-Smirnov statistic (max deviation from uniform)
- `pvalue::T` — approximate p-value
- `nobs::Int` — number of observations
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

# =============================================================================
# Fisher's Test for Periodicity
# =============================================================================

"""
    fisher_test(y::AbstractVector) -> FisherTestResult

Fisher's exact test for hidden periodicities.

Tests H₀: y is white noise (no dominant frequency) against H₁: there exists
a periodic component. The test statistic is g = max(I(ωⱼ)) / Σ I(ωⱼ).

# Arguments
- `y` — time series vector

# Returns
`FisherTestResult` with test statistic and p-value.
"""
function fisher_test(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    yd = y .- mean(y)
    Y = FFTW.fft(yd)

    # Use Fourier frequencies (exclude DC and Nyquist for odd n)
    m = div(n - 1, 2)
    m < 1 && throw(ArgumentError("Series too short for Fisher's test"))

    I = zeros(T, m)
    for j in 1:m
        I[j] = abs2(Y[j + 1]) / n  # periodogram at freq 2πj/n
    end

    total = sum(I)
    total < T(1e-30) && return FisherTestResult{T}(zero(T), one(T), zero(T), n)

    g = maximum(I) / total
    peak_idx = argmax(I)
    peak_freq = T(2π * peak_idx / n)

    # Fisher's exact p-value: P(g > x) ≈ Σ_{k=1}^{⌊1/x⌋} (-1)^{k+1} C(m,k) (1-kx)^{m-1}
    pval = zero(T)
    kmax = min(m, floor(Int, 1 / g))
    for k in 1:kmax
        term = T((-1)^(k + 1)) * binomial(BigInt(m), BigInt(k)) * (1 - k * g)^(m - 1)
        pval += T(term)
    end
    pval = clamp(pval, zero(T), one(T))

    FisherTestResult{T}(g, pval, peak_freq, n)
end

# Non-Float64 fallback
fisher_test(y::AbstractVector{<:Real}) = fisher_test(Float64.(y))

# =============================================================================
# Bartlett's White Noise Test
# =============================================================================

"""
    bartlett_white_noise_test(y::AbstractVector) -> BartlettWhiteNoiseResult

Bartlett's cumulative periodogram test for white noise.

Under H₀ (white noise), the cumulative normalized periodogram should follow
a uniform distribution. The KS statistic measures the maximum deviation.

# Arguments
- `y` — time series vector

# Returns
`BartlettWhiteNoiseResult` with KS statistic and p-value.
"""
function bartlett_white_noise_test(y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    yd = y .- mean(y)
    Y = FFTW.fft(yd)

    m = div(n - 1, 2)
    m < 1 && throw(ArgumentError("Series too short for Bartlett's test"))

    I = zeros(T, m)
    for j in 1:m
        I[j] = abs2(Y[j + 1]) / n
    end

    total = sum(I)
    total < T(1e-30) && return BartlettWhiteNoiseResult{T}(zero(T), one(T), n)

    # Cumulative periodogram
    cum_I = cumsum(I) ./ total
    # Under H₀, cum_I[j] should ≈ j/m
    expected = T[(j / m) for j in 1:m]

    ks = maximum(abs.(cum_I .- expected))

    # Asymptotic KS p-value approximation
    # P(D > d) ≈ 2 Σ_{k=1}^∞ (-1)^{k+1} exp(-2k²m²d²)  (Kolmogorov-Smirnov)
    sqrt_m = sqrt(T(m))
    pval = zero(T)
    for k in 1:100
        term = T(2) * T((-1)^(k + 1)) * exp(-T(2) * k^2 * sqrt_m^2 * ks^2)
        pval += term
        abs(term) < T(1e-12) && break
    end
    pval = clamp(pval, zero(T), one(T))

    BartlettWhiteNoiseResult{T}(ks, pval, n)
end

# Non-Float64 fallback
bartlett_white_noise_test(y::AbstractVector{<:Real}) = bartlett_white_noise_test(Float64.(y))

# =============================================================================
# Band Power
# =============================================================================

"""
    band_power(result::SpectralDensityResult, f_low::Real, f_high::Real) -> Real

Compute the power (integrated spectral density) in frequency band [f_low, f_high].

Frequencies are in radians ∈ [0, π].

# Arguments
- `result` — a `SpectralDensityResult` from `periodogram` or `spectral_density`
- `f_low` — lower frequency bound (radians)
- `f_high` — upper frequency bound (radians)

# Returns
Integrated spectral density in the specified band.
"""
function band_power(result::SpectralDensityResult{T}, f_low::Real, f_high::Real) where {T<:AbstractFloat}
    fl, fh = T(f_low), T(f_high)
    fl >= fh && throw(ArgumentError("f_low must be < f_high"))

    freq = result.freq
    density = result.density

    # Trapezoidal integration over [f_low, f_high]
    power = zero(T)
    for k in 2:length(freq)
        f1, f2 = freq[k-1], freq[k]
        # Check overlap with [fl, fh]
        f1 >= fh && break
        f2 <= fl && continue

        # Clip to band
        a = max(f1, fl)
        b = min(f2, fh)
        # Linear interpolation for density at a and b
        frac_a = (a - f1) / (f2 - f1)
        frac_b = (b - f1) / (f2 - f1)
        d_a = density[k-1] + frac_a * (density[k] - density[k-1])
        d_b = density[k-1] + frac_b * (density[k] - density[k-1])
        power += (d_a + d_b) / 2 * (b - a)
    end
    power
end
