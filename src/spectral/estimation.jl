# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Spectral density estimation: raw periodogram, Welch's method, kernel-smoothed
periodogram, and AR-based parametric estimation.

References:
- Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra."
- Brillinger, D. R. (1981). Time Series: Data Analysis and Theory.
- Priestley, M. B. (1981). Spectral Analysis and Time Series.
"""

# =============================================================================
# Raw Periodogram
# =============================================================================

"""
    periodogram(y::AbstractVector; window::Symbol=:rectangular, conf_level::Real=0.95) -> SpectralDensityResult

Compute the raw periodogram of `y`.

# Arguments
- `y` — time series vector
- `window` — data window/taper (default: `:rectangular`; see `_spectral_window`)
- `conf_level` — confidence level for chi-squared CI (default: 0.95)

# Returns
`SpectralDensityResult` with frequency grid, density, and confidence bounds.
"""
function periodogram(y::AbstractVector{T}; window::Symbol=:rectangular,
                     conf_level::Real=0.95) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations, got $n"))

    # Demean and apply window
    yd = y .- mean(y)
    w = T.(_spectral_window(n, window))
    yw = yd .* w
    U = sum(w .^ 2) / n  # window energy normalization

    # FFT
    Y = FFTW.fft(yw)
    n_freq = div(n, 2) + 1
    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]

    # Periodogram: |Y(f)|^2 / (2π n U)
    density = zeros(T, n_freq)
    scale = T(2π) * n * U
    for k in 1:n_freq
        density[k] = abs2(Y[k]) / scale
    end

    # Chi-squared CI (periodogram has chi2(2) / 2 distribution)
    alpha = T(1 - conf_level)
    nu = T(2.0)  # degrees of freedom per frequency
    ci_lower = density .* (nu / T(quantile(Chisq(nu), 1 - alpha / 2)))
    ci_upper = density .* (nu / T(quantile(Chisq(nu), alpha / 2)))

    bw = T(2π / n)  # raw bandwidth

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :periodogram, bw, n)
end

# Non-Float64 fallback
periodogram(y::AbstractVector{<:Real}; kwargs...) = periodogram(Float64.(y); kwargs...)

# =============================================================================
# Spectral Density (multi-method dispatcher)
# =============================================================================

"""
    spectral_density(y::AbstractVector; method::Symbol=:welch, kwargs...) -> SpectralDensityResult

Estimate the spectral density of `y`.

# Methods
- `:welch` — Welch's averaged modified periodogram (default)
- `:smoothed` — kernel-smoothed periodogram (Daniell kernel)
- `:ar` — AR parametric spectrum via Burg's method

# Common Keyword Arguments
- `window::Symbol=:hann` — data window
- `conf_level::Real=0.95` — confidence level

# Method-Specific Keywords
## `:welch`
- `segment_length::Int=0` — segment length (default: n ÷ 4, minimum 16)
- `overlap::Real=0.5` — fraction overlap ∈ [0, 1)

## `:smoothed`
- `bandwidth::Int=0` — kernel half-width (default: ⌊√n⌋)

## `:ar`
- `order::Int=0` — AR order (default: AIC-selected)
- `n_freq::Int=256` — number of frequency grid points
"""
function spectral_density(y::AbstractVector{T}; method::Symbol=:welch,
                          kwargs...) where {T<:AbstractFloat}
    method in (:welch, :smoothed, :ar) ||
        throw(ArgumentError("method must be :welch, :smoothed, or :ar"))

    if method == :welch
        return _spectral_welch(y; kwargs...)
    elseif method == :smoothed
        return _spectral_smoothed(y; kwargs...)
    else
        return _spectral_ar(y; kwargs...)
    end
end

# Non-Float64 fallback
spectral_density(y::AbstractVector{<:Real}; kwargs...) = spectral_density(Float64.(y); kwargs...)

# =============================================================================
# Welch's Method
# =============================================================================

function _spectral_welch(y::AbstractVector{T};
                         window::Symbol=:hann, conf_level::Real=0.95,
                         segment_length::Int=0, overlap::Real=0.5,
                         kwargs...) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    seg_len = segment_length > 0 ? segment_length : max(16, n ÷ 4)
    seg_len = min(seg_len, n)
    step = max(1, round(Int, seg_len * (1 - overlap)))

    # Window
    w = T.(_spectral_window(seg_len, window))
    U = sum(w .^ 2) / seg_len

    n_freq = div(seg_len, 2) + 1
    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]
    density_sum = zeros(T, n_freq)
    n_segments = 0

    # Demean entire series
    yd = y .- mean(y)

    # Process segments
    start = 1
    while start + seg_len - 1 <= n
        seg = yd[start:(start + seg_len - 1)] .* w
        Y = FFTW.fft(seg)
        scale = T(2π) * seg_len * U
        for k in 1:n_freq
            density_sum[k] += abs2(Y[k]) / scale
        end
        n_segments += 1
        start += step
    end

    n_segments == 0 && throw(ArgumentError("No complete segments; reduce segment_length"))
    density = density_sum ./ n_segments

    # CI with equivalent degrees of freedom
    nu = T(2 * n_segments)
    alpha = T(1 - conf_level)
    ci_lower = density .* (nu / T(quantile(Chisq(nu), 1 - alpha / 2)))
    ci_upper = density .* (nu / T(quantile(Chisq(nu), alpha / 2)))

    bw = T(2π / seg_len) * n_segments  # effective bandwidth

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :welch, bw, n)
end

# =============================================================================
# Kernel-Smoothed Periodogram
# =============================================================================

function _spectral_smoothed(y::AbstractVector{T};
                            window::Symbol=:rectangular, conf_level::Real=0.95,
                            bandwidth::Int=0, kwargs...) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    # Raw periodogram
    yd = y .- mean(y)
    w = T.(_spectral_window(n, window))
    yw = yd .* w
    U = sum(w .^ 2) / n

    Y = FFTW.fft(yw)
    n_freq = div(n, 2) + 1
    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]

    raw = zeros(T, n_freq)
    scale = T(2π) * n * U
    for k in 1:n_freq
        raw[k] = abs2(Y[k]) / scale
    end

    # Daniell kernel smoothing
    m = bandwidth > 0 ? bandwidth : max(1, round(Int, sqrt(n)))
    m = min(m, n_freq - 1)

    density = zeros(T, n_freq)
    for k in 1:n_freq
        s = zero(T)
        cnt = 0
        for j in max(1, k - m):min(n_freq, k + m)
            s += raw[j]
            cnt += 1
        end
        density[k] = s / cnt
    end

    # Equivalent DOF for Daniell kernel
    nu = T(2 * (2 * m + 1))
    alpha = T(1 - conf_level)
    ci_lower = density .* (nu / T(quantile(Chisq(nu), 1 - alpha / 2)))
    ci_upper = density .* (nu / T(quantile(Chisq(nu), alpha / 2)))

    bw = T(2π * (2 * m + 1) / n)

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :smoothed, bw, n)
end

# =============================================================================
# AR Parametric Spectrum (Burg)
# =============================================================================

"""Select AR order by AIC from Burg estimates."""
function _select_ar_order_aic(y::AbstractVector{T}; max_order::Int=0) where {T<:AbstractFloat}
    n = length(y)
    pmax = max_order > 0 ? max_order : min(round(Int, 10 * log10(n)), n ÷ 3)
    pmax = max(1, pmax)

    best_aic = T(Inf)
    best_p = 1

    yd = y .- mean(y)

    for p in 1:pmax
        a, sigma2 = _burg_coefficients(yd, p)
        sigma2 <= zero(T) && continue
        aic_val = n * log(sigma2) + 2 * p
        if aic_val < best_aic
            best_aic = aic_val
            best_p = p
        end
    end
    best_p
end

"""Burg's method for AR coefficient estimation."""
function _burg_coefficients(y::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    n = length(y)
    p >= n && throw(ArgumentError("AR order p=$p must be < n=$n"))

    # Initialize forward/backward prediction errors
    ef = copy(y)
    eb = copy(y)
    a = zeros(T, p)
    sigma2 = sum(y .^ 2) / n

    for k in 1:p
        # Reflection coefficient
        num = zero(T)
        den = zero(T)
        for t in (k+1):n
            num += ef[t] * eb[t-1]
            den += ef[t]^2 + eb[t-1]^2
        end
        den < T(1e-30) && break
        rc = -T(2) * num / den

        # Update AR coefficients
        a_new = zeros(T, p)
        a_new[k] = rc
        for j in 1:(k-1)
            a_new[j] = a[j] + rc * a[k-j]
        end
        a .= a_new

        # Update prediction errors
        sigma2 *= (1 - rc^2)

        ef_old = copy(ef)
        for t in (k+1):n
            ef[t] = ef_old[t] + rc * eb[t-1]
            eb[t] = eb[t-1] + rc * ef_old[t]
        end
    end

    a, sigma2
end

function _spectral_ar(y::AbstractVector{T};
                      conf_level::Real=0.95, order::Int=0,
                      n_freq::Int=256, kwargs...) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    yd = y .- mean(y)
    p = order > 0 ? order : _select_ar_order_aic(yd)
    a, sigma2 = _burg_coefficients(yd, p)

    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]
    density = zeros(T, n_freq)

    for k in 1:n_freq
        z = exp(-im * freq[k])
        denom = one(Complex{T})
        zp = one(Complex{T})
        for j in 1:p
            zp *= z
            denom += a[j] * zp
        end
        density[k] = sigma2 / (T(2π) * abs2(denom))
    end

    # AR spectrum CI (chi2 with 2 DOF equivalent, following Priestley 1981)
    nu = T(2 * n / p)  # equivalent DOF
    alpha = T(1 - conf_level)
    ci_lower = density .* (nu / T(quantile(Chisq(nu), 1 - alpha / 2)))
    ci_upper = density .* (nu / T(quantile(Chisq(nu), alpha / 2)))

    bw = T(2π * p / n)

    SpectralDensityResult{T}(freq, density, ci_lower, ci_upper, :ar, bw, n)
end
