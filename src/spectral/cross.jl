# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Cross-spectral analysis: cross-spectrum, coherence, phase, and gain
between two time series.

References:
- Brillinger, D. R. (1981). Time Series: Data Analysis and Theory.
- Priestley, M. B. (1981). Spectral Analysis and Time Series.
- Hamilton, J. D. (1994). Time Series Analysis, Chapter 10.
"""

# =============================================================================
# Cross-Spectrum
# =============================================================================

"""
    cross_spectrum(x::AbstractVector, y::AbstractVector;
                   window::Symbol=:hann, segment_length::Int=0,
                   overlap::Real=0.5) -> CrossSpectrumResult

Estimate the cross-spectral density between `x` and `y` using Welch's method.

# Arguments
- `x`, `y` — time series vectors (same length)
- `window` — data window (default: `:hann`)
- `segment_length` — segment length (default: n ÷ 4)
- `overlap` — fraction overlap (default: 0.5)

# Returns
`CrossSpectrumResult` with co-spectrum, quadrature spectrum, coherence, phase, and gain.
"""
function cross_spectrum(x::AbstractVector{T}, y::AbstractVector{T};
                        window::Symbol=:hann, segment_length::Int=0,
                        overlap::Real=0.5) where {T<:AbstractFloat}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    n < 4 && throw(ArgumentError("Need at least 4 observations"))

    seg_len = segment_length > 0 ? segment_length : max(16, n ÷ 4)
    seg_len = min(seg_len, n)
    step = max(1, round(Int, seg_len * (1 - overlap)))

    w = T.(_spectral_window(seg_len, window))
    U = sum(w .^ 2) / seg_len

    n_freq = div(seg_len, 2) + 1
    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]

    # Accumulators for auto- and cross-spectra
    Sxx = zeros(T, n_freq)
    Syy = zeros(T, n_freq)
    Sxy_re = zeros(T, n_freq)
    Sxy_im = zeros(T, n_freq)
    n_segments = 0

    xd = x .- mean(x)
    yd = y .- mean(y)

    start = 1
    while start + seg_len - 1 <= n
        xseg = xd[start:(start + seg_len - 1)] .* w
        yseg = yd[start:(start + seg_len - 1)] .* w

        Xf = FFTW.fft(xseg)
        Yf = FFTW.fft(yseg)

        scale = T(2π) * seg_len * U
        for k in 1:n_freq
            Sxx[k] += abs2(Xf[k]) / scale
            Syy[k] += abs2(Yf[k]) / scale
            cxy = Xf[k] * conj(Yf[k]) / scale
            Sxy_re[k] += real(cxy)
            Sxy_im[k] += imag(cxy)
        end
        n_segments += 1
        start += step
    end

    n_segments == 0 && throw(ArgumentError("No complete segments; reduce segment_length"))

    # Average
    Sxx ./= n_segments
    Syy ./= n_segments
    Sxy_re ./= n_segments
    Sxy_im ./= n_segments

    # Co-spectrum (real part) and quadrature spectrum (negative imaginary part)
    co_spectrum = Sxy_re
    quad_spectrum = -Sxy_im

    # Squared coherence: |Sxy|^2 / (Sxx * Syy)
    coh = zeros(T, n_freq)
    phase_vals = zeros(T, n_freq)
    gain_vals = zeros(T, n_freq)

    for k in 1:n_freq
        sxy2 = Sxy_re[k]^2 + Sxy_im[k]^2
        denom = Sxx[k] * Syy[k]
        coh[k] = denom > T(1e-30) ? sxy2 / denom : zero(T)
        coh[k] = clamp(coh[k], zero(T), one(T))

        # Phase: atan2(quad, co)
        phase_vals[k] = atan(quad_spectrum[k], co_spectrum[k])

        # Gain: |Sxy| / Sxx
        gain_vals[k] = Sxx[k] > T(1e-30) ? sqrt(sxy2) / Sxx[k] : zero(T)
    end

    CrossSpectrumResult{T}(freq, co_spectrum, quad_spectrum, coh, phase_vals, gain_vals, n)
end

# Non-Float64 fallbacks
cross_spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; kwargs...) =
    cross_spectrum(Float64.(x), Float64.(y); kwargs...)
cross_spectrum(x::AbstractVector{T}, y::AbstractVector{S}; kwargs...) where {T<:AbstractFloat, S<:AbstractFloat} =
    cross_spectrum(promote(x, y)...; kwargs...)
cross_spectrum(x::AbstractVector{T}, y::AbstractVector{<:Integer}; kwargs...) where {T<:AbstractFloat} =
    cross_spectrum(x, T.(y); kwargs...)
cross_spectrum(x::AbstractVector{<:Integer}, y::AbstractVector{T}; kwargs...) where {T<:AbstractFloat} =
    cross_spectrum(T.(x), y; kwargs...)

# =============================================================================
# Convenience Extractors
# =============================================================================

"""
    coherence(x::AbstractVector, y::AbstractVector; kwargs...) -> Tuple{Vector, Vector}

Compute squared coherence between `x` and `y`. Returns `(freq, coherence)`.
"""
function coherence(x::AbstractVector, y::AbstractVector; kwargs...)
    cs = cross_spectrum(x, y; kwargs...)
    (cs.freq, cs.coherence)
end

"""
    phase(x::AbstractVector, y::AbstractVector; kwargs...) -> Tuple{Vector, Vector}

Compute phase spectrum between `x` and `y`. Returns `(freq, phase)`.
"""
function phase(x::AbstractVector, y::AbstractVector; kwargs...)
    cs = cross_spectrum(x, y; kwargs...)
    (cs.freq, cs.phase)
end

"""
    gain(x::AbstractVector, y::AbstractVector; kwargs...) -> Tuple{Vector, Vector}

Compute gain (amplitude ratio) between `x` and `y`. Returns `(freq, gain)`.
"""
function gain(x::AbstractVector, y::AbstractVector; kwargs...)
    cs = cross_spectrum(x, y; kwargs...)
    (cs.freq, cs.gain)
end
