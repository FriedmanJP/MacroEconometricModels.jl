# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Frequency-domain filtering and transfer function analysis.

Implements:
- `ideal_bandpass(y, f_low, f_high)` — ideal bandpass filter via FFT
- `transfer_function(filter; n_freq)` — frequency response of HP, BK, Hamilton filters

References:
- Baxter, M. & King, R. G. (1999). "Measuring Business Cycles."
- Hodrick, R. J. & Prescott, E. C. (1997). "Postwar U.S. Business Cycles."
"""

# =============================================================================
# Ideal Bandpass Filter
# =============================================================================

"""
    ideal_bandpass(y::AbstractVector, f_low::Real, f_high::Real) -> Vector

Apply an ideal (brick-wall) bandpass filter in the frequency domain.

Retains frequency components in [f_low, f_high] (radians, ∈ [0, π]) and
zeroes everything else. Implements via FFT → zero outside band → IFFT.

# Arguments
- `y` — time series vector
- `f_low` — lower cutoff frequency (radians)
- `f_high` — upper cutoff frequency (radians)

# Returns
Filtered series (real-valued vector of same length as `y`).

# Example
```julia
# Extract business-cycle frequencies (6-32 quarters)
y_bc = ideal_bandpass(gdp, 2π/32, 2π/6)
```
"""
function ideal_bandpass(y::AbstractVector{T}, f_low::Real, f_high::Real) where {T<:AbstractFloat}
    n = length(y)
    n < 4 && throw(ArgumentError("Need at least 4 observations"))
    fl, fh = T(f_low), T(f_high)
    (fl < 0 || fh > T(π) || fl >= fh) &&
        throw(ArgumentError("Require 0 ≤ f_low < f_high ≤ π"))

    yd = y .- mean(y)
    Y = FFTW.fft(yd)

    # Zero out frequencies outside the band
    for k in 1:n
        # Map FFT index to frequency in [0, 2π)
        freq_k = T(2π) * (k - 1) / n
        # Normalize to [0, π] (use symmetry)
        f = freq_k <= T(π) ? freq_k : T(2π) - freq_k
        if f < fl || f > fh
            Y[k] = zero(Complex{T})
        end
    end

    real(FFTW.ifft(Y))
end

# Non-Float64 fallback
ideal_bandpass(y::AbstractVector{<:Real}, f_low::Real, f_high::Real) =
    ideal_bandpass(Float64.(y), f_low, f_high)

# =============================================================================
# Transfer Function Analysis
# =============================================================================

"""
    transfer_function(filter::Symbol; lambda::Real=1600, K::Int=12,
                      h::Int=8, n_freq::Int=256) -> TransferFunctionResult

Compute the frequency response (gain and phase) of a time-series filter.

# Arguments
- `filter` — filter type: `:hp`, `:bk`, or `:hamilton`
- `lambda` — HP filter smoothing parameter (default: 1600 for quarterly)
- `K` — Baxter-King half-window length (default: 12)
- `h` — Hamilton filter horizon (default: 8)
- `n_freq` — number of frequency grid points (default: 256)

# Returns
`TransferFunctionResult` with gain and phase at each frequency.

# Example
```julia
tf = transfer_function(:hp, lambda=1600)
tf.gain   # HP filter gain at each frequency
```
"""
function transfer_function(filter::Symbol; lambda::Real=1600, K::Int=12,
                           h::Int=8, n_freq::Int=256)
    filter in (:hp, :bk, :hamilton) ||
        throw(ArgumentError("filter must be :hp, :bk, or :hamilton"))

    T = Float64
    freq = T[π * (k - 1) / (n_freq - 1) for k in 1:n_freq]
    gain_vals = zeros(T, n_freq)
    phase_vals = zeros(T, n_freq)

    if filter == :hp
        # HP filter gain: G(ω) = 4λ sin²(ω/2) / (1 + 4λ sin²(ω/2))
        # This is the gain for the CYCLE (high-pass component)
        for k in 1:n_freq
            s2 = sin(freq[k] / 2)^2
            gain_vals[k] = 4 * lambda * s2 / (1 + 4 * lambda * s2)
        end
        # HP filter has zero phase (symmetric)

    elseif filter == :bk
        # Baxter-King symmetric MA filter gain
        # Ideal bandpass weights truncated to [-K, K]
        # Default: 6-32 quarter business cycle frequencies
        f_low = 2π / 32
        f_high = 2π / 6

        # Compute BK filter weights
        bk_weights = zeros(T, 2 * K + 1)
        a0 = (f_high - f_low) / π
        for j in 1:K
            bk_weights[K + 1 + j] = (sin(f_high * j) - sin(f_low * j)) / (π * j)
            bk_weights[K + 1 - j] = bk_weights[K + 1 + j]
        end
        bk_weights[K + 1] = a0

        # Adjust weights to sum to zero (remove level)
        bk_weights .-= mean(bk_weights)

        for k in 1:n_freq
            z = zero(Complex{T})
            for j in -K:K
                z += bk_weights[j + K + 1] * exp(-im * freq[k] * j)
            end
            gain_vals[k] = abs(z)
            phase_vals[k] = angle(z)
        end

    else  # :hamilton
        # Hamilton filter: y_t - ŷ_{t|t-h}
        # Transfer function: 1 - e^{-iωh} (simplified for random walk benchmark)
        for k in 1:n_freq
            z = 1.0 - exp(-im * freq[k] * h)
            gain_vals[k] = abs(z)
            phase_vals[k] = angle(z)
        end
    end

    TransferFunctionResult{T}(freq, gain_vals, phase_vals, filter)
end
