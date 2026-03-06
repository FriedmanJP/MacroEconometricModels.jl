# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Spectral window (tapering) functions for spectral density estimation.

Supported windows: rectangular, Bartlett, Hann, Hamming, Blackman, Tukey, flat-top.
"""

# =============================================================================
# Spectral Window Function
# =============================================================================

"""
    _spectral_window(n::Int, wtype::Symbol) -> Vector{Float64}

Compute a spectral window (taper) of length `n`.

# Supported window types
- `:rectangular` — uniform weights (no tapering)
- `:bartlett` — triangular (Bartlett) window
- `:hann` — Hann (raised cosine) window
- `:hanning` — alias for `:hann`
- `:hamming` — Hamming window (α = 0.54)
- `:blackman` — Blackman window
- `:tukey` — Tukey (tapered cosine) window with α = 0.5
- `:flat_top` — flat-top window (maximal amplitude accuracy)

# Arguments
- `n::Int` — window length (must be ≥ 1)
- `wtype::Symbol` — window type

# Returns
`Vector{Float64}` of length `n`.
"""
function _spectral_window(n::Int, wtype::Symbol)
    n < 1 && throw(ArgumentError("Window length n must be ≥ 1"))
    n == 1 && return [1.0]

    w = zeros(Float64, n)
    N = n - 1  # for 0-indexed formulas

    if wtype == :rectangular
        fill!(w, 1.0)

    elseif wtype == :bartlett
        for i in 0:N
            w[i+1] = 1.0 - abs(2.0 * i / N - 1.0)
        end

    elseif wtype == :hann || wtype == :hanning
        for i in 0:N
            w[i+1] = 0.5 * (1.0 - cos(2π * i / N))
        end

    elseif wtype == :hamming
        for i in 0:N
            w[i+1] = 0.54 - 0.46 * cos(2π * i / N)
        end

    elseif wtype == :blackman
        for i in 0:N
            w[i+1] = 0.42 - 0.5 * cos(2π * i / N) + 0.08 * cos(4π * i / N)
        end

    elseif wtype == :tukey
        α = 0.5
        for i in 0:N
            t = i / N
            if t < α / 2
                w[i+1] = 0.5 * (1.0 + cos(2π / α * (t - α / 2)))
            elseif t > 1.0 - α / 2
                w[i+1] = 0.5 * (1.0 + cos(2π / α * (t - 1.0 + α / 2)))
            else
                w[i+1] = 1.0
            end
        end

    elseif wtype == :flat_top
        a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
        for i in 0:N
            w[i+1] = a0 - a1 * cos(2π * i / N) + a2 * cos(4π * i / N) -
                     a3 * cos(6π * i / N) + a4 * cos(8π * i / N)
        end

    else
        throw(ArgumentError("Unknown window type: :$wtype. " *
            "Supported: :rectangular, :bartlett, :hann, :hanning, :hamming, :blackman, :tukey, :flat_top"))
    end

    return w
end
