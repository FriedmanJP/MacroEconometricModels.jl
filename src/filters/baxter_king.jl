# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Baxter-King (1999) symmetric band-pass filter.

Isolates cyclical fluctuations in a specified frequency band [ωL, ωH]
using a finite symmetric moving average approximation to the ideal band-pass filter.

Reference: Baxter, Marianne, and Robert G. King. 1999.
"Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series."
*Review of Economics and Statistics* 81 (4): 575–593.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    baxter_king(y::AbstractVector; pl=6, pu=32, K=12) -> BaxterKingResult

Apply the Baxter-King band-pass filter to isolate business cycle frequencies.

Computes symmetric moving average weights that approximate the ideal band-pass
filter for periods between `pl` and `pu`. The filter is constrained to sum to
zero (removing stochastic trends).

Loses `K` observations at each end of the sample (2K total).

# Arguments
- `y::AbstractVector`: Time series data (length > 2K)

# Keywords
- `pl::Int=6`: Minimum period of oscillation to pass (quarterly: 6 = 1.5 years)
- `pu::Int=32`: Maximum period of oscillation to pass (quarterly: 32 = 8 years)
- `K::Int=12`: Truncation length (number of leads/lags in the moving average)

# Returns
- `BaxterKingResult{T}` with fields `cycle`, `trend`, `weights`, `pl`, `pu`,
  `K`, `T_obs`, `valid_range`

# Examples
```julia
y = cumsum(randn(200))
result = baxter_king(y)                    # quarterly defaults (6-32 quarters)
result = baxter_king(y; pl=2, pu=8, K=6)  # annual data (2-8 years)
```

# References
- Baxter, M., & King, R. G. (1999). *REStat* 81(4): 575–593.
"""
function baxter_king(y::AbstractVector{T}; pl::Int=6, pu::Int=32,
                     K::Int=12) where {T<:AbstractFloat}
    T_obs = length(y)
    pl < 2 && throw(ArgumentError("Lower period pl must be at least 2, got $pl"))
    pu <= pl && throw(ArgumentError("Upper period pu must exceed pl. Got pl=$pl, pu=$pu"))
    K < 1 && throw(ArgumentError("Truncation K must be positive, got $K"))
    T_obs <= 2 * K && throw(ArgumentError(
        "Not enough observations ($T_obs) for K=$K. Need > $(2K)."))

    yv = Vector{T}(y)

    # Frequency cutoffs
    omega_H = T(2) * T(pi) / T(pl)   # high frequency cutoff
    omega_L = T(2) * T(pi) / T(pu)   # low frequency cutoff

    # Ideal band-pass filter weights (before truncation adjustment)
    # B_0 = (ω_H - ω_L) / π
    # B_j = (sin(ω_H j) - sin(ω_L j)) / (π j)  for j ≥ 1
    B = Vector{T}(undef, K + 1)
    B[1] = (omega_H - omega_L) / T(pi)
    @inbounds for j in 1:K
        B[j + 1] = (sin(omega_H * j) - sin(omega_L * j)) / (T(pi) * j)
    end

    # Constrain weights to sum to zero (removes stochastic trends)
    # θ = -(B_0 + 2 Σ_{j=1}^K B_j) / (2K + 1)
    # a_j = B_j + θ
    weight_sum = B[1] + T(2) * sum(@view B[2:end])
    theta = -weight_sum / T(2 * K + 1)

    weights = B .+ theta

    # Apply symmetric filter: c_t = a_0 y_t + Σ_{j=1}^K a_j (y_{t-j} + y_{t+j})
    n_eff = T_obs - 2 * K
    cyc = Vector{T}(undef, n_eff)

    @inbounds for i in 1:n_eff
        t = K + i  # index in original series
        ct = weights[1] * yv[t]
        for j in 1:K
            ct += weights[j + 1] * (yv[t - j] + yv[t + j])
        end
        cyc[i] = ct
    end

    # Trend = y - cycle (for the valid range)
    trd = yv[(K + 1):(T_obs - K)] .- cyc

    valid_range = (K + 1):(T_obs - K)

    BaxterKingResult(cyc, trd, weights, pl, pu, K, T_obs, valid_range)
end

# Float64 fallback for non-float input
baxter_king(y::AbstractVector; kwargs...) = baxter_king(Float64.(y); kwargs...)
