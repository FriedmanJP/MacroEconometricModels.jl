# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
Panel transformations to remove fixed effects — first-difference, forward orthogonal
deviations (Helmert), within-group demeaning, and lag construction.
"""

# =============================================================================
# First-difference transform
# =============================================================================

"""
    _panel_first_difference(Y::Matrix{T}) -> Matrix{T}

First-difference transformation: Δy_t = y_t - y_{t-1}.
Returns (T-1) × m matrix.
"""
function _panel_first_difference(Y::Matrix{T}) where {T<:AbstractFloat}
    Y[2:end, :] .- Y[1:end-1, :]
end

# =============================================================================
# Forward orthogonal deviations (Helmert transform)
# =============================================================================

"""
    _panel_fod(Y::Matrix{T}) -> Matrix{T}

Forward orthogonal deviations (Helmert transform): removes fixed effects while
preserving orthogonality of transformed errors (Arellano & Bover, 1995).

For t = 1, ..., T-1:
    z_t = √(s/(s+1)) * (y_t - ȳ_{t+1:T})

where s = T - t. Returns (T-1) × m matrix.
"""
function _panel_fod(Y::Matrix{T}) where {T<:AbstractFloat}
    Ti, m_dim = size(Y)
    Z = Matrix{T}(undef, Ti - 1, m_dim)
    @inbounds for t in 1:(Ti-1)
        s = Ti - t
        scale = sqrt(T(s) / T(s + 1))
        for j in 1:m_dim
            fwd_mean = zero(T)
            for tt in (t+1):Ti
                fwd_mean += Y[tt, j]
            end
            fwd_mean /= s
            Z[t, j] = scale * (Y[t, j] - fwd_mean)
        end
    end
    Z
end

# =============================================================================
# Within-group demeaning
# =============================================================================

"""
    _panel_demean(Y::Matrix{T}) -> Matrix{T}

Within-group demeaning for fixed-effects OLS estimator.
Returns matrix of same size with group mean subtracted.
"""
function _panel_demean(Y::Matrix{T}) where {T<:AbstractFloat}
    Y .- mean(Y, dims=1)
end

# =============================================================================
# Panel lag construction
# =============================================================================

"""
    _panel_lag(Y::Matrix{T}, p::Int) -> Matrix{T}

Construct lagged panel matrices for one group.
Given Y (T_i × m), returns (T_i - p) × (m*p) matrix: [Y_{t-1} | Y_{t-2} | ... | Y_{t-p}].
"""
function _panel_lag(Y::Matrix{T}, p::Int) where {T<:AbstractFloat}
    Ti, m_dim = size(Y)
    Ti <= p && throw(ArgumentError("Not enough observations ($Ti) for $p lags"))
    X = Matrix{T}(undef, Ti - p, m_dim * p)
    @inbounds for l in 1:p
        cols = ((l-1)*m_dim+1):(l*m_dim)
        X[:, cols] = Y[(p+1-l):(Ti-l), :]
    end
    X
end

"""
    _panel_lag_levels(Y::Matrix{T}, p::Int) -> Tuple{Matrix{T}, Matrix{T}}

Build response and regressor matrices from levels data for one group.
Returns (Y_eff, X_lag) where Y_eff is (T_i-p) × m and X_lag is (T_i-p) × (m*p).
"""
function _panel_lag_levels(Y::Matrix{T}, p::Int) where {T<:AbstractFloat}
    Ti = size(Y, 1)
    Ti <= p && throw(ArgumentError("Not enough observations ($Ti) for $p lags"))
    Y_eff = Y[(p+1):end, :]
    X_lag = _panel_lag(Y, p)
    Y_eff, X_lag
end
