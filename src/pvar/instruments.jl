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
GMM instrument matrix construction for Panel VAR — block-diagonal instruments per
Holtz-Eakin, Newey & Rosen (1988), FD-GMM (Arellano-Bond) and System GMM
(Blundell-Bond) instrument sets.
"""

# =============================================================================
# FD-GMM instruments (Arellano-Bond)
# =============================================================================

"""
    _build_instruments_fd(Y_levels::Matrix{T}, p::Int, m::Int;
                          min_lag::Int=2, max_lag::Int=99,
                          collapse::Bool=false) -> Matrix{T}

Build block-diagonal instrument matrix for FD-GMM (Arellano-Bond).

For each period t (t = p+2, ..., T): instruments are y_{i,1}, ..., y_{i,t-2}
(levels dated t-2 and earlier). The block-diagonal structure stacks instruments
per period, with the number of instruments growing with t.

# Arguments
- `Y_levels` — T_i × m matrix of levels data for one group
- `p` — number of lags
- `m` — number of endogenous variables
- `min_lag` — minimum instrument lag (default 2)
- `max_lag` — maximum instrument lag (default 99 = all available)
- `collapse` — if true, collapse instruments (one column per lag, not per period)

# Returns
- Instrument matrix: T_eff × n_instruments
"""
function _build_instruments_fd(Y_levels::Matrix{T}, p::Int, m_dim::Int;
                               min_lag::Int=2, max_lag::Int=99,
                               collapse::Bool=false) where {T<:AbstractFloat}
    Ti = size(Y_levels, 1)
    # After taking p lags and first-differencing, effective periods: p+2, ..., T
    # That gives T_eff = T - p - 1 effective transformed observations
    T_eff = Ti - p - 1
    T_eff < 1 && throw(ArgumentError("Not enough observations for FD-GMM instruments"))

    if collapse
        # Collapsed instruments: one column per lag distance, not per period
        max_actual = min(max_lag, Ti - 1)
        n_lags = max_actual - min_lag + 1
        n_lags < 1 && throw(ArgumentError("No valid instrument lags"))
        n_inst = n_lags * m_dim
        Z = zeros(T, T_eff, n_inst)
        for s in 1:T_eff
            t = p + 1 + s  # period index in levels (transformed period)
            for lag_idx in 1:n_lags
                lag = min_lag + lag_idx - 1
                t_inst = t - lag
                if 1 <= t_inst <= Ti
                    col_start = (lag_idx - 1) * m_dim + 1
                    Z[s, col_start:(col_start + m_dim - 1)] = Y_levels[t_inst, :]
                end
            end
        end
        return Z
    end

    # Standard (non-collapsed) block-diagonal instruments
    # Count total instrument columns
    n_inst_cols = 0
    for s in 1:T_eff
        t = p + 1 + s
        for lag in min_lag:min(max_lag, t - 1)
            t_inst = t - lag
            if 1 <= t_inst <= Ti
                n_inst_cols += m_dim
            end
        end
    end

    # Actually build as block-diagonal: each period gets its own set of columns
    # For efficiency, figure out max instruments per period first
    inst_per_period = Int[]
    for s in 1:T_eff
        t = p + 1 + s
        n_valid = 0
        for lag in min_lag:min(max_lag, t - 1)
            if 1 <= (t - lag) <= Ti
                n_valid += 1
            end
        end
        push!(inst_per_period, n_valid * m_dim)
    end
    total_cols = sum(inst_per_period)

    Z = zeros(T, T_eff, total_cols)
    col_offset = 0
    for s in 1:T_eff
        t = p + 1 + s
        local_col = 0
        for lag in min_lag:min(max_lag, t - 1)
            t_inst = t - lag
            if 1 <= t_inst <= Ti
                for j in 1:m_dim
                    Z[s, col_offset + local_col + j] = Y_levels[t_inst, j]
                end
                local_col += m_dim
            end
        end
        col_offset += inst_per_period[s]
    end
    Z
end

# =============================================================================
# System GMM instruments (Blundell-Bond)
# =============================================================================

"""
    _build_instruments_system(Y_levels::Matrix{T}, p::Int, m::Int;
                              min_lag::Int=2, max_lag::Int=99,
                              collapse::Bool=false,
                              include_constant::Bool=true) -> Matrix{T}

Build instrument matrix for System GMM (Blundell-Bond, 1998).

Stacks two equation sets:
1. **Differenced equations**: instruments are lagged levels (same as FD-GMM)
2. **Level equations**: instruments are lagged first-differences (lag 1)

The constant (if included) enters only the level equation block.

Returns instrument matrix for stacked [differenced; levels] system.
"""
function _build_instruments_system(Y_levels::Matrix{T}, p::Int, m_dim::Int;
                                   min_lag::Int=2, max_lag::Int=99,
                                   collapse::Bool=false,
                                   include_constant::Bool=true) where {T<:AbstractFloat}
    Ti = size(Y_levels, 1)
    T_eff_fd = Ti - p - 1  # effective obs for differenced equations
    T_eff_lev = Ti - p - 1  # effective obs for level equations (matched periods)

    T_eff_fd < 1 && throw(ArgumentError("Not enough observations for System GMM"))

    # Block 1: FD instruments (lagged levels for differenced equations)
    Z_fd = _build_instruments_fd(Y_levels, p, m_dim;
                                 min_lag=min_lag, max_lag=max_lag, collapse=collapse)
    n_fd_cols = size(Z_fd, 2)

    # Block 2: Level instruments (lagged differences for level equations)
    # For each period, use Δy_{t-1} as instrument for the level equation
    Z_lev_cols = m_dim  # one lag of differences
    if include_constant
        Z_lev_cols += 1
    end

    Z_lev = zeros(T, T_eff_lev, Z_lev_cols)
    for s in 1:T_eff_lev
        t = p + 1 + s
        col = 0
        # Lagged first-difference: Δy_{t-1} = y_{t-1} - y_{t-2}
        if t - 1 >= 2
            for j in 1:m_dim
                Z_lev[s, col + j] = Y_levels[t - 1, j] - Y_levels[t - 2, j]
            end
        end
        col += m_dim
        if include_constant
            Z_lev[s, col + 1] = one(T)
        end
    end

    # Stack: block-diagonal [Z_fd 0; 0 Z_lev]
    total_rows = T_eff_fd + T_eff_lev
    total_cols = n_fd_cols + Z_lev_cols
    Z = zeros(T, total_rows, total_cols)
    Z[1:T_eff_fd, 1:n_fd_cols] = Z_fd
    Z[(T_eff_fd+1):end, (n_fd_cols+1):end] = Z_lev

    Z
end

# =============================================================================
# PCA instrument reduction
# =============================================================================

"""
    _pca_reduce_instruments(Z::Matrix{T}; max_components::Int=0) -> Matrix{T}

Reduce instrument count via PCA (principal components of instrument matrix).
Useful when the instrument count proliferates with T.

- `max_components`: maximum number of principal components to keep (0 = auto: rank)
"""
function _pca_reduce_instruments(Z::Matrix{T}; max_components::Int=0) where {T<:AbstractFloat}
    n, q = size(Z)
    q <= n && max_components <= 0 && return Z  # not too many instruments

    # Center
    Z_c = Z .- mean(Z, dims=1)

    # SVD
    F = svd(Z_c)
    # Keep components with non-trivial singular values
    tol = T(1e-10) * F.S[1]
    r = count(s -> s > tol, F.S)
    if max_components > 0
        r = min(r, max_components)
    end
    r = min(r, n - 1)  # can't have more components than observations

    # Return scores (n × r)
    F.U[:, 1:r] .* F.S[1:r]'
end
