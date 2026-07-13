# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Kalman filter and smoother with missing data (NaN-aware).

Handles arbitrary NaN patterns per time step by eliminating missing rows
from the observation equation before each update step. Based on the approach
in Bańbura & Modugno (2014).
"""

# =============================================================================
# Missing Data Handler
# =============================================================================

"""
    _miss_data(y, C, R) -> (y_obs, C_obs, R_obs, obs_idx)

Eliminate rows corresponding to NaN observations.

Returns reduced observation vector, loadings, noise covariance, and
the indices of observed (non-NaN) entries.
"""
function _miss_data(y::AbstractVector{T}, C::AbstractMatrix{T},
                    R::AbstractMatrix{T}) where {T<:AbstractFloat}
    obs_idx = findall(!isnan, y)
    if isempty(obs_idx)
        return T[], Matrix{T}(undef, 0, size(C, 2)), Matrix{T}(undef, 0, 0), Int[]
    end
    y_obs = y[obs_idx]
    C_obs = C[obs_idx, :]
    R_obs = R[obs_idx, obs_idx]
    return y_obs, C_obs, R_obs, obs_idx
end

# =============================================================================
# Kalman Filter with Missing Data
# =============================================================================

"""
    _kalman_filter_missing(y, A, C, Q, R, x0, P0) -> (x_pred, P_pred, x_filt, P_filt, loglik)

Forward Kalman filter handling NaN observations.

At each time step, rows with NaN are removed from the observation equation.
If all observations are missing, the update step is skipped (prior = posterior).

# Arguments
- `y::Matrix{T}` — N × T_obs observation matrix (columns = time steps)
- `A::Matrix{T}` — state transition matrix (state_dim × state_dim)
- `C::Matrix{T}` — observation matrix (N × state_dim)
- `Q::Matrix{T}` — state noise covariance
- `R::Matrix{T}` — observation noise covariance
- `x0::Vector{T}` — initial state mean
- `P0::Matrix{T}` — initial state covariance

# Returns
- `x_pred` — predicted states (state_dim × T_obs)
- `P_pred` — predicted covariances
- `x_filt` — filtered states
- `P_filt` — filtered covariances
- `loglik` — log-likelihood
"""
function _kalman_filter_missing(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                                C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                                R::AbstractMatrix{T}, x0::AbstractVector{T},
                                P0::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    # Route through the consolidated Kalman kernel (T147/#246). State x_t = A x_{t-1} + η (Q),
    # obs y_t = C x_t + ν (R); NaN entries are treated as missing (row-dropped per step). The
    # caller supplies the initial x0/P0 (predict-at-top, x0 = a_{0|0}). The kernel's store fields
    # are already in this filter's time-last [:,:,t] layout, so they are returned verbatim.
    # Byte-stable vs the old filter (Joseph replaces (I-KC)P; safe_cholesky + triangular solves
    # replace robust_inv; the always-add likelihood is unchanged).
    store = KalmanFilterStore{T}(state_dim, T_obs)
    loglik = _kalman_filter!(store, y, C, A, Q, Matrix{T}(R);
                             a0=x0, P0=Matrix{T}(P0), scalar=false)
    return store.a_pred, store.P_pred, store.a_filt, store.P_filt, loglik
end

# =============================================================================
# Kalman Smoother with Missing Data
# =============================================================================

"""
    _kalman_smoother_missing(y, A, C, Q, R, x0, P0) -> (x_smooth, P_smooth, PP_smooth, loglik)

Kalman smoother (Harvey 1989 fixed-interval) with NaN handling.

Runs forward filter then backward smoother pass.

# Returns
- `x_smooth` — smoothed states (state_dim × T_obs)
- `P_smooth` — smoothed covariances (state_dim × state_dim × T_obs)
- `PP_smooth` — cross-covariances E[x_t x_{t-1}'] (state_dim × state_dim × T_obs)
- `loglik` — log-likelihood
"""
function _kalman_smoother_missing(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                                  C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                                  R::AbstractMatrix{T}, x0::AbstractVector{T},
                                  P0::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    # Forward filter (kernel) + RTS smoother with lag-1 cross-cov (T147/#246). Byte-stable vs
    # the old smoother; kernel store fields stay in this filter's time-last [:,:,t] layout.
    store = KalmanFilterStore{T}(state_dim, T_obs)
    loglik = _kalman_filter!(store, y, C, A, Q, Matrix{T}(R);
                             a0=x0, P0=Matrix{T}(P0), scalar=false)
    x_smooth, P_smooth, Plag = _rts_smoother(store, A; nlag=1)

    # Uncentered second moment E[x_t x_{t-1}'] for the EM sufficient statistics:
    #   PP[:,:,t] = Cov(x_t,x_{t-1}|Y) + x_smooth[t] x_smooth[t-1]'   (t ≥ 2)
    #   PP[:,:,1] = P_smooth[1] J_0' + x_smooth[1] x0'                (t=1, using the initial x0/P0)
    PP_smooth = zeros(T, state_dim, state_dim, T_obs)
    for t in 2:T_obs
        PP_smooth[:, :, t] = Plag[1][:, :, t] + x_smooth[:, t] * x_smooth[:, t-1]'
    end
    J_0 = Matrix{T}(P0) * A' * robust_inv(store.P_pred[:, :, 1])
    PP_smooth[:, :, 1] = P_smooth[:, :, 1] * J_0' + x_smooth[:, 1] * x0'

    return x_smooth, P_smooth, PP_smooth, loglik
end

# =============================================================================
# Kalman Smoother with Lagged Cross-Covariances
# =============================================================================

"""
    _kalman_smoother_lag(y, A, C, Q, R, x0, P0, k) -> (x_smooth, P_smooth, Plag, loglik)

Extended Kalman smoother computing lagged cross-covariances up to lag k.

Needed for news decomposition (Bańbura & Modugno 2014).

# Returns
- `x_smooth` — smoothed states
- `P_smooth` — smoothed covariances
- `Plag` — vector of k cross-covariance arrays: Plag[j] = E[x_t x_{t-j}'] - E[x_t]E[x_{t-j}]'
- `loglik` — log-likelihood
"""
function _kalman_smoother_lag(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                              C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                              R::AbstractMatrix{T}, x0::AbstractVector{T},
                              P0::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    # Forward filter (kernel) + RTS smoother with lag-k cross-covariances (T147/#246). The
    # kernel's _rts_smoother returns exactly the centered Cov(x_t, x_{t-j}|Y_T) this news
    # decomposition (Bańbura & Modugno 2014) consumes. Byte-stable vs the old smoother.
    store = KalmanFilterStore{T}(state_dim, T_obs)
    loglik = _kalman_filter!(store, y, C, A, Q, Matrix{T}(R);
                             a0=x0, P0=Matrix{T}(P0), scalar=false)
    x_smooth, P_smooth, Plag = _rts_smoother(store, A; nlag=k)

    return x_smooth, P_smooth, Plag, loglik
end
