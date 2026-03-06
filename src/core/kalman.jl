# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared Kalman filter building blocks.

Provides reusable primitives for the domain-specific Kalman filter implementations
in `factor/kalman.jl`, `arima/kalman.jl`, and `nowcast/kalman_missing.jl`.

These are documented utilities, not replacements for the domain-specific filters.
"""

# =============================================================================
# Discrete Lyapunov Solver
# =============================================================================

"""
    _solve_discrete_lyapunov(T_mat, Q; max_iter=1000, tol=1e-10)

Solve the discrete Lyapunov equation P = T * P * T' + Q by iteration.
Returns the steady-state covariance matrix P as `Hermitian`.

Used for initializing Kalman filter state covariance when the system is stationary.
The iteration starts from P = Q and converges when the element-wise maximum
absolute change is below `tol`.
"""
function _solve_discrete_lyapunov(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T};
                                   max_iter::Int=1000, tol::Real=1e-10) where {T<:AbstractFloat}
    n = size(T_mat, 1)
    P = Matrix{T}(I(n))
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        if norm(P_new - P) < tol * max(norm(P), one(T))
            return Hermitian(P_new)
        end
        P = P_new
    end
    return Hermitian(P)
end

# =============================================================================
# Kalman Filter Prediction Step
# =============================================================================

"""
    _kalman_predict(x, P, F, Q)

Kalman filter prediction step.

Returns `(x_pred, P_pred)` where:
- `x_pred = F * x` -- predicted state
- `P_pred = F * P * F' + Q` -- predicted covariance
"""
function _kalman_predict(x::AbstractVector{T}, P::AbstractMatrix{T},
                         F::AbstractMatrix{T}, Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    x_pred = F * x
    P_pred = F * P * F' + Q
    return x_pred, P_pred
end

# =============================================================================
# Kalman Filter Measurement Update Step
# =============================================================================

"""
    _kalman_update(x_pred, P_pred, y, H, R)

Kalman filter measurement update step.

Returns `(x_upd, P_upd, v, S, K)` where:
- `v = y - H * x_pred` -- innovation
- `S = H * P_pred * H' + R` -- innovation covariance
- `K = P_pred * H' * S^{-1}` -- Kalman gain
- `x_upd = x_pred + K * v` -- updated state
- `P_upd = (I - K * H) * P_pred` -- updated covariance
"""
function _kalman_update(x_pred::AbstractVector{T}, P_pred::AbstractMatrix{T},
                        y::AbstractVector{T}, H::AbstractMatrix{T},
                        R::AbstractMatrix{T}) where {T<:AbstractFloat}
    v = y - H * x_pred
    S = H * P_pred * H' + R
    S_inv = robust_inv(Hermitian(S))
    K = P_pred * H' * Matrix{T}(S_inv)
    x_upd = x_pred + K * v
    P_upd = (I - K * H) * P_pred
    return x_upd, Matrix{T}(P_upd), v, S, K
end

# =============================================================================
# Rauch-Tung-Striebel Smoother Gain
# =============================================================================

"""
    _rts_smoother_gain(P_filt, F, P_pred)

Compute the Rauch-Tung-Striebel smoother gain.

Returns `J = P_filt * F' * P_pred^{-1}`.
"""
function _rts_smoother_gain(P_filt::AbstractMatrix{T}, F::AbstractMatrix{T},
                            P_pred::AbstractMatrix{T}) where {T<:AbstractFloat}
    P_pred_inv = robust_inv(Hermitian(P_pred))
    return P_filt * F' * Matrix{T}(P_pred_inv)
end
