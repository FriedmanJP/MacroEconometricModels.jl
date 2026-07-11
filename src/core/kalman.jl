# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared Kalman filter building blocks — the correct primitives underlying the
consolidated kernel in `core/kalman_kernel.jl`: `_kalman_predict`, `_kalman_update`
(Joseph form + `safe_cholesky` + triangular solves, T058), `_rts_smoother_gain`,
and `_solve_discrete_lyapunov`.

Since T147/#246 the consolidated `_kalman_filter!` / `_rts_smoother` back the ARIMA,
Beveridge-Nelson, dynamic-factor, and nowcast filters/smoothers (retiring their
divergent non-Joseph updates, `robust_inv`/`det`-gated likelihoods, and hard-coded
`1e6·I`/`10·I` inits). The DSGE `_kalman_loglikelihood` deliberately keeps its
hand-tuned zero-allocation forward pass — it is the Bayesian-estimation hot loop, and a
kernel migration was measured byte-equivalent but 2.4x slower / 29x allocation — and is
guarded byte-equivalent to the kernel by a regression test in `test/core/test_kalman.jl`.
(A follow-on that preallocates the kernel's multivariate path would let the DSGE filter
migrate without regression and unlock the steady-state gain-freeze dividend.)
"""

# =============================================================================
# Discrete Lyapunov Solver
# =============================================================================

"""
    _solve_discrete_lyapunov(T_mat, Q; max_iter=1000, tol=1e-10)

Solve the discrete Lyapunov equation P = T * P * T' + Q by iteration.
Returns the steady-state covariance matrix P as `Hermitian`.

Used for initializing Kalman filter state covariance when the system is stationary.
The iteration starts from P = I and converges when the element-wise maximum
absolute change is below `tol`. Warns (but does not error) when the transition matrix
is not stable (spectral radius ≥ 1) — no finite stationary solution exists — or when
the iteration does not converge within `max_iter`.
"""
function _solve_discrete_lyapunov(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T};
                                   max_iter::Int=1000, tol::Real=1e-10) where {T<:AbstractFloat}
    n = size(T_mat, 1)
    rho = maximum(abs, eigvals(Matrix(T_mat)))
    rho >= one(T) && @warn "Discrete Lyapunov: transition matrix not stable (spectral radius $(rho) ≥ 1); P = T·P·T'+Q has no finite stationary solution. Consider diffuse initialization." maxlog = 1
    P = Matrix{T}(I(n))
    P_new = P
    delta = T(Inf)
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        delta = norm(P_new - P)
        if delta < tol * max(norm(P), one(T))
            return Hermitian(P_new)
        end
        P = P_new
    end
    @warn "Discrete Lyapunov iteration did not converge to tol=$(tol) in $(max_iter) iterations (last change $(delta))." maxlog = 1
    return Hermitian(P_new)
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
- `S = H * P_pred * H' + R` -- innovation covariance (returned unsymmetrized/exact)
- `K = P_pred * H' * S^{-1}` -- Kalman gain, formed via triangular solves on `chol(S)`
- `x_upd = x_pred + K * v` -- updated state
- `P_upd = (I-KH) P_pred (I-KH)' + K R K'` -- Joseph-stabilized covariance (symmetrized)

The Joseph form keeps `P_upd` symmetric and PSD under finite precision even away from
the exact optimal gain, unlike the shorthand `(I - K H) P_pred`.
"""
function _kalman_update(x_pred::AbstractVector{T}, P_pred::AbstractMatrix{T},
                        y::AbstractVector{T}, H::AbstractMatrix{T},
                        R::AbstractMatrix{T}) where {T<:AbstractFloat}
    v = y - H * x_pred
    S = H * P_pred * H' + R
    Ssym = (S + S') / 2                          # symmetrize before factoring
    Sc = safe_cholesky(Hermitian(Ssym))          # lower factor, Ssym = Sc*Sc'
    # Gain K = P_pred H' S^{-1} via triangular solves: K' = Sc' \ (Sc \ (H * P_pred')).
    Kt = Sc' \ (Sc \ (H * P_pred'))
    K = Matrix{T}(Kt')
    x_upd = x_pred + K * v
    IKH = I - K * H
    P_upd = IKH * P_pred * IKH' + K * R * K'     # Joseph stabilized form
    P_upd = (P_upd + P_upd') / 2                 # symmetrize
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
