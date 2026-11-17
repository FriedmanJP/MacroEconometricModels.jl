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

# =============================================================================
# Discrete Algebraic Riccati Equation (DARE) Solver
# =============================================================================

function _dare_eval(P::Matrix{T}, Am::Matrix{T}, Qm::Matrix{T},
                    Cm::Matrix{T}, Rm::Matrix{T}, Sm::Matrix{T}) where {T<:AbstractFloat}
    n, m = size(Sm)
    F_raw = Cm * P * Cm' + Rm
    F_innov = Matrix{T}((F_raw + F_raw') / 2)
    if m == 0
        K = zeros(T, n, 0)
        R_mat = P - (Am * P * Am' + Qm)
        denom = max(norm(P), norm(Qm), one(T))
        return K, F_innov, T(norm(R_mat) / denom)
    end
    APC_S = Am * P * Cm' + Sm
    K = try
        cf = cholesky(Hermitian(F_innov))
        Matrix{T}((cf \ APC_S')')
    catch
        Matrix{T}(APC_S * Matrix{T}(robust_inv(Hermitian(F_innov); silent=true)))
    end
    R_mat = P - (Am * P * Am' + Qm - K * F_innov * K')
    denom = max(norm(P), norm(Qm), one(T))
    resid = T(norm(R_mat) / denom)
    return K, F_innov, resid
end

function _dare_is_stabilizing(P::Matrix{T}, Am::Matrix{T},
                              K::Matrix{T}, Cm::Matrix{T}) where {T<:AbstractFloat}
    (all(isfinite, P) && all(isfinite, K)) || return false
    maximum(abs, eigvals(Am - K * Cm)) < one(T) || return false
    min_ev = minimum(eigvals(Symmetric(P)))
    return min_ev >= -sqrt(eps(T)) * max(opnorm(P), one(T))
end

function _dare_extract_schur(F, n::Int, rcond_tol::T) where {T<:AbstractFloat}
    sel = abs.(F.alpha) .< abs.(F.beta)
    count(sel) == n || return nothing
    try
        ordschur!(F, sel)
    catch e
        e isa LinearAlgebra.LAPACKException && return nothing
        rethrow()
    end
    Z11 = Matrix{T}(@view F.Z[1:n, 1:n])
    Z21 = Matrix{T}(@view F.Z[n+1:2n, 1:n])
    luZ = lu(Z11; check=false)
    issuccess(luZ) || return nothing
    Z11_inv = inv(luZ)
    kappa_Z = opnorm(Z11, 1) * opnorm(Z11_inv, 1)
    (isfinite(kappa_Z) && (one(T) / kappa_Z > rcond_tol)) || return nothing
    P = Z21 * Z11_inv
    all(isfinite, P) || return nothing
    return Matrix{T}((P + P') / 2)
end

function _dare_sda(As::Matrix{T}, Qs::Matrix{T}, Gc::Matrix{T};
                   max_iter::Int=200) where {T<:AbstractFloat}
    n = size(As, 1)
    A_k = copy(As)
    H_k = copy(Qs)
    G_k = copy(Gc)
    In = Matrix{T}(I, n, n)
    tol = T(100) * eps(T)
    for _ in 1:max_iter
        W = In + H_k * G_k
        Wf = lu(W; check=false)
        issuccess(Wf) || return nothing
        WinvAk = Wf \ A_k
        A_next = A_k * WinvAk
        H_next = H_k + A_k * (Wf \ (H_k * A_k'))
        H_next = (H_next + H_next') / 2
        G_next = G_k + (A_k' * G_k) * WinvAk
        G_next = (G_next + G_next') / 2
        (all(isfinite, H_next) && all(isfinite, A_next) && all(isfinite, G_next)) || return nothing
        diff_H = norm(H_next - H_k)
        A_k = A_next
        H_k = H_next
        G_k = G_next
        if diff_H <= tol * max(norm(H_k), one(T))
            return Matrix{T}(H_k)
        end
    end
    return all(isfinite, H_k) ? Matrix{T}(H_k) : nothing
end

"""
    _solve_dare(A, Q, C, R; S=nothing, rcond_tol=1e-12) -> (P, K, F_innov, residual)

Solve the filtering discrete-time algebraic Riccati equation (DARE):

    P = A * P * A' + Q - (A * P * C' + S) * (C * P * C' + R)^{-1} * (A * P * C' + S)'

Uses a `2n × 2n` symplectic pencil generalized real Schur decomposition (`schur` +
`ordschur!`, selecting `|α| < |β|`) when `R` is well-conditioned, automatically switches
to a Van Dooren QR-compressed `(2n+m) × (2n+m) → 2n × 2n` extended pencil when `R` is
singular or ill-conditioned, falls back to the Structure-Preserving Doubling Algorithm
(SDA) if needed, and applies Newton–Kleinman symmetrized Lyapunov polish.

Returns `(P, K, F_innov, residual)` where:
- `P` (`n × n`): symmetric positive-semidefinite stabilizing solution
- `K` (`n × m`): predictor gain `(A * P * C' + S) * F_innov^{-1}` (closed-loop `A - K * C`)
- `F_innov` (`m × m`): symmetrized innovation covariance `C * P * C' + R`
- `residual`: relative Riccati residual `norm(P - RHS) / max(norm(P), norm(Q), 1)`
"""
function _solve_dare(A::AbstractMatrix, Q::AbstractMatrix,
                     C::AbstractMatrix, R::AbstractMatrix;
                     S::Union{Nothing,AbstractMatrix}=nothing,
                     rcond_tol::Real=1e-12)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("A must be square, got $(size(A))"))
    size(Q) == (n, n) || throw(ArgumentError("Q must be $n×$n, got $(size(Q))"))
    m = size(C, 1)
    size(C, 2) == n || throw(ArgumentError("C must have $n columns, got $(size(C))"))
    size(R) == (m, m) || throw(ArgumentError("R must be $m×$m, got $(size(R))"))
    if S !== nothing
        size(S) == (n, m) || throw(ArgumentError("S must be $n×$m, got $(size(S))"))
    end

    T = S === nothing ?
        float(promote_type(eltype(A), eltype(Q), eltype(C), eltype(R))) :
        float(promote_type(eltype(A), eltype(Q), eltype(C), eltype(R), eltype(S)))
    rtol = T(rcond_tol)

    Am = Matrix{T}(A)
    Qraw = Matrix{T}(Q)
    Qm = Matrix{T}((Qraw + Qraw') / 2)
    Cm = Matrix{T}(C)
    Rraw = Matrix{T}(R)
    Rm = Matrix{T}((Rraw + Rraw') / 2)
    Sm = S === nothing ? zeros(T, n, m) : Matrix{T}(S)

    if n == 0
        return zeros(T, 0, 0), zeros(T, 0, m), Rm, zero(T)
    end
    if m == 0
        maximum(abs, eigvals(Am)) >= one(T) &&
            throw(ArgumentError("DARE solver failed to find a stabilizing solution (system is unstable with m = 0 observations)"))
        P0 = Matrix{T}(_dlyap(Am, Qm; tol=T(1e-13), warn_label="_solve_dare"))
        P0 = Matrix{T}((P0 + P0') / 2)
        min_ev = minimum(eigvals(Symmetric(P0)))
        min_ev >= -sqrt(eps(T)) * max(opnorm(P0), one(T)) ||
            throw(ArgumentError("DARE solver failed to find a positive-semidefinite stabilizing solution for m = 0"))
        K0, F0, res0 = _dare_eval(P0, Am, Qm, Cm, Rm, Sm)
        return P0, K0, F0, res0
    end

    luR = lu(Rm; check=false)
    R_ok = false
    Rinv = zeros(T, m, m)
    if issuccess(luR)
        Rinv = inv(luR)
        kappa_R = opnorm(Rm, 1) * opnorm(Rinv, 1)
        R_ok = isfinite(kappa_R) && (one(T) / kappa_R > rtol)
    end

    P_best = nothing
    K_best = zeros(T, n, m)
    F_best = zeros(T, m, m)
    res_best = T(Inf)

    In = Matrix{T}(I, n, n)
    As = Matrix{T}(undef, n, n)
    Qs = Matrix{T}(undef, n, n)
    Gc = Matrix{T}(undef, n, n)

    if R_ok
        As .= Am - Sm * Rinv * Cm
        Qs_raw = Qm - Sm * Rinv * Sm'
        Qs .= (Qs_raw + Qs_raw') / 2
        Gc_raw = Cm' * Rinv * Cm
        Gc .= (Gc_raw + Gc_raw') / 2
        M_pen = [As' zeros(T, n, n); -Qs In]
        L_pen = [In Gc; zeros(T, n, n) As]
        F = LinearAlgebra.schur(M_pen, L_pen)
        P_qz = _dare_extract_schur(F, n, rtol)
        if P_qz !== nothing
            K_qz, F_qz, res_qz = _dare_eval(P_qz, Am, Qm, Cm, Rm, Sm)
            if isfinite(res_qz) && _dare_is_stabilizing(P_qz, Am, K_qz, Cm)
                P_best, K_best, F_best, res_best = P_qz, K_qz, F_qz, res_qz
            end
        end
    end

    # Van Dooren QR-compressed extended pencil when R is singular/ill-conditioned or 2n×2n QZ failed
    if !R_ok || P_best === nothing || res_best > T(1e-10)
        B_ext = [Cm'; -Sm; Rm]
        sv = svd(B_ext)
        smax = isempty(sv.S) ? zero(T) : sv.S[1]
        tol_rank = max(2 * n + m, 1) * eps(T) * smax
        r_eff = smax == zero(T) ? 0 : count(>(tol_rank), sv.S)

        if r_eff == 0
            if P_best === nothing
                maximum(abs, eigvals(Am)) >= one(T) &&
                    throw(ArgumentError("DARE solver failed to find a stabilizing solution (system is unstable with rank-0 observation block)"))
                P0 = Matrix{T}(_dlyap(Am, Qm; tol=T(1e-13), warn_label="_solve_dare"))
                P0 = Matrix{T}((P0 + P0') / 2)
                min_ev = minimum(eigvals(Symmetric(P0)))
                min_ev >= -sqrt(eps(T)) * max(opnorm(P0), one(T)) ||
                    throw(ArgumentError("DARE solver failed to find a positive-semidefinite stabilizing solution"))
                F_raw = Cm * P0 * Cm' + Rm
                F0 = Matrix{T}((F_raw + F_raw') / 2)
                K0 = zeros(T, n, m)
                R_mat = P0 - (Am * P0 * Am' + Qm)
                res0 = T(norm(R_mat) / max(norm(P0), norm(Qm), one(T)))
                return P0, K0, F0, res0
            end
        elseif r_eff < m
            V1 = Matrix{T}(sv.Vt[1:r_eff, :]')
            Cm_red = Matrix{T}(V1' * Cm)
            Rm_red_raw = V1' * Rm * V1
            Rm_red = Matrix{T}((Rm_red_raw + Rm_red_raw') / 2)
            Sm_red = Matrix{T}(Sm * V1)
            P_red, K_red, _, _ = _solve_dare(Am, Qm, Cm_red, Rm_red; S=Sm_red, rcond_tol=rcond_tol)
            K_proj = Matrix{T}(K_red * V1')
            F_raw = Cm * P_red * Cm' + Rm
            F_proj = Matrix{T}((F_raw + F_raw') / 2)
            R_mat = P_red - (Am * P_red * Am' + Qm - K_proj * F_proj * K_proj')
            res_proj = T(norm(R_mat) / max(norm(P_red), norm(Qm), one(T)))
            if isfinite(res_proj) && res_proj < res_best && _dare_is_stabilizing(P_red, Am, K_proj, Cm)
                P_best, K_best, F_best, res_best = P_red, K_proj, F_proj, res_proj
            end
        else
            M_ext = [Am'          zeros(T, n, n) Cm';
                     -Qm          In             -Sm;
                     Sm'          zeros(T, m, n) Rm]
            L_ext = [In              zeros(T, n, n + m);
                     zeros(T, n, n)  Am              zeros(T, n, m);
                     zeros(T, m, n)  -Cm             zeros(T, m, m)]
            qrB = qr(M_ext[:, (2n+1):(2n+m)])
            Qfull = qrB.Q * Matrix{T}(I, 2n + m, 2n + m)
            Qt = Qfull'
            M_red = (Qt * M_ext)[(m+1):(2n+m), 1:2n]
            L_red = (Qt * L_ext)[(m+1):(2n+m), 1:2n]
            F_ext = LinearAlgebra.schur(M_red, L_red)
            P_ext = _dare_extract_schur(F_ext, n, rtol)
            if P_ext !== nothing
                K_ext, F_ext_m, res_ext = _dare_eval(P_ext, Am, Qm, Cm, Rm, Sm)
                if isfinite(res_ext) && res_ext < res_best && _dare_is_stabilizing(P_ext, Am, K_ext, Cm)
                    P_best, K_best, F_best, res_best = P_ext, K_ext, F_ext_m, res_ext
                end
            end
        end
    end

    # SDA fallback if residual is still above 1e-10 or QZ failed (when R is invertible)
    if R_ok && (P_best === nothing || res_best > T(1e-10))
        P_sda = _dare_sda(As, Qs, Gc)
        if P_sda !== nothing
            K_sda, F_sda, res_sda = _dare_eval(P_sda, Am, Qm, Cm, Rm, Sm)
            if isfinite(res_sda) && res_sda < res_best && _dare_is_stabilizing(P_sda, Am, K_sda, Cm)
                P_best, K_best, F_best, res_best = P_sda, K_sda, F_sda, res_sda
            end
        end
    end

    P_best === nothing &&
        throw(ArgumentError("DARE solver failed to find a stabilizing solution (system may be unstabilizable or undetectable)"))

    # Newton–Kleinman symmetrized Lyapunov polish (up to 2 steps)
    for _ in 1:2
        res_best <= T(1e-13) && break
        A_cl = Am - K_best * Cm
        maximum(abs, eigvals(A_cl)) < one(T) - T(1e-10) || break
        Q_cl_raw = Qm + K_best * Rm * K_best' - Sm * K_best' - K_best * Sm'
        Q_cl = Matrix{T}((Q_cl_raw + Q_cl_raw') / 2)
        P_nk = _suppress_warnings() do
            Matrix{T}(_dlyap(A_cl, Q_cl; tol=T(1e-13)))
        end
        all(isfinite, P_nk) || break
        P_nk = Matrix{T}((P_nk + P_nk') / 2)
        K_nk, F_nk, res_nk = _dare_eval(P_nk, Am, Qm, Cm, Rm, Sm)
        if isfinite(res_nk) && res_nk < res_best && _dare_is_stabilizing(P_nk, Am, K_nk, Cm)
            P_best, K_best, F_best, res_best = P_nk, K_nk, F_nk, res_nk
        else
            break
        end
    end

    _dare_is_stabilizing(P_best, Am, K_best, Cm) ||
        throw(ArgumentError("DARE solver failed to find a stabilizing solution (system may be unstabilizable or undetectable)"))

    return P_best, K_best, F_best, res_best
end
