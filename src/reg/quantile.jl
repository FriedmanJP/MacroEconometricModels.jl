# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# MacroEconometricModels.jl — Quantile regression (Koenker & Bassett 1978)
#
# References:
#   Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica 46(1), 33–50.
#   Portnoy, S. & Koenker, R. (1997). The Gaussian Hare and the Laplacian Tortoise:
#     Computability of Squared-Error versus Absolute-Error Estimators.
#     Statistical Science 12(4), 279–300.
#   Koenker, R. (2005). Quantile Regression. Cambridge University Press, Appendix A.
#   Powell, J. L. (1991). Estimation of monotonic regression models under quantile
#     restrictions. In Nonparametric and Semiparametric Methods in Econometrics.
#   Hall, P. & Sheather, S. J. (1988). On the distribution of a Studentized quantile.

using LinearAlgebra
using Statistics

# =============================================================================
# Check function and the solver
# =============================================================================

"""
    _check_loss(u, tau) → T

Koenker-Bassett check-function loss `Σ ρ_τ(u_i)` with `ρ_τ(u) = u(τ − 1{u < 0})`.
Asymmetric absolute loss: over-prediction is penalized `1-τ` and under-prediction `τ`, so
the minimizer is the conditional `τ`-quantile.
"""
function _check_loss(u::AbstractVector{T}, tau::T) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for ui in u
        acc += ui * (tau - (ui < zero(T) ? one(T) : zero(T)))
    end
    return acc
end

"""
    _qreg_fit(X, y, tau; maxit, tol, n_stage) → (beta, converged)

Minimize the check loss by iteratively reweighted least squares.

The check function is majorized by a weighted quadratic, `ρ_τ(u) ≈ w u²` with
`w = |τ − 1{u<0}| / |u|`, so each step is an ordinary weighted least-squares solve. The
`|u|` in the denominator is floored at `delta`, and `delta` is annealed down across
`n_stage` stages: a fixed floor would leave a smoothed (Huber-like) solution rather than the
true piecewise-linear one, and annealing is what drives it onto the exact LP vertex.

Verified against exact enumeration of the basic solutions — the Koenker-Bassett optimum
interpolates `k` observations exactly, so for small problems every `k`-subset can be checked.
Agreement is to 1e-12..1e-9 in the coefficients.
"""
function _qreg_fit(X::Matrix{T}, y::Vector{T}, tau::T;
                   maxit::Int=200, tol::Real=1e-10, n_stage::Int=12) where {T<:AbstractFloat}
    n, k = size(X)
    b = X \ y                                    # OLS start
    delta = one(T)
    converged = false
    for _ in 1:n_stage
        for _ in 1:maxit
            u = y .- X * b
            w = similar(u)
            @inbounds for i in eachindex(u)
                w[i] = abs(tau - (u[i] < zero(T) ? one(T) : zero(T))) /
                       max(abs(u[i]), delta)
            end
            Xw = X .* w
            A = Xw' * X
            bnew = Matrix{T}(robust_inv(Symmetric(A))) * (Xw' * y)
            step = maximum(abs.(bnew .- b))
            b = bnew
            if step < T(tol)
                converged = true
                break
            end
        end
        delta = max(delta / 10, T(1e-12))
    end
    return b, converged
end

"""
    _QRegWorkspace{T<:AbstractFloat}(n::Int, k::Int)
    _QRegWorkspace(n::Int, k::Int)

Pre-allocated workspace for zero-allocation repeated quantile regression solves via
[`_qreg_ipm!`](@ref) with fixed dimensions `(n, k)`.

Stores primal slacks (`u`, `v`), dual variables (`d`, `s`), normal weights (`w`, `denom`),
affine predictor steps (`du_aff`, `dv_aff`, `dd_aff`, `ds_aff`), corrector steps
(`du`, `dv`, `dd`, `ds`), residuals (`r_p`, `r_ds`, `r_d`), scratch vectors
(`xi`, `Xb`, `xt1`, `rhs`, `dbeta`), normal matrix buffers (`Xw`, `A`, `A_fac`),
and post-solve diagnostics (`iters::Int`, `gap::T`).
"""
mutable struct _QRegWorkspace{T<:AbstractFloat}
    n::Int
    k::Int
    iters::Int
    gap::T
    u::Vector{T}
    v::Vector{T}
    d::Vector{T}
    s::Vector{T}
    w::Vector{T}
    denom::Vector{T}
    du_aff::Vector{T}
    dv_aff::Vector{T}
    dd_aff::Vector{T}
    ds_aff::Vector{T}
    du::Vector{T}
    dv::Vector{T}
    dd::Vector{T}
    ds::Vector{T}
    r_p::Vector{T}
    r_ds::Vector{T}
    xi::Vector{T}
    Xb::Vector{T}
    r_d::Vector{T}
    xt1::Vector{T}
    rhs::Vector{T}
    dbeta::Vector{T}
    Xw::Matrix{T}
    A::Matrix{T}
    A_fac::Matrix{T}

    function _QRegWorkspace{T}(n::Int, k::Int) where {T<:AbstractFloat}
        (n > 0 && k > 0) ||
            throw(ArgumentError("Workspace dimensions must be positive, got n=$n, k=$k"))
        return new{T}(
            n, k, 0, zero(T),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, k), Vector{T}(undef, k),
            Vector{T}(undef, k), Vector{T}(undef, k),
            Matrix{T}(undef, n, k),
            Matrix{T}(undef, k, k),
            Matrix{T}(undef, k, k)
        )
    end
end

_QRegWorkspace(n::Int, k::Int) = _QRegWorkspace{Float64}(n, k)

function _qreg_factor_normal!(ws::_QRegWorkspace{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, k = ws.n, ws.k
    @inbounds for j in 1:k
        for i in 1:n
            ws.Xw[i, j] = ws.w[i] * X[i, j]
        end
    end
    mul!(ws.A, transpose(ws.Xw), X)
    @inbounds for j in 1:k
        for i in 1:(j - 1)
            avg = T(0.5) * (ws.A[i, j] + ws.A[j, i])
            ws.A[i, j] = avg
            ws.A[j, i] = avg
        end
    end
    copyto!(ws.A_fac, ws.A)
    F = cholesky!(Hermitian(ws.A_fac); check=false)
    issuccess(F) && return F
    # Zero-allocation scale-relative diagonal jitter fallback for ill-conditioned weights / Float32
    trA = zero(T)
    @inbounds for i in 1:k
        trA += ws.A[i, i]
    end
    s = trA / T(k)
    rel_base = max(T(1e-10), T(10) * eps(T))
    base = s > zero(T) ? rel_base * s : rel_base
    for scale in (1, 10, 100, 1_000, 10_000, 100_000)
        copyto!(ws.A_fac, ws.A)
        applied = T(scale) * base
        @inbounds for i in 1:k
            ws.A_fac[i, i] += applied
        end
        F = cholesky!(Hermitian(ws.A_fac); check=false)
        issuccess(F) && return F
    end
    return F
end

function _qreg_solve_normal!(out::AbstractVector{T}, ws::_QRegWorkspace{T},
                             F::Cholesky{T,Matrix{T}},
                             rhs::AbstractVector{T}) where {T<:AbstractFloat}
    if issuccess(F)
        if out === ws.dbeta || (out isa StridedVector && stride(out, 1) == 1)
            ldiv!(out, F, rhs)
            if all(isfinite, out)
                return nothing
            end
        else
            ldiv!(ws.dbeta, F, rhs)
            if all(isfinite, ws.dbeta)
                copyto!(out, ws.dbeta)
                return nothing
            end
        end
    end
    try
        L = safe_cholesky(ws.A; silent=true)
        sol = L' \ (L \ rhs)
        if all(isfinite, sol)
            copyto!(out, sol)
            return nothing
        end
    catch
    end
    Ainv = Matrix{T}(robust_inv(Symmetric(ws.A); silent=true))
    if out === ws.dbeta || (out isa StridedVector && stride(out, 1) == 1)
        mul!(out, Ainv, rhs)
    else
        mul!(ws.dbeta, Ainv, rhs)
        copyto!(out, ws.dbeta)
    end
    return nothing
end

"""
    _qreg_ipm!(beta, X, y, tau, ws; tol=1e-10, maxit=60) → (beta, converged)
    _qreg_ipm!(beta, X, y, tau; tol=1e-10, maxit=60) → (beta, converged)

Minimize the quantile check loss `Σ ρ_τ(y_i - x_i'β)` using the Portnoy–Koenker (1997)
primal–dual Mehrotra predictor–corrector interior-point method (Koenker 2005, Appendix A).

Solves the bounded-variable dual LP `max y'd` subject to `X'd = (1 - τ)X'1` and
`0 ≤ d ≤ 1`, performing one normal-matrix Cholesky factorization per iteration via
`cholesky!(Hermitian(...); check=false)` with `safe_cholesky` / `robust_inv` fallback for
collinear or rank-deficient designs. Stores iteration count and final duality gap in
`ws.iters` and `ws.gap`.
"""
function _qreg_ipm!(beta::AbstractVector{T}, X::AbstractMatrix{T},
                    y::AbstractVector{T}, tau::T, ws::_QRegWorkspace{T};
                    tol::Real=1e-10, maxit::Int=60) where {T<:AbstractFloat}
    n, k = ws.n, ws.k
    size(X) == (n, k) ||
        throw(DimensionMismatch("X must have size ($n, $k), got $(size(X))"))
    length(y) == n ||
        throw(DimensionMismatch("y must have length $n, got $(length(y))"))
    length(beta) == k ||
        throw(DimensionMismatch("beta must have length $k, got $(length(beta))"))
    zero(T) < tau < one(T) ||
        throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
    maxit >= 1 || throw(ArgumentError("maxit must be >= 1, got $maxit"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))

    # Precompute dual equality RHS: (1 - tau) * X' * 1
    one_minus_tau = one(T) - tau
    @inbounds for j in 1:k
        acc = zero(T)
        for i in 1:n
            acc += X[i, j]
        end
        ws.xt1[j] = one_minus_tau * acc
    end

    # Problem norms for scale-aware stopping
    norm_y_sq = zero(T)
    @inbounds for i in 1:n
        yi = y[i]
        norm_y_sq += yi * yi
    end
    norm_y = sqrt(norm_y_sq)
    norm_X_sq = zero(T)
    @inbounds for j in 1:k
        for i in 1:n
            xij = X[i, j]
            norm_X_sq += xij * xij
        end
    end
    norm_X = sqrt(norm_X_sq)

    # OLS start via normal equations with collinearity fallback (solve into contiguous ws.dbeta)
    fill!(ws.w, one(T))
    F0 = _qreg_factor_normal!(ws, X)
    mul!(ws.rhs, transpose(X), y)
    _qreg_solve_normal!(ws.dbeta, ws, F0, ws.rhs)
    copyto!(beta, ws.dbeta)
    if !all(isfinite, beta)
        fill!(beta, zero(T))
    end

    # Initialize primal slacks and feasible dual variables
    mul!(ws.Xb, X, beta)
    sum_abs_r0 = zero(T)
    @inbounds for i in 1:n
        sum_abs_r0 += abs(y[i] - ws.Xb[i])
    end
    delta0 = max(sum_abs_r0 / T(n), T(1e-2))
    d0 = one_minus_tau
    s0 = tau
    @inbounds for i in 1:n
        r0_i = y[i] - ws.Xb[i]
        ws.u[i] = max(r0_i, zero(T)) + delta0
        ws.v[i] = max(-r0_i, zero(T)) + delta0
        ws.d[i] = d0
        ws.s[i] = s0
    end

    tol_T = T(tol)
    mach_tol = T(500) * eps(T)
    feas_tol = max(tol_T, mach_tol, sqrt(eps(T)), T(1e-6))
    gamma = one(T) - max(T(5e-5), sqrt(eps(T)))
    tiny = floatmin(T)
    two_n = T(2 * n)

    # Initial residuals and duality gap
    mul!(ws.r_d, transpose(X), ws.d)
    @inbounds for j in 1:k
        ws.r_d[j] = ws.xt1[j] - ws.r_d[j]
    end
    gap = zero(T)
    p_obj = zero(T)
    @inbounds for i in 1:n
        ui = ws.u[i]
        vi = ws.v[i]
        di = ws.d[i]
        si = ws.s[i]
        ws.r_p[i] = y[i] - ws.Xb[i] - ui + vi
        ws.r_ds[i] = one(T) - di - si
        gap += ui * si + vi * di
        p_obj += tau * ui + one_minus_tau * vi
    end

    converged = false
    ws.iters = 0
    ws.gap = gap

    for iter in 1:maxit
        # Normal matrix weights and affine RHS
        @inbounds for i in 1:n
            ui = ws.u[i]
            vi = ws.v[i]
            di = ws.d[i]
            si = ws.s[i]
            den = max(ui * di + vi * si, tiny)
            ws.denom[i] = den
            ws.w[i] = (di * si) / den
            ws.xi[i] = (di * si * (ws.r_p[i] + ui - vi) + di * ui * ws.r_ds[i]) / den
        end
        F = _qreg_factor_normal!(ws, X)

        # Predictor (affine-scaling) solve
        mul!(ws.rhs, transpose(X), ws.xi)
        @inbounds for j in 1:k
            ws.rhs[j] -= ws.r_d[j]
        end
        _qreg_solve_normal!(ws.dbeta, ws, F, ws.rhs)
        mul!(ws.Xb, X, ws.dbeta)

        alpha_P_aff = one(T)
        alpha_D_aff = one(T)
        @inbounds for i in 1:n
            ui = ws.u[i]
            vi = ws.v[i]
            di = ws.d[i]
            si = ws.s[i]
            Xdb_i = ws.Xb[i]
            ddi = ws.xi[i] - ws.w[i] * Xdb_i
            dsi = ws.r_ds[i] - ddi
            if di >= si
                dvi = (-vi * di - vi * ddi) / di
                dui = ws.r_p[i] - Xdb_i + dvi
            else
                dui = (-ui * si - ui * dsi) / si
                dvi = dui - ws.r_p[i] + Xdb_i
            end
            ws.dd_aff[i] = ddi
            ws.ds_aff[i] = dsi
            ws.du_aff[i] = dui
            ws.dv_aff[i] = dvi
            if dui < zero(T)
                ratio = -ui / dui
                if ratio < alpha_P_aff
                    alpha_P_aff = ratio
                end
            end
            if dvi < zero(T)
                ratio = -vi / dvi
                if ratio < alpha_P_aff
                    alpha_P_aff = ratio
                end
            end
            if ddi < zero(T)
                ratio = -di / ddi
                if ratio < alpha_D_aff
                    alpha_D_aff = ratio
                end
            end
            if dsi < zero(T)
                ratio = -si / dsi
                if ratio < alpha_D_aff
                    alpha_D_aff = ratio
                end
            end
        end

        mu = gap / two_n
        mu_aff = zero(T)
        @inbounds for i in 1:n
            u_try = max(ws.u[i] + alpha_P_aff * ws.du_aff[i], zero(T))
            v_try = max(ws.v[i] + alpha_P_aff * ws.dv_aff[i], zero(T))
            d_try = max(ws.d[i] + alpha_D_aff * ws.dd_aff[i], zero(T))
            s_try = max(ws.s[i] + alpha_D_aff * ws.ds_aff[i], zero(T))
            mu_aff += u_try * s_try + v_try * d_try
        end
        mu_aff /= two_n
        sigma = mu > zero(T) ? clamp((mu_aff / mu)^3, zero(T), one(T)) : zero(T)
        sigma_mu = sigma * mu

        # Corrector RHS
        @inbounds for i in 1:n
            ui = ws.u[i]
            vi = ws.v[i]
            di = ws.d[i]
            si = ws.s[i]
            r_us = sigma_mu - ui * si - ws.du_aff[i] * ws.ds_aff[i]
            r_vd = sigma_mu - vi * di - ws.dv_aff[i] * ws.dd_aff[i]
            ws.xi[i] = (di * (si * ws.r_p[i] - r_us + ui * ws.r_ds[i]) + si * r_vd) /
                       ws.denom[i]
        end

        # Corrector solve reusing factorization F
        mul!(ws.rhs, transpose(X), ws.xi)
        @inbounds for j in 1:k
            ws.rhs[j] -= ws.r_d[j]
        end
        _qreg_solve_normal!(ws.dbeta, ws, F, ws.rhs)
        mul!(ws.Xb, X, ws.dbeta)

        alpha_P = one(T)
        alpha_D = one(T)
        @inbounds for i in 1:n
            ui = ws.u[i]
            vi = ws.v[i]
            di = ws.d[i]
            si = ws.s[i]
            Xdb_i = ws.Xb[i]
            ddi = ws.xi[i] - ws.w[i] * Xdb_i
            dsi = ws.r_ds[i] - ddi
            if di >= si
                r_vd = sigma_mu - vi * di - ws.dv_aff[i] * ws.dd_aff[i]
                dvi = (r_vd - vi * ddi) / di
                dui = ws.r_p[i] - Xdb_i + dvi
            else
                r_us = sigma_mu - ui * si - ws.du_aff[i] * ws.ds_aff[i]
                dui = (r_us - ui * dsi) / si
                dvi = dui - ws.r_p[i] + Xdb_i
            end
            ws.dd[i] = ddi
            ws.ds[i] = dsi
            ws.du[i] = dui
            ws.dv[i] = dvi
            if dui < zero(T)
                ratio = gamma * (-ui / dui)
                if ratio < alpha_P
                    alpha_P = ratio
                end
            end
            if dvi < zero(T)
                ratio = gamma * (-vi / dvi)
                if ratio < alpha_P
                    alpha_P = ratio
                end
            end
            if ddi < zero(T)
                ratio = gamma * (-di / ddi)
                if ratio < alpha_D
                    alpha_D = ratio
                end
            end
            if dsi < zero(T)
                ratio = gamma * (-si / dsi)
                if ratio < alpha_D
                    alpha_D = ratio
                end
            end
        end
        alpha_P = max(alpha_P, zero(T))
        alpha_D = max(alpha_D, zero(T))

        @inbounds for j in 1:k
            beta[j] += alpha_P * ws.dbeta[j]
        end
        @inbounds for i in 1:n
            ws.u[i] = max(ws.u[i] + alpha_P * ws.du[i], tiny)
            ws.v[i] = max(ws.v[i] + alpha_P * ws.dv[i], tiny)
            ws.d[i] = max(ws.d[i] + alpha_D * ws.dd[i], tiny)
            ws.s[i] = max(ws.s[i] + alpha_D * ws.ds[i], tiny)
        end

        # Update residuals, gap, and check convergence
        mul!(ws.Xb, X, beta)
        mul!(ws.r_d, transpose(X), ws.d)
        dual_err = zero(T)
        @inbounds for j in 1:k
            rdj = ws.xt1[j] - ws.r_d[j]
            ws.r_d[j] = rdj
            a = abs(rdj)
            if a > dual_err
                dual_err = a
            end
        end

        gap = zero(T)
        p_obj = zero(T)
        prim_err = zero(T)
        @inbounds for i in 1:n
            ui = ws.u[i]
            vi = ws.v[i]
            di = ws.d[i]
            si = ws.s[i]
            rpi = y[i] - ws.Xb[i] - ui + vi
            rdsi = one(T) - di - si
            ws.r_p[i] = rpi
            ws.r_ds[i] = rdsi
            gap += ui * si + vi * di
            p_obj += tau * ui + one_minus_tau * vi
            ap = abs(rpi)
            ads = abs(rdsi)
            if ap > prim_err
                prim_err = ap
            end
            if ads > prim_err
                prim_err = ads
            end
        end

        ws.iters = iter
        ws.gap = gap

        gap_tol = max(tol_T, mach_tol * (one(T) + abs(p_obj)))
        if gap <= gap_tol &&
           prim_err <= feas_tol * (one(T) + norm_y) &&
           dual_err <= feas_tol * (one(T) + norm_X)
            converged = true
            break
        end
    end

    return beta, converged
end

function _qreg_ipm!(beta::AbstractVector{T}, X::AbstractMatrix{T},
                    y::AbstractVector{T}, tau::Real, ws::_QRegWorkspace{T};
                    tol::Real=1e-10, maxit::Int=60) where {T<:AbstractFloat}
    return _qreg_ipm!(beta, X, y, T(tau), ws; tol=tol, maxit=maxit)
end

function _qreg_ipm!(beta::AbstractVector{T}, X::AbstractMatrix{T},
                    y::AbstractVector{T}, tau::Real;
                    tol::Real=1e-10, maxit::Int=60) where {T<:AbstractFloat}
    ws = _QRegWorkspace{T}(size(X, 1), size(X, 2))
    return _qreg_ipm!(beta, X, y, T(tau), ws; tol=tol, maxit=maxit)
end

"""
    _qreg_batch!(B_out, X, Y_mat, tau; threaded=true, tol=1e-10, maxit=60) → B_out

Solve `M = size(Y_mat, 2)` quantile regressions sharing a fixed `n × k` design matrix `X`
at quantile `tau`, storing column `j`'s coefficients in `B_out[:, j]`.

When `threaded=true` and `Threads.nthreads() > 1`, partitions `1:M` into contiguous chunks
and assigns one pre-allocated [`_QRegWorkspace`](@ref) per chunk.
"""
function _qreg_batch!(B_out::AbstractMatrix{T}, X::AbstractMatrix{T},
                      Y_mat::AbstractMatrix{T}, tau::Real;
                      threaded::Bool=true, tol::Real=1e-10,
                      maxit::Int=60) where {T<:AbstractFloat}
    n, k = size(X)
    M = size(Y_mat, 2)
    (n > 0 && k > 0) ||
        throw(ArgumentError("X dimensions must be positive, got ($n, $k)"))
    size(Y_mat, 1) == n ||
        throw(DimensionMismatch("Y_mat must have $n rows, got $(size(Y_mat, 1))"))
    size(B_out) == (k, M) ||
        throw(DimensionMismatch("B_out must have size ($k, $M), got $(size(B_out))"))
    tau_T = T(tau)
    zero(T) < tau_T < one(T) ||
        throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
    maxit >= 1 || throw(ArgumentError("maxit must be >= 1, got $maxit"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    M == 0 && return B_out

    if threaded && Threads.nthreads() > 1 && M > 1
        n_workers = min(M, Threads.nthreads())
        pool = [_QRegWorkspace{T}(n, k) for _ in 1:n_workers]
        base, rem = divrem(M, n_workers)
        Threads.@threads for c in 1:n_workers
            local ws_c = pool[c]
            start_j = (c - 1) * base + min(c - 1, rem) + 1
            len_j = base + (c <= rem ? 1 : 0)
            for j in start_j:(start_j + len_j - 1)
                _qreg_ipm!(@view(B_out[:, j]), X, @view(Y_mat[:, j]), tau_T, ws_c;
                           tol=tol, maxit=maxit)
            end
        end
    else
        ws_seq = _QRegWorkspace{T}(n, k)
        for j in 1:M
            _qreg_ipm!(@view(B_out[:, j]), X, @view(Y_mat[:, j]), tau_T, ws_seq;
                       tol=tol, maxit=maxit)
        end
    end
    return B_out
end

# =============================================================================
# Sparsity and covariance
# =============================================================================

"""
    _hall_sheather_bandwidth(n, tau, alpha) → T

Hall & Sheather (1988) bandwidth for the sparsity estimate, the `qreg` default:
`h = n^{-1/3} z_α^{2/3} [1.5 φ(z_τ)² / (2 z_τ² + 1)]^{1/3}`.
"""
function _hall_sheather_bandwidth(n::Int, tau::T, alpha::T=T(0.05)) where {T<:AbstractFloat}
    nd = Distributions.Normal()
    z_a = Distributions.quantile(nd, one(T) - alpha / 2)
    z_t = Distributions.quantile(nd, tau)
    phi = Distributions.pdf(nd, z_t)
    h = T(n)^(-one(T) / 3) * z_a^(T(2) / 3) *
        (T(1.5) * phi^2 / (2 * z_t^2 + one(T)))^(one(T) / 3)
    return T(clamp(h, T(1e-6), min(tau, one(T) - tau) / 2))
end

"""
    _qreg_sparsity(resid, tau, h) → T

Siddiqui-Hendricks-Koenker sparsity estimate `s(τ) = [F⁻¹(τ+h) − F⁻¹(τ−h)] / (2h)`, the
reciprocal of the residual density at the quantile. The asymptotic variance of a regression
quantile is proportional to `s(τ)²`, so this is what sets the scale of the standard errors.
"""
function _qreg_sparsity(resid::AbstractVector{T}, tau::T, h::Real) where {T<:AbstractFloat}
    h_T = T(h)
    lo = clamp(tau - h_T, T(1e-8), one(T) - T(1e-8))
    hi = clamp(tau + h_T, T(1e-8), one(T) - T(1e-8))
    s = (Statistics.quantile(resid, hi) - Statistics.quantile(resid, lo)) / (hi - lo)
    return max(T(s), T(1e-8))
end

"""
    _qreg_vcov(X, resid, tau, se; alpha) → Matrix{T}

Asymptotic covariance of the regression quantile.

- `:iid` — Koenker's sparsity form `τ(1−τ) s(τ)² (X'X)⁻¹`, valid when the conditional
  density of the error at the quantile does not vary with `x`. This is the `qreg` default.
- `:robust` — Powell's kernel sandwich
  `τ(1−τ) (X'DX)⁻¹ (X'X) (X'DX)⁻¹` with `D = diag(K(u_i/h)/h)`, which allows the density at
  the quantile to depend on `x` (the analogue of moving from OLS to HC).
"""
function _qreg_vcov(X::Matrix{T}, resid::Vector{T}, tau::T, se::Symbol;
                    alpha::T=T(0.05)) where {T<:AbstractFloat}
    n, k = size(X)
    h = _hall_sheather_bandwidth(n, tau, alpha)
    XtXinv = Matrix{T}(robust_inv(Symmetric(X' * X)))

    if se === :iid
        s = _qreg_sparsity(resid, tau, h)
        return tau * (one(T) - tau) * s^2 .* XtXinv
    end

    # Powell sandwich: a Gaussian kernel density of the residuals at zero.
    bw = max(h * Statistics.std(resid), T(1e-8))
    D = similar(resid)
    @inbounds for i in eachindex(resid)
        D[i] = exp(-T(0.5) * (resid[i] / bw)^2) / (bw * sqrt(T(2π)))
    end
    XDX = Matrix{T}((X .* D)' * X)
    XDXinv = Matrix{T}(robust_inv(Symmetric(XDX)))
    return tau * (one(T) - tau) .* (XDXinv * (X' * X) * XDXinv)
end

# =============================================================================
# Type
# =============================================================================

"""
    QuantileRegModel{T} <: StatsAPI.RegressionModel

Quantile regression fit (Koenker & Bassett 1978) at one or several quantiles.

Every matrix field is stored with **one column per quantile**, in the order the quantiles
were supplied.

Fields:
- `y`, `X` — data
- `taus::Vector{T}` — the quantiles fitted
- `beta::Matrix{T}` — `k × n_tau` coefficients
- `vcov_mats::Vector{Matrix{T}}` — covariance per quantile
- `stderr::Matrix{T}` — `k × n_tau` standard errors
- `residuals`, `fitted` — `n × n_tau`
- `objective::Vector{T}` — check-function loss at the optimum, per quantile
- `pseudo_r2::Vector{T}` — Koenker-Machado `R¹(τ) = 1 − V̂(τ)/Ṽ(τ)`, the loss relative to
  the intercept-only fit. It is NOT comparable to an OLS `R²`.
- `varnames`, `se_type`, `n_obs`, `converged`
"""
struct QuantileRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    taus::Vector{T}
    beta::Matrix{T}
    vcov_mats::Vector{Matrix{T}}
    stderr::Matrix{T}
    residuals::Matrix{T}
    fitted::Matrix{T}
    objective::Vector{T}
    pseudo_r2::Vector{T}
    varnames::Vector{String}
    se_type::Symbol
    n_obs::Int
    converged::Vector{Bool}
    alpha::T
    manifest::Union{ReproManifest,Nothing}
end

QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                    pseudo_r2, varnames, se_type, n_obs, converged;
                    alpha::Real=1 - DEFAULT_CI_MICRO, manifest=nothing) where {T<:AbstractFloat} =
    QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                        pseudo_r2, varnames, se_type, n_obs, converged, T(alpha), manifest)

QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                    pseudo_r2, varnames, se_type, n_obs, converged,
                    manifest::Union{ReproManifest,Nothing}) where {T<:AbstractFloat} =
    QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                        pseudo_r2, varnames, se_type, n_obs, converged,
                        T(1 - DEFAULT_CI_MICRO), manifest)

QuantileRegModel(y::Vector{T}, X::Matrix{T}, taus::Vector{T}, beta::Matrix{T},
                 vcov_mats::Vector{Matrix{T}}, stderr::Matrix{T}, residuals::Matrix{T},
                 fitted::Matrix{T}, objective::Vector{T}, pseudo_r2::Vector{T},
                 varnames::Vector{String}, se_type::Symbol, n_obs::Int,
                 converged::Vector{Bool};
                 alpha::Real=1 - DEFAULT_CI_MICRO,
                 manifest::Union{ReproManifest,Nothing}=nothing) where {T<:AbstractFloat} =
    QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                        pseudo_r2, varnames, se_type, n_obs, converged, T(alpha), manifest)

QuantileRegModel(y::Vector{T}, X::Matrix{T}, taus::Vector{T}, beta::Matrix{T},
                 vcov_mats::Vector{Matrix{T}}, stderr::Matrix{T}, residuals::Matrix{T},
                 fitted::Matrix{T}, objective::Vector{T}, pseudo_r2::Vector{T},
                 varnames::Vector{String}, se_type::Symbol, n_obs::Int,
                 converged::Vector{Bool},
                 manifest::Union{ReproManifest,Nothing}) where {T<:AbstractFloat} =
    QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                        pseudo_r2, varnames, se_type, n_obs, converged,
                        T(1 - DEFAULT_CI_MICRO), manifest)

# =============================================================================
# Estimation
# =============================================================================

"""
    estimate_qreg(y, X, tau=0.5; se=:iid, solver=:ipm, varnames=nothing, n_boot=500,
                  seed=nothing, rng=Random.default_rng(), alpha=0.10) → QuantileRegModel

Quantile regression (Koenker & Bassett 1978): estimate `β(τ)` by minimizing the
check-function loss

```math
\\min_\\beta \\sum_i \\rho_\\tau(y_i - x_i'\\beta), \\qquad
\\rho_\\tau(u) = u\\,(\\tau - \\mathbf{1}\\{u < 0\\})
```

Unlike OLS, which fits the conditional **mean**, this fits the conditional `τ`-quantile — so
a covariate is allowed to shift the lower tail of the outcome distribution differently from
the upper tail. `tau` may be a scalar or a vector; one fit is returned per quantile.

# Arguments
- `y::AbstractVector` — outcome (length `n`)
- `X::AbstractMatrix` — `n × k` design matrix (include your own intercept column)
- `tau` — quantile(s) in `(0, 1)`; default `0.5` (median regression, i.e. LAD)

# Keyword Arguments
- `se::Symbol = :iid` — `:iid` (Koenker sparsity, the `qreg` default), `:robust`
  (Powell kernel sandwich), or `:boot` (xy-pair bootstrap)
- `solver::Symbol = :ipm` — `:ipm` (Portnoy & Koenker 1997 Frisch–Newton primal–dual
  interior-point method) or `:irls` (annealed iteratively reweighted least squares)
- `n_boot::Int = 500` — bootstrap replications when `se = :boot`
- `seed::Union{Integer,Nothing} = nothing` — random seed for reproducible bootstrap draws
- `rng` — random number generator for the bootstrap
- `alpha::Real = 0.10` — significance level (`1 - conf_level`, default `0.10` for 90% CIs)
  used for the Hall-Sheather bandwidth, `Tables.jl` / `DataFrame(m)`, `confint(m)`, and `report(m)`
- `varnames` — coefficient names

# Returns
A [`QuantileRegModel`](@ref). `report(m)` prints one coefficient table per quantile.

# Examples
```julia
m = estimate_qreg(y, X, [0.1, 0.5, 0.9])
report(m)
coef(m)              # k × 3
```

See also [`estimate_reg`](@ref).

# References
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
- Portnoy, S., & Koenker, R. (1997). The Gaussian Hare and the Laplacian Tortoise:
  Computability of Squared-Error versus Absolute-Error Estimators. *Statistical Science*,
  12(4), 279–300.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
"""
function estimate_qreg(y::AbstractVector{T}, X::AbstractMatrix{T},
                       tau::Union{Real,AbstractVector}=T(0.5);
                       se::Symbol=:iid,
                       solver::Symbol=:ipm,
                       varnames::Union{Nothing,Vector{String}}=nothing,
                       n_boot::Int=500,
                       seed::Union{Integer,Nothing}=nothing,
                       rng=Random.default_rng(),
                       alpha::Real=1 - DEFAULT_CI_MICRO) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    _validate_conf_level(1 - (alpha isa Bool ? alpha : Float64(alpha)))
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    se in (:iid, :robust, :boot) ||
        throw(ArgumentError("se must be :iid, :robust, or :boot; got :$se"))
    solver in (:ipm, :irls) ||
        throw(ArgumentError("solver must be :ipm or :irls; got :$solver"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive, got $n_boot"))

    taus = tau isa Real ? T[T(tau)] : Vector{T}(collect(T, tau))
    isempty(taus) && throw(ArgumentError("tau must not be empty"))
    all(t -> zero(T) < t < one(T), taus) ||
        throw(ArgumentError("every tau must lie strictly in (0, 1); got $taus"))

    Xm = Matrix{T}(X)
    yv = Vector{T}(y)
    nt = length(taus)

    beta = zeros(T, k, nt)
    stderr = zeros(T, k, nt)
    vcovs = Vector{Matrix{T}}(undef, nt)
    resid = zeros(T, n, nt)
    fit = zeros(T, n, nt)
    obj = zeros(T, nt)
    pr2 = zeros(T, nt)
    conv = Vector{Bool}(undef, nt)

    ws = solver === :ipm ? _QRegWorkspace{T}(n, k) : nothing

    for (j, t) in enumerate(taus)
        b, ok = if solver === :ipm
            bj = Vector{T}(undef, k)
            _, conv_j = _qreg_ipm!(bj, Xm, yv, t, ws)
            bj, conv_j
        else
            _qreg_fit(Xm, yv, t)
        end
        beta[:, j] = b
        conv[j] = ok
        f = Xm * b
        r = yv .- f
        fit[:, j] = f
        resid[:, j] = r
        obj[j] = _check_loss(r, t)

        # Koenker-Machado goodness of fit: the loss relative to the intercept-only model,
        # whose fitted value is the unconditional tau-quantile.
        r0 = yv .- Statistics.quantile(yv, t)
        v0 = _check_loss(r0, t)
        pr2[j] = v0 > zero(T) ? one(T) - obj[j] / v0 : zero(T)

        V = if se === :boot
            _qreg_boot_vcov(Xm, yv, t, n_boot, rng; solver=solver)
        else
            _qreg_vcov(Xm, r, t, se; alpha=T(alpha))
        end
        vcovs[j] = V
        stderr[:, j] = sqrt.(max.(diag(V), zero(T)))
    end

    names = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    length(names) == k || throw(ArgumentError("varnames must have length $k"))

    result = QuantileRegModel{T}(yv, Xm, taus, beta, vcovs, stderr, resid, fit,
                                 obj, pr2, names, se, n, conv; alpha=T(alpha))
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_boot" => n_boot, "se" => String(se),
                                  "solver" => String(solver),
                                  "alpha" => Float64(alpha))))
end

function estimate_qreg(y::AbstractVector, X::AbstractMatrix,
                       tau::Union{Real,AbstractVector}=0.5; kwargs...)
    estimate_qreg(Float64.(y), Float64.(X), tau; kwargs...)
end

"""
    _qreg_boot_vcov(X, y, tau, n_boot, rng; solver=:ipm) → Matrix{T}

xy-pair (design) bootstrap covariance: resample observations with replacement and refit.
The pair bootstrap is the standard choice for quantile regression because it makes no
assumption about the conditional density at the quantile — which is exactly the object the
asymptotic formulas have to estimate.
"""
function _qreg_boot_vcov(X::Matrix{T}, y::Vector{T}, tau::T, n_boot::Int, rng;
                         solver::Symbol=:ipm) where {T<:AbstractFloat}
    n, k = size(X)
    draws = zeros(T, k, n_boot)
    idx = Vector{Int}(undef, n)
    kept = 0
    if solver === :ipm
        ws = _QRegWorkspace{T}(n, k)
        Xb = Matrix{T}(undef, n, k)
        yb = Vector{T}(undef, n)
        bb = Vector{T}(undef, k)
        for _ in 1:n_boot
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end
            @inbounds for col in 1:k, row in 1:n
                Xb[row, col] = X[idx[row], col]
            end
            rank(Xb) < k && continue             # degenerate resample; skip
            @inbounds for row in 1:n
                yb[row] = y[idx[row]]
            end
            _qreg_ipm!(bb, Xb, yb, tau, ws)
            kept += 1
            draws[:, kept] = bb
        end
    else
        for _ in 1:n_boot
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end
            Xb = X[idx, :]
            rank(Xb) < k && continue             # degenerate resample; skip
            bb, _ = _qreg_fit(Xb, y[idx], tau)
            kept += 1
            draws[:, kept] = bb
        end
    end
    kept >= 2 || return fill(T(NaN), k, k)
    D = @view draws[:, 1:kept]
    mu = vec(Statistics.mean(D; dims=2))
    C = zeros(T, k, k)
    for bnum in 1:kept
        d = @view D[:, bnum]
        C .+= (d .- mu) * (d .- mu)'
    end
    return C ./ T(kept - 1)
end

# =============================================================================
# StatsAPI interface — `fitted` has no default method, so define it explicitly
# =============================================================================

StatsAPI.coef(m::QuantileRegModel) = size(m.beta, 2) == 1 ? vec(m.beta) : m.beta
StatsAPI.fitted(m::QuantileRegModel) = size(m.fitted, 2) == 1 ? vec(m.fitted) : m.fitted
StatsAPI.residuals(m::QuantileRegModel) = size(m.residuals, 2) == 1 ? vec(m.residuals) : m.residuals
StatsAPI.nobs(m::QuantileRegModel) = m.n_obs
StatsAPI.vcov(m::QuantileRegModel) = length(m.vcov_mats) == 1 ? m.vcov_mats[1] : m.vcov_mats
StatsAPI.stderror(m::QuantileRegModel) = size(m.stderr, 2) == 1 ? vec(m.stderr) : m.stderr
StatsAPI.dof(m::QuantileRegModel) = size(m.X, 2)
StatsAPI.dof_residual(m::QuantileRegModel) = max(m.n_obs - size(m.X, 2), 1)

"""
    StatsAPI.confint(m::QuantileRegModel; level=1 - m.alpha) -> Matrix{T}

Confidence intervals (`(k * n_tau) × 2` matrix `[lower upper]`, stacked column-major
across quantiles matching `vec(m.beta)` and `DataFrame(m)`) using the `t(n - k)` distribution.
Defaults to `1 - m.alpha` (`0.90` under default `alpha = 0.10`).
"""
function StatsAPI.confint(m::QuantileRegModel{T};
                          level::Real=default_ci_level(m)) where {T}
    lv = _validate_conf_level(level)
    dof_r = max(size(m.X, 1) - size(m.X, 2), 1)
    crit = T(quantile(TDist(dof_r), 1 - (1 - lv) / 2))
    b = vec(m.beta)
    s = vec(m.stderr)
    return hcat(b .- crit .* s, b .+ crit .* s)
end

"""
    predict(m::QuantileRegModel, X_new) → Vector or Matrix

Predicted conditional quantiles at `X_new`: one column per fitted quantile.
"""
function StatsAPI.predict(m::QuantileRegModel{T}, X_new::AbstractMatrix) where {T}
    size(X_new, 2) == size(m.X, 2) ||
        throw(ArgumentError("X_new must have $(size(m.X, 2)) columns"))
    P = Matrix{T}(X_new) * m.beta
    return size(P, 2) == 1 ? vec(P) : P
end
StatsAPI.predict(m::QuantileRegModel) = StatsAPI.fitted(m)

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::QuantileRegModel{T}) where {T}
    ts = join(string.(round.(m.taus; digits=3)), ", ")
    print(io, "QuantileRegModel{$T}: n=$(m.n_obs), k=$(size(m.X, 2)), tau=[$ts], se=:$(m.se_type)")
end

"""
    report(m::QuantileRegModel)

Print one Stata-style coefficient table per quantile, plus the check-function loss and the
Koenker-Machado pseudo-`R¹`.
"""
function report(io::IO, m::QuantileRegModel{T}) where {T}
    n, k = size(m.X)
    se_label = m.se_type === :iid ? "sparsity (iid)" :
               m.se_type === :robust ? "Powell kernel (robust)" : "xy-pair bootstrap"
    _show_spec_table(io, "Quantile Regression (Koenker-Bassett)",
        ["Observations" => string(n), "Coefficients" => string(k),
         "Quantiles" => join(string.(round.(m.taus; digits=3)), ", "),
         "Std. errors" => se_label])
    for (j, t) in enumerate(m.taus)
        m.converged[j] || @warn "Quantile regression at tau = $t did not converge." maxlog = 1
        _coef_table(io, "tau = $(round(t; digits=3))  (loss = $(_fmt(m.objective[j]; digits=4)), " *
                        "pseudo R1 = $(_fmt(m.pseudo_r2[j]; digits=4)))",
                    m.varnames, Vector{T}(m.beta[:, j]), Vector{T}(m.stderr[:, j]);
                    dist=:t, dof_r=n - k, level=default_ci_level(m))
    end
    return nothing
end
report(m::QuantileRegModel) = report(stdout, m)
