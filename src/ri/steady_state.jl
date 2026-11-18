# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Steady-state LQG Rational Inattention Kalman filter solvers (Sims 2003;
Afrouzi & Yang 2021; Miao, Wu & Young 2022).

Implements dual fixed-point solvers:
- `:water_filling`: exact whitened water-filling operator with Anderson acceleration.
- `:logdet_barrier`: Cholesky precision-factorization barrier solved via `Optim.LBFGS()`
  and `ForwardDiff.gradient`, wrapped in an outer envelope fixed-point loop.

Supports both marginal information cost (`omega`) and Shannon mutual-information
capacity (`capacity` in `:bits` or `:nats`) via bracketed Brent root-finding.
"""

# =============================================================================
# Internal Linear Algebra & Packing Helpers
# =============================================================================

@inline _ri_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

function _ri_project_psd(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    Ms = _ri_symmetrize(M)
    F = eigen(Symmetric(Ms))
    min_ev = minimum(F.values)
    if min_ev >= zero(T)
        return Ms, true
    end
    scale = max(maximum(F.values), one(T))
    ok = min_ev >= -T(1e-10) * scale
    vals = max.(F.values, zero(T))
    R = F.vectors * Diagonal(vals) * F.vectors'
    return _ri_symmetrize(R), ok
end

function _ri_pack_pair(S::Matrix{T}, X::Matrix{T}) where {T<:AbstractFloat}
    n = size(S, 1)
    m = div(n * (n + 1), 2)
    v = Vector{T}(undef, 2 * m)
    s2 = sqrt(T(2))
    idx = 0
    @inbounds for j in 1:n, i in j:n
        idx += 1
        w = (i == j) ? one(T) : s2
        v[idx] = w * S[i, j]
        v[m + idx] = w * X[i, j]
    end
    return v
end

function _ri_unpack_pair(v::Vector{T}, n::Int) where {T<:AbstractFloat}
    m = div(n * (n + 1), 2)
    S = Matrix{T}(undef, n, n)
    X = Matrix{T}(undef, n, n)
    inv_s2 = one(T) / sqrt(T(2))
    idx = 0
    @inbounds for j in 1:n, i in j:n
        idx += 1
        if i == j
            S[i, i] = v[idx]
            X[i, i] = v[m + idx]
        else
            vs = inv_s2 * v[idx]
            vx = inv_s2 * v[m + idx]
            S[i, j] = vs
            S[j, i] = vs
            X[i, j] = vx
            X[j, i] = vx
        end
    end
    return S, X
end

function _ri_pack_tril(L::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    v = Vector{T}(undef, div(n * (n + 1), 2))
    idx = 0
    @inbounds for j in 1:n, i in j:n
        idx += 1
        v[idx] = L[i, j]
    end
    return v
end

function _ri_unpack_tril(theta::AbstractVector{E}, n::Int) where {E}
    L = zeros(E, n, n)
    idx = 0
    @inbounds for j in 1:n, i in j:n
        idx += 1
        L[i, j] = theta[idx]
    end
    return L
end

function _ri_canonicalize_eigvecs!(V::Matrix{T}) where {T<:AbstractFloat}
    nrows, ncols = size(V)
    for j in 1:ncols
        pivot = 0
        for i in 1:nrows
            if abs(V[i, j]) > T(1e-8)
                pivot = i
                break
            end
        end
        if pivot == 0
            pivot = argmax(abs, @view V[:, j])
        end
        if V[pivot, j] < zero(T)
            @views V[:, j] .*= -one(T)
        end
    end
    return V
end

function _ri_whitened_modes(Sp::Matrix{T}, Oeff::Matrix{T}) where {T<:AbstractFloat}
    n = size(Sp, 1)
    Fp = eigen(Symmetric(Sp))
    ep = max.(Fp.values, zero(T))
    S_half = _ri_symmetrize(Fp.vectors * Diagonal(sqrt.(ep)) * Fp.vectors')
    M_w = _ri_symmetrize(S_half * Oeff * S_half)
    Fe = eigen(Symmetric(M_w))
    # Sort eigenvalues descending
    perm = n:-1:1
    d = Fe.values[perm]
    U = Matrix{T}(Fe.vectors[:, perm])
    return S_half, U, d, ep
end

# =============================================================================
# Single-Step Inner Operators (:water_filling and :logdet_barrier)
# =============================================================================

function _ri_wf_step(Sp::Matrix{T}, Oeff::Matrix{T}, omega::T) where {T<:AbstractFloat}
    n = size(Sp, 1)
    S_half, U, d, _ = _ri_whitened_modes(Sp, Oeff)
    Spost_new = copy(Sp)
    Xi_new = copy(Oeff)
    Inats = zero(T)
    half_om = omega / T(2)

    for i in 1:n
        di = d[i]
        if T(2) * di > omega
            z = S_half * @view U[:, i]
            Spost_new .-= (one(T) - omega / (T(2) * di)) .* (z * z')
            v = (Oeff * z) / di
            Xi_new .-= (di - half_om) .* (v * v')
            Inats += T(0.5) * log(T(2) * di / omega)
        end
    end

    Spost_psd, _ = _ri_project_psd(Spost_new)
    Xi_psd, _ = _ri_project_psd(Xi_new)
    return Spost_psd, Xi_psd, Inats
end

function _ri_lb_step(
    Sp::Matrix{T},
    Oeff::Matrix{T},
    omega::T,
    L_init::Union{Nothing,Matrix{T}},
    g_tol::T,
) where {T<:AbstractFloat}
    n = size(Sp, 1)
    s_scale = sqrt(max(tr(Sp) / n, T(1e-30)))
    o_scale = max(tr(Oeff * Sp) / n, omega)
    Sp_hat = _ri_symmetrize(Sp / (s_scale * s_scale))
    Oeff_hat = _ri_symmetrize(Oeff * (s_scale * s_scale) / o_scale)
    omega_hat = omega / o_scale
    base_loss_hat = tr(Oeff_hat * Sp_hat)
    tie = T(1e-16)

    function f_barrier(theta::AbstractVector{E}) where {E}
        L = _ri_unpack_tril(theta, n)
        P = Sp_hat * L
        W = L' * P
        @inbounds for i in 1:n
            W[i, i] += one(E)
        end
        W_sym = (W + W') / 2
        Cw_L = safe_cholesky(Symmetric(W_sym); silent=true)
        logdetW = T(2) * sum(i -> log(Cw_L[i, i]), 1:n)
        Z = Cw_L \ P'
        return base_loss_hat - sum(Z .* (Z * Oeff_hat)) +
               (omega_hat / T(2)) * logdetW + tie * sum(abs2, theta)
    end

    L0_hat = L_init === nothing ? (T(0.5) * Matrix{T}(I, n, n)) : (L_init * s_scale)
    th0 = _ri_pack_tril(L0_hat)
    cfg = ForwardDiff.GradientConfig(f_barrier, th0)
    g! = (G, th) -> ForwardDiff.gradient!(G, f_barrier, th, cfg)
    alg = Optim.LBFGS(m=min(max(12, 2 * n), 30))
    opts = Optim.Options(g_tol=g_tol, iterations=600, show_trace=false)

    res = Optim.optimize(f_barrier, g!, th0, alg, opts)

    # Global zero-attention check: if L = 0 beats or ties optimizer minimum, snap to exact zero
    if Optim.minimum(res) >= base_loss_hat - T(1e-15)
        zeros_mat = zeros(T, n, n)
        return copy(Sp), copy(Oeff), zero(T), zeros_mat, zeros_mat
    end

    Lstar = _ri_unpack_tril(Optim.minimizer(res), n) / s_scale
    Wstar = Lstar' * Sp * Lstar
    @inbounds for i in 1:n
        Wstar[i, i] += one(T)
    end
    Wsym = _ri_symmetrize(Wstar)
    Cw_L = safe_cholesky(Symmetric(Wsym); silent=true)
    Z = Cw_L \ (Lstar' * Sp)
    Spost_lb, _ = _ri_project_psd(Sp - Z' * Z)

    # KKT global optimality check to escape any rank-deficient saddle from warm-start
    Fp_s = eigen(Symmetric(Spost_lb))
    Sh_post = _ri_symmetrize(
        Fp_s.vectors * Diagonal(sqrt.(max.(Fp_s.values, zero(T)))) * Fp_s.vectors')
    lam_max = eigmax(Symmetric(_ri_symmetrize(Sh_post * Oeff * Sh_post)))
    if lam_max > (omega / T(2)) * (one(T) + T(1e-7))
        th_fresh = _ri_pack_tril(Lstar * s_scale + T(0.25) * Matrix{T}(I, n, n))
        res2 = Optim.optimize(f_barrier, g!, th_fresh, alg, opts)
        if Optim.minimum(res2) < Optim.minimum(res)
            Lstar = _ri_unpack_tril(Optim.minimizer(res2), n) / s_scale
            Wstar = Lstar' * Sp * Lstar
            @inbounds for i in 1:n
                Wstar[i, i] += one(T)
            end
            Wsym = _ri_symmetrize(Wstar)
            Cw_L = safe_cholesky(Symmetric(Wsym); silent=true)
            Z = Cw_L \ (Lstar' * Sp)
            Spost_lb, _ = _ri_project_psd(Sp - Z' * Z)
        end
    end

    # Envelope-consistent factor (exact even when Σ_prior is singular/ill-conditioned)
    L_env = (T(2) / omega) * Oeff * Spost_lb * Lstar
    Delta_lb = _ri_symmetrize(L_env * L_env')
    Xi_lb, _ = _ri_project_psd(Oeff - (omega / T(2)) * Delta_lb)
    Inats = sum(i -> log(Cw_L[i, i]), 1:n)
    return Spost_lb, Xi_lb, Inats, Lstar, L_env
end

# =============================================================================
# Fixed-ω Steady-State Fixed-Point Solver
# =============================================================================

function _ri_solve_fixed_omega(
    A::Matrix{T},
    B::Matrix{T},
    Q::Matrix{T},
    Obase::Matrix{T},
    beta::T,
    omega::T,
    method::Symbol,
    tol::T,
    max_iter::Int;
    Sigma_uncond::Union{Nothing,Matrix{T}}=nothing,
    Xi_zero::Union{Nothing,Matrix{T}}=nothing,
    warm_state::Union{Nothing,Tuple{Matrix{T},Matrix{T},Union{Nothing,Matrix{T}}}}=nothing,
) where {T<:AbstractFloat}
    n = size(A, 1)
    rho_A = maximum(abs, eigvals(A); init=zero(T))
    is_stable = rho_A < one(T)

    if is_stable
        Su = Sigma_uncond === nothing ? solve_lyapunov(A, B) : Sigma_uncond
        Xz = Xi_zero === nothing ?
            _dlyap(sqrt(beta) * Matrix{T}(A'), Obase; tol=T(1e-13), warn_label="solve_ri_kalman") :
            Xi_zero
        Su_sym = _ri_symmetrize(Su)
        Xz_sym = _ri_symmetrize(Xz)

        # Zero-uncertainty or zero-attention fast path
        if norm(Su_sym) <= T(1e-30) || norm(Obase) <= T(1e-30)
            return (
                Sigma_prior = Su_sym,
                Sigma_post = copy(Su_sym),
                Xi = Xz_sym,
                Omega_eff = copy(Xz_sym),
                Inats = zero(T),
                L_star = zeros(T, n, n),
                L_env = zeros(T, n, n),
                converged = true,
                iterations = 1,
            )
        end

        _, _, d0, _ = _ri_whitened_modes(Su_sym, Xz_sym)
        if isempty(d0) || T(2) * d0[1] <= omega * (one(T) + T(1e-9))
            return (
                Sigma_prior = Su_sym,
                Sigma_post = copy(Su_sym),
                Xi = Xz_sym,
                Omega_eff = copy(Xz_sym),
                Inats = zero(T),
                L_star = zeros(T, n, n),
                L_env = zeros(T, n, n),
                converged = true,
                iterations = 1,
            )
        end
    end

    # Initialize (Σ_post, Ξ)
    Spost, Xi, L_prev = if warm_state !== nothing
        copy(warm_state[1]), copy(warm_state[2]),
        warm_state[3] === nothing ? nothing : copy(warm_state[3])
    elseif is_stable
        copy(Sigma_uncond === nothing ? solve_lyapunov(A, B) : Sigma_uncond),
        copy(Xi_zero === nothing ?
            _dlyap(sqrt(beta) * Matrix{T}(A'), Obase; tol=T(1e-13), warn_label="solve_ri_kalman") :
            Xi_zero),
        nothing
    else
        max(norm(Q), one(T)) * Matrix{T}(I, n, n), copy(Obase), nothing
    end

    hist = Vector{T}[]
    resids = Vector{T}[]
    damp = T(0.6)
    g_tol_lb = min(tol * T(0.01), T(1e-13))

    safe_x = nothing
    safe_g = nothing
    safe_L = nothing
    safe_rnorm = T(Inf)

    S_limit = T(1e18) * max(norm(Q), one(T))

    for iter in 1:max_iter
        Sp = _ri_symmetrize(A * Spost * A' + Q)
        Oeff = _ri_symmetrize(Obase + beta * (A' * Xi * A))

        if method === :water_filling
            Sn, Xn, Inats = _ri_wf_step(Sp, Oeff, omega)
            Lstar = zeros(T, n, n)
            L_env = zeros(T, n, n)
        else
            Sn, Xn, Inats, Lstar, L_env = _ri_lb_step(Sp, Oeff, omega, L_prev, g_tol_lb)
            L_prev = Lstar
        end

        if !all(isfinite, Sn) || !all(isfinite, Xn) || norm(Sn) > S_limit
            throw(ArgumentError(
                "RI steady-state covariance diverged (unstable mode may be undetectable through H or ω too large to stabilize)."))
        end

        scale_s = max(norm(Sn), norm(Sp), norm(Q), T(1e-30))
        scale_x = max(norm(Xn), norm(Oeff), norm(Obase), T(1e-30))
        rel_err = max(norm(Sn - Spost) / scale_s, norm(Xn - Xi) / scale_x)

        if rel_err < tol
            Sp_final = _ri_symmetrize(A * Sn * A' + Q)
            Oeff_final = _ri_symmetrize(Obase + beta * (A' * Xn * A))
            return (
                Sigma_prior = Sp_final,
                Sigma_post = Sn,
                Xi = Xn,
                Omega_eff = Oeff_final,
                Inats = Inats,
                L_star = Lstar,
                L_env = L_env,
                converged = true,
                iterations = iter,
            )
        end

        xk = _ri_pack_pair(Spost, Xi)
        gk = _ri_pack_pair(Sn, Xn)
        rk = gk - xk
        rnorm = norm(rk)

        if method === :logdet_barrier && safe_x !== nothing && rnorm > T(2) * safe_rnorm
            # Revert Anderson overshoot without wasting an extra LBFGS call on accepted steps
            empty!(hist)
            empty!(resids)
            xd = (one(T) - damp) * safe_x + damp * safe_g
            Spost, Xi = _ri_unpack_pair(xd, n)
            L_prev = safe_L
            safe_x = nothing
            safe_rnorm = T(Inf)
            continue
        end

        safe_x = xk
        safe_g = gk
        safe_L = L_prev
        safe_rnorm = rnorm

        push!(hist, xk)
        push!(resids, rk)
        if length(hist) > 8
            popfirst!(hist)
            popfirst!(resids)
        end

        # For unstable systems with zero active attention so far, expand smoothly rather than extrapolating negatively
        allow_aa = length(hist) >= 2 && (is_stable || Inats > T(1e-12))
        if allow_aa
            xc = _anderson_step(hist, resids, 6)
            Sc_raw, Xc_raw = _ri_unpack_pair(xc, n)
            Sc, okS = _ri_project_psd(Sc_raw)
            Xc, okX = _ri_project_psd(Xc_raw)
            if okS && okX && norm(Sc) < S_limit
                if method === :water_filling
                    Spc = _ri_symmetrize(A * Sc * A' + Q)
                    Oeffc = _ri_symmetrize(Obase + beta * (A' * Xc * A))
                    Snc, Xnc, _ = _ri_wf_step(Spc, Oeffc, omega)
                    rc_norm = norm(_ri_pack_pair(Snc, Xnc) - _ri_pack_pair(Sc, Xc))
                    if isfinite(rc_norm) && rc_norm <= rnorm
                        Spost = Sc
                        Xi = Xc
                        continue
                    end
                else
                    Spost = Sc
                    Xi = Xc
                    continue
                end
            end
            empty!(hist)
            empty!(resids)
        end

        Spost = (one(T) - damp) * Spost + damp * Sn
        Xi = (one(T) - damp) * Xi + damp * Xn
    end

    Sp_last = _ri_symmetrize(A * Spost * A' + Q)
    Oeff_last = _ri_symmetrize(Obase + beta * (A' * Xi * A))
    if method === :water_filling
        Sn, Xn, Inats = _ri_wf_step(Sp_last, Oeff_last, omega)
        Lstar = zeros(T, n, n)
        L_env = zeros(T, n, n)
    else
        Sn, Xn, Inats, Lstar, L_env = _ri_lb_step(Sp_last, Oeff_last, omega, L_prev, g_tol_lb)
    end
    return (
        Sigma_prior = Sp_last,
        Sigma_post = Sn,
        Xi = Xn,
        Omega_eff = Oeff_last,
        Inats = Inats,
        L_star = Lstar,
        L_env = L_env,
        converged = false,
        iterations = max_iter,
    )
end

# =============================================================================
# Brent Root-Finder for Fixed-Capacity Solve
# =============================================================================

function _ri_brent_root(
    f,
    a::T,
    b::T,
    fa::T,
    fb::T;
    tol::T,
    max_eval::Int=100,
) where {T<:AbstractFloat}
    fa == zero(T) && return a, fa, 0
    fb == zero(T) && return b, fb, 0
    fa * fb < zero(T) || throw(ArgumentError(
        "Brent root-finder requires opposite signs at bracket endpoints"))

    c = a
    fc = fa
    d = b - a
    e = d
    eps_T = eps(T)
    evals = 0

    for _ in 1:max_eval
        if (fb > zero(T) && fc > zero(T)) || (fb < zero(T) && fc < zero(T))
            c = a
            fc = fa
            d = b - a
            e = d
        end
        if abs(fc) < abs(fb)
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        end

        tol_act = T(2) * eps_T * abs(b) + tol * max(abs(b), one(T)) / T(2)
        m = (c - b) / T(2)

        if abs(m) <= tol_act || abs(fb) <= tol * T(0.1)
            return b, fb, evals
        end

        if abs(e) >= tol_act && abs(fa) > abs(fb)
            s = fb / fa
            if a == c
                p = T(2) * m * s
                q = one(T) - s
            else
                q = fa / fc
                r = fb / fc
                p = s * (T(2) * m * q * (q - r) - (b - a) * (r - one(T)))
                q = (q - one(T)) * (r - one(T)) * (s - one(T))
            end
            if p > zero(T)
                q = -q
            else
                p = -p
            end
            s = e
            e = d
            if T(2) * p < min(T(3) * m * q - abs(tol_act * q), abs(s * q))
                d = p / q
            else
                d = m
                e = d
            end
        else
            d = m
            e = d
        end

        a = b
        fa = fb
        if abs(d) > tol_act
            b += d
        else
            b += m > zero(T) ? tol_act : -tol_act
        end
        fb = f(b)
        evals += 1
    end
    return b, fb, evals
end

# =============================================================================
# Endogenous Signal Extraction
# =============================================================================

function _ri_extract_signals(
    Sigma_prior::Matrix{T},
    Sigma_post::Matrix{T},
    Omega_eff::Matrix{T},
    omega::T,
    method::Symbol,
    L_env::Matrix{T},
) where {T<:AbstractFloat}
    n = size(Sigma_prior, 1)
    S_half, U, d, _ = _ri_whitened_modes(Sigma_prior, Omega_eff)
    act = findall(i -> T(2) * d[i] > omega * (one(T) + T(1e-9)), 1:n)
    r_s = length(act)

    if r_s == 0
        return (
            Sigma_post_out = copy(Sigma_prior),
            signal_rank = 0,
            Y_signal = zeros(T, n, 0),
            Sigma_psi = zeros(T, 0, 0),
            K_gain = zeros(T, n, 0),
            capacity_nats = zero(T),
            capacity_bits = zero(T),
        )
    end

    Delta_prec = if method === :logdet_barrier
        Symmetric(_ri_symmetrize(L_env * L_env'))
    else
        V_prec = Matrix{T}(undef, n, r_s)
        for (k, idx) in enumerate(act)
            z = S_half * @view U[:, idx]
            V_prec[:, k] .= (Omega_eff * z) ./ d[idx]
        end
        w_prec = (T(2) .* d[act] ./ omega) .- one(T)
        Symmetric(_ri_symmetrize(V_prec * Diagonal(w_prec) * V_prec'))
    end

    F_dp = eigen(Delta_prec)
    top_idx = n:-1:(n - r_s + 1)
    lam_p = max.(F_dp.values[top_idx], T(1e-30))
    V_s = Matrix{T}(F_dp.vectors[:, top_idx])
    _ri_canonicalize_eigvecs!(V_s)

    Sigma_psi = Matrix{T}(Diagonal(one(T) ./ lam_p))
    K_gain = Matrix{T}(Sigma_post * V_s * Diagonal(lam_p))
    capacity_nats = T(0.5) * sum(i -> log(T(2) * d[i] / omega), act)
    capacity_bits = capacity_nats / log(T(2))

    return (
        Sigma_post_out = Sigma_post,
        signal_rank = r_s,
        Y_signal = V_s,
        Sigma_psi = Sigma_psi,
        K_gain = K_gain,
        capacity_nats = capacity_nats,
        capacity_bits = capacity_bits,
    )
end

# =============================================================================
# Public API: solve_ri_kalman
# =============================================================================

"""
    solve_ri_kalman(prob::RationalInattentionProblem{T};
                    method::Symbol=:water_filling,
                    tol::Real=1e-11,
                    max_iter::Int=2000) where {T<:AbstractFloat} -> SimsKalmanSolution{T}

Solve a steady-state LQG Rational Inattention Kalman filter problem (Sims 2003;
Afrouzi & Yang 2021; Miao, Wu & Young 2022) for the optimal endogenous signal
structure `(Y_signal, Sigma_psi)`, steady-state Kalman gain `K_gain`, prior/posterior
covariances `(Sigma_prior, Sigma_post)`, shadow information cost `omega`, and
action policy `F_policy = H'`.

# Arguments
- `prob::RationalInattentionProblem{T}`: problem specification (either `capacity` or `omega`).
- `method::Symbol=:water_filling`: `:water_filling` (closed-form whitened water-filling
  with Anderson acceleration) or `:logdet_barrier` (Cholesky precision-factorization
  barrier optimized via `Optim.LBFGS()` + `ForwardDiff.gradient`).
- `tol::Real=1e-11`: convergence tolerance on relative fixed-point change and Brent root-finding.
- `max_iter::Int=2000`: maximum outer fixed-point iterations.
"""
function solve_ri_kalman(
    prob::RationalInattentionProblem{T};
    method::Symbol=:water_filling,
    tol::Real=1e-11,
    max_iter::Int=2000,
) where {T<:AbstractFloat}
    method in (:water_filling, :logdet_barrier) || throw(ArgumentError(
        "method must be :water_filling or :logdet_barrier, got :$method"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))

    # Promote Float16/Float32 internally to Float64 for numerical robustness, then cast back to T
    if T <: Union{Float16, Float32}
        return _solve_ri_kalman_impl(Float64, prob, method, max(Float64(tol), 10 * eps(Float64)), max_iter)
    else
        return _solve_ri_kalman_impl(T, prob, method, max(T(tol), T(10) * eps(T)), max_iter)
    end
end

function _ri_check_detectable(A::Matrix{T}, B::Matrix{T}, Obase::Matrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    vals = eigvals(A)
    scale_A = max(opnorm(A, Inf), one(T))
    scale_O = max(opnorm(Obase, Inf), one(T))
    scale_B = max(opnorm(B, Inf), one(T))
    tol_pbh = sqrt(eps(T)) * max(scale_A, scale_O)
    tol_eig = sqrt(eps(T)) * scale_A
    Ac = Matrix{Complex{T}}(A)
    Oc = Matrix{Complex{T}}(Obase)
    Bc = Matrix{Complex{T}}(B)
    Ic = Matrix{Complex{T}}(I, n, n)

    for lam in vals
        abs(lam) >= one(T) || continue
        M_pbh = vcat(lam * Ic - Ac, Oc)
        sv = svd(M_pbh)
        null_idx = findall(<=(tol_pbh), sv.S)
        isempty(null_idx) && continue
        svA = svd(lam * Ic - Ac)
        left_idx = findall(<=(tol_eig), svA.S)
        isempty(left_idx) && continue
        U_L = svA.U[:, left_idx]
        V_R = sv.V[:, null_idx]
        if opnorm(U_L' * V_R, Inf) > sqrt(eps(T)) && opnorm(U_L' * Bc, Inf) > sqrt(eps(T)) * scale_B
            throw(ArgumentError(
                "RI problem has an undetectable unstable mode at λ = $(lam) excited by shocks; " *
                "no finite steady-state covariance exists."))
        end
    end
    return nothing
end

function _solve_ri_kalman_impl(
    ::Type{TW},
    prob::RationalInattentionProblem{T},
    method::Symbol,
    tol_w::TW,
    max_iter::Int,
) where {T<:AbstractFloat, TW<:AbstractFloat}

    A = Matrix{TW}(prob.A)
    B = Matrix{TW}(prob.B)
    H = Matrix{TW}(prob.H)
    M = Matrix{TW}(prob.M)
    beta = TW(prob.beta)

    Obase = _ri_symmetrize(H * M * H')
    Q = _ri_symmetrize(B * B')
    F_policy = Matrix{TW}(H')
    n = size(A, 1)

    eigs_A = eigvals(A)
    rho_A = maximum(abs, eigs_A; init=zero(TW))
    is_stable = rho_A < one(TW)
    is_stable || _ri_check_detectable(A, B, Obase)

    Sigma_uncond = is_stable ? _ri_symmetrize(solve_lyapunov(A, B)) : nothing
    Xi_zero = is_stable ?
        _ri_symmetrize(_dlyap(sqrt(beta) * Matrix{TW}(A'), Obase;
                              tol=TW(1e-13), warn_label="solve_ri_kalman")) :
        nothing

    local omega_star::TW
    local res_fp
    total_iters = 0

    if prob.omega !== nothing
        omega_star = TW(prob.omega)
        res_fp = _ri_solve_fixed_omega(
            A, B, Q, Obase, beta, omega_star, method, tol_w, max_iter;
            Sigma_uncond=Sigma_uncond, Xi_zero=Xi_zero,
        )
        total_iters = res_fp.iterations
    else
        kappa_nats = prob.units === :bits ?
            TW(prob.capacity) * log(TW(2)) : TW(prob.capacity)

        if !is_stable
            min_rate = sum(ev -> log(max(abs(ev), one(TW))), eigs_A)
            kappa_nats <= min_rate && throw(ArgumentError(
                "Insufficient Shannon capacity κ = $(kappa_nats) nats to stabilize unstable " *
                "state transition (minimum required > $(min_rate) nats)."))
        end

        inner_tol = min(tol_w, TW(1e-12)) * TW(0.05)
        brent_tol = min(tol_w, TW(1e-12))
        warm_ref = Ref{Union{Nothing,Tuple{Matrix{TW},Matrix{TW},Union{Nothing,Matrix{TW}}}}}(nothing)
        last_res = Ref{Any}(nothing)
        last_omega = Ref{TW}(TW(NaN))

        function eval_excess_info(om::TW)
            r = _ri_solve_fixed_omega(
                A, B, Q, Obase, beta, om, method, inner_tol, max_iter;
                Sigma_uncond=Sigma_uncond, Xi_zero=Xi_zero, warm_state=warm_ref[],
            )
            if r.converged
                warm_ref[] = (r.Sigma_post, r.Xi, method === :logdet_barrier ? r.L_star : nothing)
            end
            last_res[] = r
            last_omega[] = om
            total_iters += 1
            return r.Inats - kappa_nats
        end

        local om_lo::TW
        local om_hi::TW
        local f_lo::TW
        local f_hi::TW

        if is_stable
            if norm(Sigma_uncond) <= TW(1e-30) || norm(Obase) <= TW(1e-30)
                throw(ArgumentError(
                    "Cannot allocate positive Shannon capacity κ > 0 when shock covariance Q " *
                    "or tracking payoff Ω_base is zero."))
            end
            _, _, d0, _ = _ri_whitened_modes(Sigma_uncond, Xi_zero)
            om_zero = isempty(d0) ? zero(TW) : TW(2) * d0[1]
            om_zero <= TW(1e-30) && throw(ArgumentError(
                "Zero-attention threshold ω_zero is zero; cannot allocate positive capacity κ > 0."))

            om_hi = om_zero
            f_hi = -kappa_nats
            om_lo = max(om_zero * exp(-TW(2) * kappa_nats / max(n, 1)) * TW(0.5),
                        om_zero * TW(1e-12))
            f_lo = eval_excess_info(om_lo)
            for _ in 1:60
                f_lo >= zero(TW) && break
                om_hi = om_lo
                f_hi = f_lo
                om_lo *= TW(0.25)
                f_lo = eval_excess_info(om_lo)
            end
            f_lo >= zero(TW) || throw(ArgumentError(
                "Failed to bracket Shannon capacity κ = $(kappa_nats) nats from below."))
        else
            om_try = max(tr(Obase * Q), norm(Obase) * max(norm(Q), one(TW)), one(TW))
            f_try = eval_excess_info(om_try)
            if f_try >= zero(TW)
                om_lo = om_try
                f_lo = f_try
                om_hi = om_try
                f_hi = f_try
                for _ in 1:60
                    om_hi *= TW(4)
                    f_hi = eval_excess_info(om_hi)
                    if f_hi <= zero(TW)
                        break
                    else
                        om_lo = om_hi
                        f_lo = f_hi
                    end
                end
                f_hi <= zero(TW) || throw(ArgumentError(
                    "Failed to bracket Shannon capacity κ = $(kappa_nats) nats from above."))
            else
                om_hi = om_try
                f_hi = f_try
                om_lo = om_try
                f_lo = f_try
                for _ in 1:60
                    om_lo *= TW(0.25)
                    f_lo = eval_excess_info(om_lo)
                    if f_lo >= zero(TW)
                        break
                    else
                        om_hi = om_lo
                        f_hi = f_lo
                    end
                end
                f_lo >= zero(TW) || throw(ArgumentError(
                    "Failed to bracket Shannon capacity κ = $(kappa_nats) nats from below."))
            end
        end

        omega_star, _, _ = _ri_brent_root(
            eval_excess_info, om_lo, om_hi, f_lo, f_hi;
            tol=brent_tol, max_eval=100,
        )
        # Polish final fixed point at converged ω*
        res_fp = _ri_solve_fixed_omega(
            A, B, Q, Obase, beta, omega_star, method, tol_w, max_iter;
            Sigma_uncond=Sigma_uncond, Xi_zero=Xi_zero, warm_state=warm_ref[],
        )
        total_iters = max(total_iters, res_fp.iterations)
    end

    sig = _ri_extract_signals(
        res_fp.Sigma_prior,
        res_fp.Sigma_post,
        res_fp.Omega_eff,
        omega_star,
        method,
        res_fp.L_env,
    )

    expected_loss = TW(tr(Obase * sig.Sigma_post_out))

    return SimsKalmanSolution{T}(
        prob,
        Matrix{T}(res_fp.Sigma_prior),
        Matrix{T}(sig.Sigma_post_out),
        Matrix{T}(res_fp.Omega_eff),
        T(omega_star),
        T(sig.capacity_bits),
        T(sig.capacity_nats),
        sig.signal_rank,
        Matrix{T}(sig.Y_signal),
        Matrix{T}(sig.Sigma_psi),
        Matrix{T}(sig.K_gain),
        Matrix{T}(F_policy),
        T(expected_loss),
        res_fp.converged,
        total_iters,
        method,
    )
end
