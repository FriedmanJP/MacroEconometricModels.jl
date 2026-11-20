# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Two-step semi-parametric 2SLS-GMM estimator for Strong Structural VARMA(p, q) models
(Gouriéroux, Monfort & Renne 2020, §4.2 & Appendices C.1–C.3).

Step 1 estimates reduced-form AR parameters ``(\\mu, \\Phi_1, \\ldots, \\Phi_p)`` by
2SLS using lagged instruments ``Y_{t-q-1}, \\ldots, Y_{t-q-k_{iv}}``.
Step 2 estimates ``(C_0, C_1, \\ldots, C_q, \\kappa_3, \\kappa_4)`` via cumulant GMM
across non-symmetric direction pairs with exact first-step-error-adjusted Newey–West
HAC optimal weighting ``\\Omega^* = Q_0^{-1}`` and asymptotic covariance ``\\Sigma(\\Omega^*)``.
"""

using LinearAlgebra, Statistics, Random
import ForwardDiff
import Optim

# =============================================================================
# Step 1: 2SLS Estimation of (μ, Φ₁, ..., Φ_p) (GMR 2020, App. C.1)
# =============================================================================

function _svarma_gmm_step1_2sls(Y::Matrix{T}, p::Int, q::Int, k_iv::Int) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    k_iv >= p || throw(ArgumentError("k_iv ($k_iv) must be >= p ($p)"))

    if p == 0
        mu_hat = vec(mean(Y; dims=1))
        Phi_3d = zeros(T, n, n, 0)
        Z_all = Y .- transpose(mu_hat)
        T1 = T_obs
        h1_mat = copy(Z_all)
        Q_pi = Matrix{T}(I, n, n)
        return (
            mu = mu_hat,
            Phi = Phi_3d,
            alpha = mu_hat,
            A_hat = reshape(mu_hat, 1, n),
            Z_all = Z_all,
            h1_mat = h1_mat,
            Q_pi = Q_pi,
            t_start = 1,
            T1 = T1,
        )
    end

    t_start = max(p, q + k_iv) + 1
    T1 = T_obs - t_start + 1
    n_reg = 1 + n * p
    n_inst = 1 + n * k_iv
    T1 > n_inst || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for p=$p, q=$q, k_iv=$k_iv " *
        "(need T_1 = $T1 > n_inst = $n_inst)."))

    Y1 = Matrix{T}(@view Y[t_start:T_obs, :])
    X1 = Matrix{T}(undef, T1, n_reg)
    W1 = Matrix{T}(undef, T1, n_inst)
    @inbounds for tau in 1:T1
        t = t_start + tau - 1
        X1[tau, 1] = one(T)
        W1[tau, 1] = one(T)
        for k in 1:p
            col0 = 2 + (k - 1) * n
            for j in 1:n
                X1[tau, col0 + j - 1] = Y[t - k, j]
            end
        end
        for l in 1:k_iv
            col0 = 2 + (l - 1) * n
            for j in 1:n
                W1[tau, col0 + j - 1] = Y[t - q - l, j]
            end
        end
    end

    S_WW = (transpose(W1) * W1) / T(T1)
    S_XW = (transpose(X1) * W1) / T(T1)
    S_WY = (transpose(W1) * Y1) / T(T1)
    S_WW_inv = Matrix{T}(robust_inv(S_WW; silent=true))
    G_XWX = S_XW * S_WW_inv * transpose(S_XW)
    P_1 = Matrix{T}(robust_inv(G_XWX; silent=true) * S_XW * S_WW_inv)

    A_hat = P_1 * S_WY
    alpha_hat = vec(A_hat)
    mu_hat = Vector{T}(@view A_hat[1, :])
    Phi_3d = Array{T,3}(undef, n, n, p)
    @inbounds for k in 1:p
        rows = (2 + (k - 1) * n):(1 + k * n)
        Phi_3d[:, :, k] .= transpose(@view A_hat[rows, :])
    end

    Z_all = _svarma_ar_residuals(Y, mu_hat, Phi_3d)
    Z1_res = Y1 .- X1 * A_hat
    h1_mat = Matrix{T}(undef, T1, n * n_inst)
    @inbounds for tau in 1:T1
        for i in 1:n
            off = (i - 1) * n_inst
            z_ti = Z1_res[tau, i]
            for m in 1:n_inst
                h1_mat[tau, off + m] = W1[tau, m] * z_ti
            end
        end
    end

    Q_pi = kron(Matrix{T}(I, n, n), P_1)

    return (
        mu = mu_hat,
        Phi = Phi_3d,
        alpha = alpha_hat,
        A_hat = A_hat,
        Z_all = Z_all,
        h1_mat = h1_mat,
        Q_pi = Q_pi,
        t_start = t_start,
        T1 = T1,
    )
end

function _svarma_gmm_residuals_from_alpha(Y::AbstractMatrix, alpha::AbstractVector{Tα},
                                          p::Int, n::Int) where {Tα<:Real}
    T_obs = size(Y, 1)
    T_eff = T_obs - p
    n_reg = 1 + n * p
    A = reshape(alpha, n_reg, n)
    Z = Matrix{Tα}(undef, T_eff, n)
    @inbounds for tau in 1:T_eff
        t = p + tau
        for i in 1:n
            val = Tα(Y[t, i]) - A[1, i]
            for k in 1:p
                row0 = 1 + (k - 1) * n
                for j in 1:n
                    val -= A[row0 + j, i] * Tα(Y[t - k, j])
                end
            end
            Z[tau, i] = val
        end
    end
    return Z
end

# =============================================================================
# Step 2: Direction Bank & Pairwise Cumulant Moments (GMR 2020, §4.2.1, App. C.2)
# =============================================================================

function _svarma_gmm_direction_bank(n::Int, q::Int, ::Type{T}) where {T<:AbstractFloat}
    dirs = Matrix{T}[]
    if q == 0
        for i in 1:n
            M = zeros(T, n, 1)
            M[i, 1] = one(T)
            push!(dirs, M)
        end
        for i in 1:(n - 1), j in (i + 1):n
            for (a, b) in ((1.0, 1.0), (1.0, -1.0), (1.0, 0.5), (0.5, -1.0))
                M = zeros(T, n, 1)
                M[i, 1] = T(a)
                M[j, 1] = T(b)
                push!(dirs, M ./ norm(M))
            end
        end
        return dirs
    end

    if n == 1
        for m in 1:q
            for c in (0.5, -0.5, 2.0, -2.0, 0.25, -0.25, 1.5, -1.5)
                M = zeros(T, 1, q + 1)
                M[1, 1] = one(T)
                M[1, m + 1] = T(c)
                push!(dirs, M ./ norm(M))
            end
        end
        if q >= 2
            for (c1, c2) in ((0.5, -0.5), (0.5, 0.25), (-0.5, 0.25))
                M = zeros(T, 1, q + 1)
                M[1, 1] = one(T)
                M[1, 2] = T(c1)
                M[1, 3] = T(c2)
                push!(dirs, M ./ norm(M))
            end
        end
        return dirs
    end

    if n == 2 && q == 1
        raw_pairs = (
            ([1.0, 0.0], [0.5, 0.0]),
            ([0.0, 1.0], [0.0, 0.5]),
            ([1.0, 0.0], [0.0, 0.5]),
            ([0.0, 1.0], [0.5, 0.0]),
            ([1.0, 1.0], [0.5, -0.5]),
            ([1.0, -1.0], [0.5, 0.5]),
            ([0.5, 1.0], [-1.0, 0.5]),
            ([1.0, 0.5], [0.5, -1.0]),
            ([1.0, 0.0], [-2.0, 1.0]),
            ([0.0, 1.0], [1.0, -2.0]),
        )
        for (u, v) in raw_pairs
            M = hcat(T.(u), T.(v))
            push!(dirs, M ./ norm(M))
        end
        return dirs
    end

    for m in 1:q
        for i in 1:n, j in 1:n
            M = zeros(T, n, q + 1)
            M[i, 1] = one(T)
            M[j, m + 1] = T(0.5)
            push!(dirs, M ./ norm(M))
        end
        for i in 1:(n - 1), j in (i + 1):n
            for (u0a, u0b, uma, umb) in (
                (1.0, 1.0, 0.5, -0.5),
                (1.0, -1.0, 0.5, 0.5),
                (0.5, 1.0, -1.0, 0.5),
                (1.0, 0.5, 0.5, -1.0),
                (1.0, 0.0, -2.0, 1.0),
                (0.0, 1.0, 1.0, -2.0),
            )
                M = zeros(T, n, q + 1)
                M[i, 1] = T(u0a); M[j, 1] = T(u0b)
                M[i, m + 1] = T(uma); M[j, m + 1] = T(umb)
                push!(dirs, M ./ norm(M))
            end
        end
    end
    return dirs
end

function _svarma_gmm_data_moments(Z::AbstractMatrix{Tz}, dirs::Vector{Matrix{T}},
                                  q::Int) where {Tz<:Real,T<:AbstractFloat}
    T_z, n = size(Z)
    T2 = T_z - q
    P = length(dirs)
    M = Matrix{Tz}(undef, T2, 3 * P)
    @inbounds for d in 1:P
        U_d = dirs[d]
        c0 = 3 * (d - 1)
        for tau in 1:T2
            w = zero(Tz)
            t_idx = q + tau
            for m in 0:q
                for i in 1:n
                    w += Z[t_idx - m, i] * Tz(U_d[i, m + 1])
                end
            end
            w2 = w * w
            M[tau, c0 + 1] = w2
            M[tau, c0 + 2] = (w2 * w) * Tz(0.25)
            M[tau, c0 + 3] = (w2 * w2) * Tz(1 / 24)
        end
    end
    return M
end

function _svarma_gmm_scaled_inv(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    As = Matrix{T}((A .+ transpose(A)) ./ T(2))
    d = T[sqrt(max(As[i, i], sqrt(eps(T)))) for i in 1:n]
    d_inv = Diagonal(one(T) ./ d)
    R = Matrix{T}(d_inv * As * d_inv)
    R = Matrix{T}((R .+ transpose(R)) ./ T(2))
    F = eigen(Symmetric(R))
    lam_floor = max(sqrt(eps(T)) * max(maximum(F.values), one(T)), eps(T))
    inv_vals = one(T) ./ max.(F.values, lam_floor)
    R_inv = F.vectors * Diagonal(inv_vals) * transpose(F.vectors)
    A_inv = Matrix{T}(d_inv * R_inv * d_inv)
    A_inv = Matrix{T}((A_inv .+ transpose(A_inv)) ./ T(2))
    if !isposdef(Symmetric(A_inv))
        lam_min = try
            eigmin(Symmetric(A_inv))
        catch
            zero(T)
        end
        scale = max(maximum(abs, diag(A_inv)), one(T))
        jitter = (isfinite(lam_min) && lam_min < zero(T) ? -lam_min : zero(T)) +
                 max(sqrt(eps(T)) * scale, eps(T))
        for _ in 1:6
            cand = A_inv + jitter * Matrix{T}(I, n, n)
            isposdef(Symmetric(cand)) && return cand
            jitter *= T(10)
        end
        return A_inv + jitter * Matrix{T}(I, n, n)
    end
    return A_inv
end

function _svarma_gmm_unpack_theta(theta::AbstractVector{Tθ}, n::Int, q::Int) where {Tθ<:Real}
    nn = n * n
    C_list = Vector{Matrix{Tθ}}(undef, q + 1)
    @inbounds for j in 0:q
        off = j * nn
        C_list[j + 1] = reshape(theta[(off + 1):(off + nn)], n, n)
    end
    k_off = (q + 1) * nn
    kappa3 = @view theta[(k_off + 1):(k_off + n)]
    kappa4 = @view theta[(k_off + n + 1):(k_off + 2 * n)]
    return C_list, kappa3, kappa4
end

function _svarma_gmm_pack_theta(C_list::Vector{<:AbstractMatrix},
                                kappa3::AbstractVector,
                                kappa4::AbstractVector)
    Tθ = float(promote_type(eltype(C_list[1]), eltype(kappa3), eltype(kappa4)))
    parts = Tθ[]
    for Cj in C_list
        append!(parts, vec(Tθ.(Cj)))
    end
    append!(parts, Tθ.(kappa3))
    append!(parts, Tθ.(kappa4))
    return parts
end

function _svarma_gmm_theo_moments(theta::AbstractVector{Tθ}, dirs::Vector{Matrix{T}},
                                  n::Int, q::Int) where {Tθ<:Real,T<:AbstractFloat}
    C_list, kappa3, kappa4 = _svarma_gmm_unpack_theta(theta, n, q)
    P = length(dirs)
    mu_theo = Vector{Tθ}(undef, 3 * P)
    a_s = Vector{Tθ}(undef, n)

    @inbounds for d in 1:P
        U_d = dirs[d]
        m2 = zero(Tθ)
        m3 = zero(Tθ)
        m4_cum = zero(Tθ)
        for s in 0:(2 * q)
            fill!(a_s, zero(Tθ))
            m_lo = max(0, s - q)
            m_hi = min(q, s)
            for m in m_lo:m_hi
                j = s - m
                Cj = C_list[j + 1]
                for i in 1:n
                    u_mi = Tθ(U_d[i, m + 1])
                    if !iszero(u_mi)
                        for k in 1:n
                            a_s[k] += Cj[i, k] * u_mi
                        end
                    end
                end
            end
            for k in 1:n
                ak = a_s[k]
                ak2 = ak * ak
                m2 += ak2
                m3 += ak2 * ak * kappa3[k]
                m4_cum += ak2 * ak2 * kappa4[k]
            end
        end
        c0 = 3 * (d - 1)
        mu_theo[c0 + 1] = m2
        mu_theo[c0 + 2] = m3 * Tθ(0.25)
        mu_theo[c0 + 3] = (Tθ(3) * m2 * m2 + m4_cum) * Tθ(1 / 24)
    end
    return mu_theo
end

# =============================================================================
# Initial Fundamental Factorization & Blaschke Candidate Generator
# =============================================================================

function _svarma_gmm_fundamental_init(Z_all::Matrix{T}, q::Int) where {T<:AbstractFloat}
    T_z, n = size(Z_all)
    if q == 0
        Gamma0 = Matrix{T}(Symmetric((transpose(Z_all) * Z_all) / T(T_z)))
        L0 = Matrix{T}(safe_cholesky(Gamma0; silent=true))
        return zeros(T, n, n, 0), L0
    end

    Gamma0 = Matrix{T}(Symmetric((transpose(Z_all) * Z_all) / T(T_z)))
    Theta_fund = zeros(T, n, n, q)
    Sigma_e = copy(Gamma0)

    if q == 1 && T_z > 2
        Gamma1 = (transpose(@view Z_all[2:T_z, :]) * @view Z_all[1:(T_z - 1), :]) / T(T_z)
        S_cur = copy(Gamma0)
        ok = false
        for _ in 1:80
            S_inv = Matrix{T}(robust_inv(S_cur; silent=true))
            S_next = Gamma0 - Gamma1 * S_inv * transpose(Gamma1)
            S_next = Matrix{T}((S_next .+ transpose(S_next)) ./ T(2))
            if isposdef(Symmetric(S_next))
                if norm(S_next - S_cur) <= T(1e-8) * max(one(T), norm(S_cur))
                    S_cur = S_next
                    ok = true
                    break
                end
                S_cur = T(0.5) .* S_cur .+ T(0.5) .* S_next
            else
                ok = false
                break
            end
        end
        if ok
            Th_try = Gamma1 * Matrix{T}(robust_inv(S_cur; silent=true))
            max_lam = maximum(abs, eigvals(Th_try))
            if isfinite(max_lam) && max_lam < T(0.98)
                Theta_fund[:, :, 1] .= Th_try
                Sigma_e = S_cur
                L0 = Matrix{T}(safe_cholesky(Sigma_e; silent=true))
                return Theta_fund, L0
            end
        end
    end

    # Fallback / q >= 2: long-VAR Wold inversion on Z_all
    L_var = min(max(4 + 2 * q, q + 2), max(q + 1, div(T_z, max(4 * n, 8))))
    if T_z > L_var * n + 4
        Y_v, X_v = construct_var_matrices(Z_all, L_var)
        B_v = robust_inv(transpose(X_v) * X_v; silent=true) * (transpose(X_v) * Y_v)
        E_v = Y_v .- X_v * B_v
        Sigma_e = Matrix{T}(Symmetric((transpose(E_v) * E_v) / T(size(E_v, 1))))
        A_ar = extract_ar_coefficients(B_v, n, L_var)
        # Invert A(L) = I - sum A_k L^k to get Wold MA coefficients Psi_1..Psi_q
        Psi = Vector{Matrix{T}}(undef, q + 1)
        Psi[1] = Matrix{T}(I, n, n)
        for j in 1:q
            acc = zeros(T, n, n)
            for k in 1:min(j, L_var)
                acc .+= A_ar[k] * Psi[j - k + 1]
            end
            Psi[j + 1] = acc
            Theta_fund[:, :, j] .= acc
        end
        # Ensure companion eigenvalues lie strictly inside unit circle
        F_ma = _svarma_companion(Theta_fund)
        max_lam = maximum(abs, eigvals(F_ma))
        if max_lam >= T(0.96)
            rho = T(0.90) / max_lam
            for j in 1:q
                Theta_fund[:, :, j] .*= rho^j
            end
        end
    end

    L0 = Matrix{T}(safe_cholesky(Sigma_e; silent=true))
    return Theta_fund, L0
end

function _svarma_gmm_ica_rotations(shocks::Matrix{T}, rng::AbstractRNG) where {T<:AbstractFloat}
    T_eff, n = size(shocks)
    rots = Matrix{T}[Matrix{T}(I, n, n)]
    n == 1 && return rots

    # Whiten shocks for JADE and FastICA
    Uc = shocks .- mean(shocks; dims=1)
    S_u = Symmetric((transpose(Uc) * Uc) / T(T_eff))
    L_u = Matrix{T}(safe_cholesky(Matrix{T}(S_u); silent=true))
    L_u_inv = Matrix{T}(robust_inv(L_u; silent=true))
    Z_w = Matrix{T}(Uc * transpose(L_u_inv))

    # 1. JADE rotation
    try
        cum_mats = _jade_cumulant_matrices(Z_w)
        V_jade, _ = _joint_diagonalization(cum_mats; max_iter=100, tol=T(1e-6))
        push!(rots, Matrix{T}(V_jade))
    catch
    end

    # 2. FastICA symmetric (:logcosh) rotation
    try
        W_ica, _ = _fastica_symmetric(Z_w, n; contrast=:logcosh, max_iter=200,
                                      tol=T(1e-6), rng=rng)
        push!(rots, Matrix{T}(transpose(W_ica)))
    catch
    end

    return rots
end

# =============================================================================
# Assumption A.7 Normalization (GMR 2020)
# =============================================================================

function _svarma_normalize_a7(C_mat::Matrix{T}, Theta_3d::Array{T,3},
                              other_col_vecs::Vector{<:AbstractVector{T}}=Vector{T}[]) where {T<:AbstractFloat}
    n = size(C_mat, 1)
    q = size(Theta_3d, 3)
    C_s = copy(C_mat)
    signs = ones(Int, n)
    tol_z = T(1e-12)
    @inbounds for j in 1:n
        sgn = one(T)
        for i in 1:n
            if abs(C_s[i, j]) > tol_z
                if C_s[i, j] < zero(T)
                    sgn = -one(T)
                    signs[j] = -1
                end
                break
            end
        end
        if sgn < zero(T)
            C_s[:, j] .*= -one(T)
        end
    end

    # Sort columns so row 1 is non-decreasing (tie-breaking on rows 2..n)
    perm = sortperm(1:n; by=j -> ntuple(i -> abs(C_s[i, j]) <= tol_z ? zero(T) : C_s[i, j], n))
    C_out = C_s[:, perm]
    # Note: Θ_j = C_j C_0⁻¹ is invariant to signed column permutations on C_0 and C_j!
    # We still return perm and signs so shocks and shock-wise parameters follow A.7.
    vecs_out = Vector{T}[]
    for (idx, v) in enumerate(other_col_vecs)
        vp = Vector{T}(v[perm])
        if idx == 1  # odd cumulant / skewness parameter flips with column sign
            @inbounds for j in 1:n
                if signs[perm[j]] < 0
                    vp[j] = -vp[j]
                end
            end
        end
        push!(vecs_out, vp)
    end
    return C_out, copy(Theta_3d), perm, signs, vecs_out
end

# =============================================================================
# Nonparametric KDE Log-Likelihood Helper (GMR 2020, §4.2.3)
# =============================================================================

function _svarma_gmm_kde_loglik(shocks::Matrix{T}, log_jac::T) where {T<:AbstractFloat}
    T_eff, n = size(shocks)
    ll = -T(T_eff) * log_jac
    kdes = Vector{KernelDensity{T}}(undef, n)
    for j in 1:n
        s = @view shocks[:, j]
        kd = kernel_density(s)
        kdes[j] = kd
        gx = kd.x
        gf = kd.density
        m = length(gx)
        dx = gx[2] - gx[1]
        @inbounds for t in 1:T_eff
            xv = s[t]
            val = if xv <= gx[1]
                gf[1]
            elseif xv >= gx[m]
                gf[m]
            else
                pos = (xv - gx[1]) / dx + one(T)
                lo = clamp(floor(Int, pos), 1, m - 1)
                w = pos - T(lo)
                (one(T) - w) * gf[lo] + w * gf[lo + 1]
            end
            ll += log(max(val, T(1e-300)))
        end
    end
    return ll, kdes
end

# =============================================================================
# Core Two-Step 2SLS-GMM Estimator
# =============================================================================

function _svarma_gmm_estimate(Y::Matrix{T}, p::Int, q::Int;
                              k_iv::Int=p + 3,
                              max_iter::Int=500,
                              tol::Real=1e-7,
                              varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                              shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    tol_T = T(tol)

    # Step 1: 2SLS for (μ, Φ₁..Φ_p)
    s1 = _svarma_gmm_step1_2sls(Y, p, q, k_iv)
    mu_hat = s1.mu
    Phi_3d = s1.Phi
    Z_all = s1.Z_all  # (T_obs - p) × n

    # Step 2 direction bank and data moments
    dirs = _svarma_gmm_direction_bank(n, q, T)
    M_data = _svarma_gmm_data_moments(Z_all, dirs, q)
    T2, M_dim = size(M_data)
    m_bar_data = vec(mean(M_data; dims=1))

    # Stage-1 diagonal scaling weighting matrix
    var_m = vec(var(M_data; dims=1, corrected=false))
    W1 = Diagonal(T[one(T) / max(var_m[k], T(1e-4)) for k in 1:M_dim])

    # Build multi-start initializations from fundamental factorization + JADE/FastICA + Blaschke flips
    Theta_fund, L_fund = _svarma_gmm_fundamental_init(Z_all, q)
    rots = _svarma_gmm_ica_rotations(
        q == 0 ? Z_all : ssvarma_filter_shocks(Z_all, Theta_fund, L_fund; tol=1e-7).shocks,
        rng,
    )

    candidate_starts = Vector{Vector{T}}()
    seen_keys = Vector{Vector{T}}()
    for R_rot in rots
        C_cand0 = L_fund * R_rot
        reps = try
            blaschke_representations(Theta_fund, C_cand0; tol=1e-6)
        catch
            [(Theta=Theta_fund, C=C_cand0)]
        end
        # Cap total Blaschke enumerations to avoid combinatorial explosion on huge n*q
        n_keep = min(length(reps), 32)
        for idx in 1:n_keep
            rep = reps[idx]
            C0_s = Matrix{T}(rep.C)
            Th_s = Array{T,3}(rep.Theta)
            # Filter candidate shocks to initialize κ₃, κ₄
            sh_s = try
                ssvarma_filter_shocks(Z_all, Th_s, C0_s; tol=1e-7).shocks
            catch
                continue
            end
            k3_s = T[mean(@view(sh_s[:, j]) .^ 3) for j in 1:n]
            k4_s = T[mean(@view(sh_s[:, j]) .^ 4) - T(3) for j in 1:n]
            C_list_s = Matrix{T}[C0_s]
            for j in 1:q
                push!(C_list_s, @view(Th_s[:, :, j]) * C0_s)
            end
            th_s = _svarma_gmm_pack_theta(C_list_s, k3_s, k4_s)
            is_dup = any(k -> maximum(abs, k .- th_s) < T(1e-4), seen_keys)
            if !is_dup
                push!(seen_keys, th_s)
                push!(candidate_starts, th_s)
            end
        end
    end

    if isempty(candidate_starts)
        C_list_0 = Matrix{T}[L_fund]
        for j in 1:q
            push!(C_list_0, @view(Theta_fund[:, :, j]) * L_fund)
        end
        push!(candidate_starts, _svarma_gmm_pack_theta(C_list_0, zeros(T, n), zeros(T, n)))
    end

    function stage_obj(th::AbstractVector{Tθ}, W_mat::AbstractMatrix{T}) where {Tθ<:Real}
        diff = Tθ.(m_bar_data) .- _svarma_gmm_theo_moments(th, dirs, n, q)
        return dot(diff, W_mat * diff)
    end

    # Stage 1 GMM optimization across all regime candidates
    stage1_results = Tuple{T,Vector{T},Bool,Int}[]
    opts1 = Optim.Options(iterations=max_iter, g_tol=tol_T, f_reltol=T(1e-12),
                          allow_f_increases=true, show_trace=false)
    for th0 in candidate_starts
        f1 = th -> stage_obj(th, W1)
        cfg1 = ForwardDiff.GradientConfig(f1, th0)
        g1! = (G, x) -> ForwardDiff.gradient!(G, f1, x, cfg1)
        res = try
            Optim.optimize(f1, g1!, th0, Optim.LBFGS(), opts1)
        catch
            nothing
        end
        res === nothing && continue
        val = T(Optim.minimum(res))
        isfinite(val) || continue
        push!(stage1_results, (val, Vector{T}(Optim.minimizer(res)), Optim.converged(res), Optim.iterations(res)))
    end

    sort!(stage1_results; by=first)
    # Keep distinct local minima across regimes so Stage 2 evaluates every basin with Ω*
    distinct_s1 = Tuple{T,Vector{T},Bool,Int}[]
    for cand in stage1_results
        if !any(d -> maximum(abs, d[2] .- cand[2]) < T(1e-2), distinct_s1)
            push!(distinct_s1, cand)
            length(distinct_s1) >= 8 && break
        end
    end
    if isempty(distinct_s1)
        th_fallback = candidate_starts[1]
        distinct_s1 = [(T(stage_obj(th_fallback, W1)), th_fallback, false, 0)]
    end
    stage1_results = distinct_s1
    top_k = length(stage1_results)

    # Helper: compute G_alpha, D, Q_0, Omega_star at a parameter vector theta_pt
    function compute_q0_and_D(theta_pt::Vector{T})
        G_alpha = ForwardDiff.jacobian(a -> begin
            Z_a = _svarma_gmm_residuals_from_alpha(Y, a, p, n)
            vec(mean(_svarma_gmm_data_moments(Z_a, dirs, q); dims=1))
        end, s1.alpha)

        D_mat = ForwardDiff.jacobian(
            th -> T.(m_bar_data) .- _svarma_gmm_theo_moments(th, dirs, n, q),
            theta_pt,
        )

        mu_pt = _svarma_gmm_theo_moments(theta_pt, dirs, n, q)
        H_y = M_data .- transpose(mu_pt)

        # Combine first-step 2SLS influence: ψ_t = h_{Y,t} + 1_{t >= t_start} * (T2/T1) * G_alpha * Q_pi * h_{1,t}
        Psi = copy(H_y)
        shift = s1.t_start - (p + q + 1)
        ratio = T(T2) / T(s1.T1)
        G_Q = G_alpha * s1.Q_pi
        @inbounds for tau1 in 1:s1.T1
            tau2 = tau1 + shift
            if 1 <= tau2 <= T2
                contrib = G_Q * @view(s1.h1_mat[tau1, :])
                for m in 1:M_dim
                    Psi[tau2, m] += ratio * contrib[m]
                end
            end
        end

        active_psi = [j for j in 1:M_dim if sum(abs2, @view(Psi[1:(end - 1), j])) > eps(T)]
        bw = isempty(active_psi) ? (2 * q + 2) :
             max(2 * q + 2, optimal_bandwidth_nw(Matrix{T}(@view Psi[:, active_psi]); kernel=:bartlett))
        Q_0 = Matrix{T}(lrcov(Psi; kernel=:bartlett, bandwidth=bw, demean=true))
        Omega_star = _svarma_gmm_scaled_inv(Q_0)
        return G_alpha, D_mat, Q_0, Omega_star, Psi, bw
    end

    best_s2_val = T(Inf)
    best_th = stage1_results[1][2]
    best_conv = stage1_results[1][3]
    best_iters = stage1_results[1][4]

    opts2 = Optim.Options(iterations=max_iter, g_tol=tol_T, f_reltol=T(1e-12),
                          allow_f_increases=true, show_trace=false)

    for idx in 1:top_k
        th1 = stage1_results[idx][2]
        _, _, _, Omega_init, _, _ = compute_q0_and_D(th1)
        f2 = th -> stage_obj(th, Omega_init)
        cfg2 = ForwardDiff.GradientConfig(f2, th1)
        g2! = (G, x) -> ForwardDiff.gradient!(G, f2, x, cfg2)
        res2 = try
            Optim.optimize(f2, g2!, th1, Optim.LBFGS(), opts2)
        catch
            nothing
        end
        th2 = res2 === nothing ? th1 : Vector{T}(Optim.minimizer(res2))
        val2 = res2 === nothing ? T(f2(th1)) : T(Optim.minimum(res2))
        conv2 = res2 === nothing ? stage1_results[idx][3] : Optim.converged(res2)
        iters2 = res2 === nothing ? stage1_results[idx][4] : Optim.iterations(res2)
        if isfinite(val2) && val2 < best_s2_val
            best_s2_val = val2
            best_th = th2
            best_conv = conv2
            best_iters = iters2
        end
    end

    # Unpack, check nonsingular C0, normalize by Assumption A.7
    C_list_raw, k3_raw, k4_raw = _svarma_gmm_unpack_theta(best_th, n, q)
    C0_raw = Matrix{T}(C_list_raw[1])
    _svarma_check_C_nonsingular(C0_raw)
    C0_raw_inv = Matrix{T}(robust_inv(C0_raw; silent=true))
    Theta_raw = Array{T,3}(undef, n, n, q)
    for j in 1:q
        @views Theta_raw[:, :, j] .= C_list_raw[j + 1] * C0_raw_inv
    end

    C0_norm, Theta_norm, perm, signs, vecs_norm = _svarma_normalize_a7(
        C0_raw, Theta_raw, Vector{T}[Vector{T}(k3_raw), Vector{T}(k4_raw)],
    )
    k3_norm, k4_norm = vecs_norm[1], vecs_norm[2]
    C_list_norm = Matrix{T}[C0_norm]
    for j in 1:q
        push!(C_list_norm, @view(Theta_norm[:, :, j]) * C0_norm)
    end
    theta_norm = _svarma_gmm_pack_theta(C_list_norm, k3_norm, k4_norm)

    # Re-evaluate exact Q_0, Omega_star, D, Sigma(Omega_star) at A.7-normalized θ̂
    G_alpha, D_mat, Q_0, Omega_star, Psi_final, bw_final = compute_q0_and_D(theta_norm)
    H_gmm = transpose(D_mat) * Omega_star * D_mat
    H_gmm_inv = _svarma_gmm_scaled_inv(H_gmm)
    V_theta = H_gmm_inv ./ T(T2)
    V_theta = Matrix{T}((V_theta .+ transpose(V_theta)) ./ T(2))

    # Full joint covariance of (α, θ) and delta-method mapping to [μ; vec(Φ); vec(Θ); vec(C)]
    dim_alpha = length(s1.alpha)
    dim_theta = length(theta_norm)
    B_theta = -H_gmm_inv * transpose(D_mat) * Omega_star

    # Build joint influence matrix over common sample T2: [ξ_{1,t}; ξ_{2,t}]
    Xi_joint = Matrix{T}(undef, T2, dim_alpha + dim_theta)
    fill!(Xi_joint, zero(T))
    shift = s1.t_start - (p + q + 1)
    ratio = T(T2) / T(s1.T1)
    @inbounds for tau1 in 1:s1.T1
        tau2 = tau1 + shift
        if 1 <= tau2 <= T2
            xi1 = ratio .* (s1.Q_pi * @view(s1.h1_mat[tau1, :]))
            for k in 1:dim_alpha
                Xi_joint[tau2, k] = xi1[k]
            end
        end
    end
    @inbounds for tau2 in 1:T2
        xi2 = B_theta * @view(Psi_final[tau2, :])
        for k in 1:dim_theta
            Xi_joint[tau2, dim_alpha + k] = xi2[k]
        end
    end
    V_joint = Matrix{T}(lrcov(Xi_joint; kernel=:bartlett, bandwidth=bw_final, demean=true)) ./ T(T2)
    V_joint = Matrix{T}((V_joint .+ transpose(V_joint)) ./ T(2))
    # Replace lower-right block directly with exact V_theta = (1/T2)*(D' Ω* D)⁻¹ for precision
    V_joint[(dim_alpha + 1):end, (dim_alpha + 1):end] .= V_theta

    # Map (α, θ) -> coef(SVARMAModel) = [μ; vec(Φ); vec(Θ); vec(C)]
    function map_to_coef(at::AbstractVector{Tθ}) where {Tθ<:Real}
        a_v = @view at[1:dim_alpha]
        th_v = @view at[(dim_alpha + 1):end]
        A_m = reshape(a_v, 1 + n * p, n)
        mu_v = @view A_m[1, :]
        Phi_vec = Tθ[]
        for k in 1:p
            rows = (2 + (k - 1) * n):(1 + k * n)
            append!(Phi_vec, vec(transpose(@view A_m[rows, :])))
        end
        C_l, _, _ = _svarma_gmm_unpack_theta(th_v, n, q)
        C0_m = C_l[1]
        C0_m_inv = robust_inv(C0_m; silent=true)
        Th_vec = Tθ[]
        for j in 1:q
            append!(Th_vec, vec(C_l[j + 1] * C0_m_inv))
        end
        return vcat(Vector{Tθ}(mu_v), Phi_vec, Th_vec, vec(C0_m))
    end

    at_hat = vcat(s1.alpha, theta_norm)
    J_coef = ForwardDiff.jacobian(map_to_coef, at_hat)
    vcov_coef = J_coef * V_joint * transpose(J_coef)
    vcov_coef = Matrix{T}((vcov_coef .+ transpose(vcov_coef)) ./ T(2))
    se_coef = T[sqrt(max(vcov_coef[i, i], zero(T))) for i in 1:size(vcov_coef, 1)]

    # Filter shocks, classify regime, fit nonparametric shock KDEs
    f_res = ssvarma_filter_shocks(Y, mu_hat, Phi_3d, Theta_norm, C0_norm; tol=1e-8)
    reg = fundamentalness_regime(Theta_norm, C0_norm; tol=1e-8)
    ll_kde, _ = _svarma_gmm_kde_loglik(f_res.shocks, f_res.log_jacobian)

    T_eff = size(f_res.shocks, 1)
    n_params = n + p * n^2 + q * n^2 + n^2
    aic_val = -T(2) * ll_kde + T(2) * T(n_params)
    bic_val = -T(2) * ll_kde + log(T(T_eff)) * T(n_params)

    re_vec = T[T(real(z)) for z in reg.eigenvalues]
    im_vec = T[T(imag(z)) for z in reg.eigenvalues]
    dists_vec = fill(:nonparametric, n)
    dp_vec = Vector{T}[T[] for _ in 1:n]
    vn_vec = varnames === nothing ? default_var_names(n) : String[string(v) for v in varnames]
    sn_vec = shock_names === nothing ? default_shock_names(n) : String[string(s) for s in shock_names]

    model = SVARMAModel{T}(
        Y, p, q, mu_hat, Phi_3d, Theta_norm, C0_norm, f_res.shocks,
        re_vec, im_vec, reg.regime, reg.gmr_region, dists_vec, dp_vec,
        ll_kde, aic_val, bic_val, vn_vec, sn_vec, best_conv, :gmm,
        vcov_coef, se_coef,
    )

    return (
        model = model,
        theta = theta_norm,
        theta_vcov = V_theta,
        Q_pi = s1.Q_pi,
        Q_0 = Q_0,
        Omega_star = Omega_star,
        D = D_mat,
        G_alpha = G_alpha,
        J_stat = T(T2) * best_s2_val,
        T1 = s1.T1,
        T2 = T2,
    )
end
