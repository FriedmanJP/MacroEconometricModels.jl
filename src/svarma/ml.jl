# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Truncated Maximum Likelihood and Pseudo-ML estimator for Strong Structural VARMA(p, q)
models with non-fundamental Jacobian and multi-regime Blaschke basin exploration
(Gouriéroux, Monfort & Renne 2020, §4.1.3, eq. 4.16, Appendix B.2).
"""

using LinearAlgebra, Statistics, Distributions, Random
import ForwardDiff
import Optim

# =============================================================================
# Dual-Compatible Schur Partition & Truncated Filter Helpers
# =============================================================================

@inline _svarma_ml_primal(x::AbstractFloat) = x
@inline _svarma_ml_primal(x::ForwardDiff.Dual) = _svarma_ml_primal(ForwardDiff.value(x))

function _svarma_ml_logabsdet(M::AbstractMatrix{Tm}) where {Tm<:Real}
    Tm <: AbstractFloat && return Tm(first(logabsdet(M)))
    F = lu(M)
    s = zero(Tm)
    @inbounds for i in 1:size(M, 1)
        s += log(abs(F.U[i, i]))
    end
    return s
end

function _svarma_ml_ar_residuals(Y::AbstractMatrix{Ty}, mu::AbstractVector{Tm},
                                  Phi_3d::AbstractArray{Tp,3}) where {Ty<:Real,Tm<:Real,Tp<:Real}
    T_obs, n = size(Y)
    p = size(Phi_3d, 3)
    T_out = float(promote_type(Ty, Tm, Tp))
    T_eff = T_obs - p
    U = Matrix{T_out}(undef, T_eff, n)
    @inbounds for t in 1:T_eff
        idx = p + t
        for i in 1:n
            val = T_out(Y[idx, i]) - T_out(mu[i])
            for k in 1:p
                for j in 1:n
                    val -= T_out(Phi_3d[i, j, k]) * T_out(Y[idx - k, j])
                end
            end
            U[t, i] = val
        end
    end
    return U
end

function _svarma_ml_filter(U::AbstractMatrix{Tu}, Theta_3d::AbstractArray{Tth,3},
                           C::AbstractMatrix{Tc}; tol::Real=1e-7) where {Tu<:Real,Tth<:Real,Tc<:Real}
    T_eff, n = size(U)
    q = size(Theta_3d, 3)
    T_work = float(promote_type(Tu, Tth, Tc))
    C_inv = robust_inv(C; silent=true)
    ld_C = T_work(_svarma_ml_logabsdet(C))

    if q == 0
        eps_mat = Matrix{T_work}(U)
        eta = eps_mat * transpose(C_inv)
        return (shocks=eta, log_jacobian=ld_C, n_fundamental=0, n_nonfundamental=0)
    end

    Theta_0 = _svarma_ml_primal.(Theta_3d)
    T_p = typeof(Theta_0[1])
    part = _svarma_schur_partition(Array{T_p,3}(Theta_0), tol)
    Z_0 = part.Z
    m1 = part.m1
    m2 = part.m2
    nq = n * q

    if Tth <: AbstractFloat
        T_schur = part.T_mat
        Z1_top = @view Z_0[1:n, 1:m1]
        Z2_top = @view Z_0[1:n, (m1 + 1):nq]
        M1_top = Z1_top
        M2_top = Z2_top
        Ut1 = U * Z1_top
        Ut2 = U * Z2_top
        T11 = @view T_schur[1:m1, 1:m1]
        T12 = @view T_schur[1:m1, (m1 + 1):nq]
        T22 = T_schur[(m1 + 1):nq, (m1 + 1):nq]
    else
        # Exact Schur-Sylvester invariant-subspace perturbation for arbitrary Dual order
        F_ma = _svarma_companion(Array{Tth,3}(Theta_3d))
        Z_th = Matrix{Tth}(Z_0)
        R = transpose(Z_th) * F_ma * Z_th
        R11 = R[1:m1, 1:m1]
        R12 = R[1:m1, (m1 + 1):nq]
        R21 = R[(m1 + 1):nq, 1:m1]
        R22 = R[(m1 + 1):nq, (m1 + 1):nq]

        if m1 > 0 && m2 > 0
            T11_0 = @view part.T_mat[1:m1, 1:m1]
            T22_0 = @view part.T_mat[(m1 + 1):nq, (m1 + 1):nq]
            A_sylv_0 = kron(Matrix{T_p}(I, m1, m1), T22_0) -
                       kron(transpose(T11_0), Matrix{T_p}(I, m2, m2))
            F_sylv = lu(A_sylv_0)
            X = zeros(Tth, m2, m1)
            for _ in 1:3
                res = R21 .+ R22 * X .- X * R11 .- X * R12 * X
                dX_vec = F_sylv \ vec(res)
                X .-= reshape(dX_vec, m2, m1)
            end
            Z1_full = Matrix{Tth}(@view Z_0[:, 1:m1])
            Z2_full = Matrix{Tth}(@view Z_0[:, (m1 + 1):nq])
            M1_raw = Z1_full .+ Z2_full * X
            M2_raw = Z2_full .- Z1_full * transpose(X)
            # X has primal 0, so X'X and XX' are O(ϵ²); 3-term series is exact up to 5th order
            A1 = transpose(X) * X
            A2 = X * transpose(X)
            S1 = Matrix{Tth}(I, m1, m1) .- Tth(0.5) .* A1 .+ Tth(0.375) .* (A1 * A1)
            S2 = Matrix{Tth}(I, m2, m2) .- Tth(0.5) .* A2 .+ Tth(0.375) .* (A2 * A2)
            Q1 = M1_raw * S1
            Q2 = M2_raw * S2
            Q_orth = hcat(Q1, Q2)
            T_schur = transpose(Q_orth) * F_ma * Q_orth
            T11 = T_schur[1:m1, 1:m1]
            T12 = T_schur[1:m1, (m1 + 1):nq]
            T22 = T_schur[(m1 + 1):nq, (m1 + 1):nq]
            M1_top = Q1[1:n, :]
            M2_top = Q2[1:n, :]
            Ut1 = U * M1_top
            Ut2 = U * M2_top
        elseif m1 == 0
            T11 = R11
            T12 = R12
            T22 = R22
            M1_top = zeros(Tth, n, 0)
            M2_top = Matrix{Tth}(@view Z_0[1:n, 1:nq])
            Ut1 = zeros(T_work, T_eff, 0)
            Ut2 = U * M2_top
        else
            T11 = R11
            T12 = R12
            T22 = R22
            M1_top = Matrix{Tth}(@view Z_0[1:n, 1:nq])
            M2_top = zeros(Tth, n, 0)
            Ut1 = U * M1_top
            Ut2 = zeros(T_work, T_eff, 0)
        end
    end

    # Block 2: backward recursion from terminal condition ε̃_{2, T_eff} = 0
    E2 = zeros(T_work, m2, T_eff)
    E2_0 = zeros(T_work, m2)
    ld_T22 = zero(T_work)
    if m2 > 0
        T22_inv = robust_inv(T22; silent=true)
        @inbounds for t in (T_eff - 1):-1:1
            E2[:, t] = T22_inv * (@view(E2[:, t + 1]) .- @view(Ut2[t + 1, :]))
        end
        E2_0 = T22_inv * (@view(E2[:, 1]) .- @view(Ut2[1, :]))
        ld_T22 = T_work(_svarma_ml_logabsdet(T22))
    end

    # Block 1: forward recursion from initial condition ε̃_{1, 0} = 0
    E1 = zeros(T_work, m1, T_eff)
    if m1 > 0
        e1_prev = zeros(T_work, m1)
        @inbounds for t in 1:T_eff
            e1_curr = @view(Ut1[t, :]) .+ T11 * e1_prev
            if m2 > 0
                e2_prev = t == 1 ? E2_0 : @view(E2[:, t - 1])
                e1_curr = e1_curr .+ T12 * e2_prev
            end
            E1[:, t] = e1_curr
            e1_prev = e1_curr
        end
    end

    eps_mat = zeros(T_work, T_eff, n)
    if m1 > 0
        eps_mat .+= transpose(E1) * transpose(M1_top)
    end
    if m2 > 0
        eps_mat .+= transpose(E2) * transpose(M2_top)
    end
    eta = eps_mat * transpose(C_inv)

    return (
        shocks = eta,
        log_jacobian = ld_C + ld_T22,
        n_fundamental = m1,
        n_nonfundamental = m2,
    )
end

# =============================================================================
# Shock Distribution Families & 1D Warm-Start MLE (GMR fn. 40)
# =============================================================================

@inline function _svarma_ml_canonical_dist(dist::Symbol)
    dist === :student_t && return :student_t
    dist === :mixture_normal && return :mixture_normal
    dist === :mixture_normal_asym && return :mixture_normal_asym
    (dist === :pearson_iv || dist === :pml) && return :pearson_iv
    dist === :skew_normal && return :skew_normal
    throw(ArgumentError(
        "Unsupported distribution :$dist (use :student_t, :mixture_normal_asym, " *
        ":mixture_normal, :pearson_iv, :pml, or :skew_normal)."))
end

@inline function _svarma_ml_n_dp(dist::Symbol)
    dist === :student_t && return 1
    dist === :mixture_normal && return 2
    dist === :mixture_normal_asym && return 3
    dist === :pearson_iv && return 2
    dist === :skew_normal && return 1
    throw(ArgumentError("Unknown canonical distribution :$dist"))
end

@inline function _svarma_ml_logpdf_1d(x::Tx, dp_j::AbstractVector{Tθ}, dist::Symbol) where {Tx<:Real,Tθ<:Real}
    if dist === :student_t
        nu = Tθ(2.01) + exp(clamp(dp_j[1], Tθ(-30), Tθ(12)))
        return _student_t_logpdf(x, nu)
    elseif dist === :mixture_normal
        p_mix, sigma1, sigma2 = _unpack_shock_mixture(dp_j[1], dp_j[2])
        return _mixture_normal_logpdf(x, p_mix, sigma1, sigma2)
    elseif dist === :mixture_normal_asym
        p_mix, mu1, sigma1, mu2, sigma2 = _unpack_shock_mixture_asym(dp_j[1], dp_j[2], dp_j[3])
        return _mixture_normal_logpdf(x, p_mix, mu1, sigma1, mu2, sigma2)
    elseif dist === :pearson_iv
        skew = dp_j[1]
        m = Tθ(2.05) + exp(clamp(dp_j[2], Tθ(-30), Tθ(9.2)))
        return _pearson_iv_logpdf(x, skew, m)
    elseif dist === :skew_normal
        alpha = dp_j[1]
        Tacc = float(promote_type(Tx, Tθ))
        delta = alpha / sqrt(one(Tacc) + alpha^2)
        mu_z = delta * sqrt(Tacc(2) / Tacc(pi))
        sd_z = sqrt(one(Tacc) - mu_z^2)
        z = mu_z + sd_z * Tacc(x)
        return _skew_normal_logpdf(z, alpha) + log(sd_z)
    end
    throw(ArgumentError("Unknown canonical distribution :$dist"))
end

function _svarma_ml_dist_starts_1d(::Type{T}, dist::Symbol) where {T<:AbstractFloat}
    if dist === :student_t
        return [T[log(T(5.0) - T(2.01))], T[log(T(3.5) - T(2.01))], T[log(T(10.0) - T(2.01))]]
    elseif dist === :mixture_normal
        return [T[-1.1, 0.85], T[1.1, -0.85]]
    elseif dist === :mixture_normal_asym
        return [T[-1.5, 0.6, 0.5], T[-1.5, -0.6, 0.5], T[1.1, 0.5, -0.5], T[-1.0, -0.6, -0.5]]
    elseif dist === :pearson_iv
        return [T[0.3, log(T(5.0) - T(2.05))], T[-0.3, log(T(5.0) - T(2.05))]]
    elseif dist === :skew_normal
        return [T[0.5], T[-0.5], T[2.0], T[-2.0]]
    end
    throw(ArgumentError("Unknown canonical distribution :$dist"))
end

function _svarma_ml_fit_1d_dist(shocks::AbstractMatrix{T}, dist::Symbol) where {T<:AbstractFloat}
    T_eff, n = size(shocks)
    k_dp = _svarma_ml_n_dp(dist)
    dp_out = Vector{T}(undef, n * k_dp)
    starts_1d = _svarma_ml_dist_starts_1d(T, dist)
    opts = Optim.Options(iterations=80, g_tol=T(1e-6), allow_f_increases=true, show_trace=false)

    for j in 1:n
        s_col = Vector{T}(@view shocks[:, j])
        nll_1d = d -> begin
            acc = zero(eltype(d))
            @inbounds for t in 1:T_eff
                acc -= _svarma_ml_logpdf_1d(s_col[t], d, dist)
            end
            return acc
        end
        best_val = T(Inf)
        best_d = copy(starts_1d[1])
        for d0 in starts_1d
            cfg = ForwardDiff.GradientConfig(nll_1d, d0)
            g! = (G, x) -> ForwardDiff.gradient!(G, nll_1d, x, cfg)
            res = try
                Optim.optimize(nll_1d, g!, d0, Optim.LBFGS(), opts)
            catch
                nothing
            end
            res === nothing && continue
            v = T(Optim.minimum(res))
            if isfinite(v) && v < best_val
                best_val = v
                best_d = Vector{T}(Optim.minimizer(res))
            end
        end
        off = (j - 1) * k_dp
        dp_out[(off + 1):(off + k_dp)] .= best_d
    end
    return dp_out
end

function _svarma_ml_extract_dist_params(dp_opt::Vector{T}, signs::Vector{Int}, perm::Vector{Int},
                                        n::Int, dist::Symbol) where {T<:AbstractFloat}
    k_dp = _svarma_ml_n_dp(dist)
    dp_raw = [dp_opt[((j - 1) * k_dp + 1):(j * k_dp)] for j in 1:n]
    dp_perm = dp_raw[perm]
    dp_norm_vec = Vector{T}(undef, n * k_dp)
    dp_display = Vector{Vector{T}}(undef, n)

    for j in 1:n
        d = copy(dp_perm[j])
        sgn = signs[perm[j]]
        if sgn < 0
            if dist === :mixture_normal_asym
                d[2] = -d[2]
            elseif dist === :pearson_iv || dist === :skew_normal
                d[1] = -d[1]
            end
        end
        off = (j - 1) * k_dp
        dp_norm_vec[(off + 1):(off + k_dp)] .= d
        if dist === :student_t
            dp_display[j] = T[T(2.01) + exp(clamp(d[1], T(-30), T(12)))]
        elseif dist === :mixture_normal
            pm, s1, s2 = _unpack_shock_mixture(d[1], d[2])
            dp_display[j] = T[pm, s1, s2]
        elseif dist === :mixture_normal_asym
            pm, m1, s1, m2, s2 = _unpack_shock_mixture_asym(d[1], d[2], d[3])
            dp_display[j] = T[pm, m1, s1, m2, s2]
        elseif dist === :pearson_iv
            sk = d[1]
            m = T(2.05) + exp(clamp(d[2], T(-30), T(9.2)))
            dp_display[j] = T[sk, m]
        elseif dist === :skew_normal
            dp_display[j] = T[d[1]]
        end
    end
    return dp_norm_vec, dp_display
end

# =============================================================================
# Parameter Packing / Unpacking & Observation Log-Likelihood (GMR eq. 4.16)
# =============================================================================

function _svarma_ml_pack(mu::AbstractVector, Phi::AbstractArray{<:Real,3},
                         Theta::AbstractArray{<:Real,3}, C::AbstractMatrix,
                         dp::AbstractVector)
    T = float(promote_type(eltype(mu), eltype(Phi), eltype(Theta), eltype(C), eltype(dp)))
    return vcat(Vector{T}(mu), Vector{T}(vec(Phi)), Vector{T}(vec(Theta)),
                Vector{T}(vec(C)), Vector{T}(dp))
end

function _svarma_ml_unpack(params::AbstractVector{Tθ}, n::Int, p::Int, q::Int,
                           k_dp::Int) where {Tθ<:Real}
    nn = n * n
    off = 0
    mu = @view params[(off + 1):(off + n)]
    off += n
    Phi = reshape(@view(params[(off + 1):(off + p * nn)]), n, n, p)
    off += p * nn
    Theta = reshape(@view(params[(off + 1):(off + q * nn)]), n, n, q)
    off += q * nn
    C = reshape(@view(params[(off + 1):(off + nn)]), n, n)
    off += nn
    dp = @view params[(off + 1):(off + n * k_dp)]
    return mu, Phi, Theta, C, dp
end

function _svarma_ml_obs_loglik(params::AbstractVector{Tθ}, Y::Matrix{T},
                               n::Int, p::Int, q::Int, dist::Symbol) where {Tθ<:Real,T<:AbstractFloat}
    k_dp = _svarma_ml_n_dp(dist)
    mu, Phi, Theta, C, dp = _svarma_ml_unpack(params, n, p, q, k_dp)
    U = _svarma_ml_ar_residuals(Y, mu, Phi)
    f_res = _svarma_ml_filter(U, Theta, C; tol=1e-7)
    shocks = f_res.shocks
    T_eff = size(shocks, 1)
    log_jac = f_res.log_jacobian

    ell = Vector{Tθ}(undef, T_eff)
    @inbounds for t in 1:T_eff
        acc = -Tθ(log_jac)
        for j in 1:n
            off = (j - 1) * k_dp
            acc += _svarma_ml_logpdf_1d(shocks[t, j], @view(dp[(off + 1):(off + k_dp)]), dist)
        end
        ell[t] = acc
    end
    return ell
end

function _svarma_ml_neg_loglik(params::AbstractVector{Tθ}, Y::Matrix{T},
                               n::Int, p::Int, q::Int, dist::Symbol,
                               target_m2::Int) where {Tθ<:Real,T<:AbstractFloat}
    k_dp = _svarma_ml_n_dp(dist)
    _, _, Theta, C, _ = _svarma_ml_unpack(params, n, p, q, k_dp)

    # Guard nonsingular C and unit-circle boundaries / regime containment at primal values
    C_0 = _svarma_ml_primal.(C)
    det_C0 = det(C_0)
    if !isfinite(det_C0) || abs(det_C0) <= T(1e-8)
        return Tθ(Inf)
    end
    if q > 0
        Theta_0 = _svarma_ml_primal.(Theta)
        F_0 = _svarma_companion(Array{typeof(Theta_0[1]),3}(Theta_0))
        eigs_0 = try
            eigvals(F_0)
        catch
            return Tθ(Inf)
        end
        if any(z -> !isfinite(abs(z)) || abs(abs(z) - 1.0) <= 1e-4, eigs_0)
            return Tθ(Inf)
        end
        m2_cur = count(z -> abs(z) > 1.0, eigs_0)
        if target_m2 >= 0 && m2_cur != target_m2
            return Tθ(Inf)
        end
    end

    ell = try
        _svarma_ml_obs_loglik(params, Y, n, p, q, dist)
    catch
        return Tθ(Inf)
    end
    total_ll = sum(ell)
    return isfinite(ForwardDiff.value(total_ll)) ? -total_ll : Tθ(Inf)
end

# =============================================================================
# Core Truncated ML / PML Estimator
# =============================================================================

function _svarma_ml_estimate(Y::Matrix{T}, p::Int, q::Int;
                             method::Symbol=:pml,
                             distribution::Symbol=:student_t,
                             k_iv::Int=p + 3,
                             se::Symbol=:auto,
                             max_iter::Int=500,
                             tol::Real=1e-6,
                             n_starts::Int=3,
                             varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                             shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    dist_canon = _svarma_ml_canonical_dist(distribution)
    se in (:auto, :hessian, :sandwich, :opg) ||
        throw(ArgumentError("se must be :auto, :hessian, :sandwich, or :opg (got :$se)"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))

    T_obs, n = size(Y)
    tol_T = T(tol)
    k_dp = _svarma_ml_n_dp(dist_canon)

    # 1. Obtain 2SLS-GMM initializer (adjust k_iv safely if T_obs is tight)
    k_iv_eff = k_iv
    if p >= 1
        max_k_safe = div(T_obs - q - 2, n) - 1
        if T_obs - max(p, q + k_iv_eff) <= 1 + n * k_iv_eff && max_k_safe >= p
            k_iv_eff = max(p, min(k_iv, max_k_safe))
        end
    end

    gmm_model = try
        _svarma_gmm_estimate(Y, p, q; k_iv=k_iv_eff, max_iter=min(max_iter, 300),
                             tol=max(tol_T, T(1e-6)), rng=rng).model
    catch
        nothing
    end

    # Fallback AR(p) OLS + fundamental Wold init if 2SLS-GMM could not run
    mu_base, Phi_base, Theta_base, C_base = if gmm_model !== nothing
        gmm_model.mu, gmm_model.Phi, gmm_model.Theta, gmm_model.C
    else
        mu_f = p == 0 ? vec(mean(Y; dims=1)) : zeros(T, n)
        Phi_f = zeros(T, n, n, p)
        if p >= 1
            v_fit = estimate_var(Y, p; check_stability=false)
            mu_f = Vector{T}(v_fit.B[1, :])
            A_list = extract_ar_coefficients(v_fit.B, n, p)
            for k in 1:p
                Phi_f[:, :, k] .= A_list[k]
            end
        end
        Z_f = _svarma_ar_residuals(Y, mu_f, Phi_f)
        Th_f, L_f = _svarma_gmm_fundamental_init(Z_f, q)
        (mu_f, Phi_f, Th_f, L_f)
    end

    # 2. Multi-regime basin seeds: Blaschke representations from GMM + fundamental JADE/ICA starts
    seed_pairs = Tuple{Vector{T},Array{T,3},Array{T,3},Matrix{T}}[]
    seen_sigs = Vector{T}[]

    function add_blaschke_family!(mu_s, Phi_s, Th_s, C_s)
        reps = try
            blaschke_representations(Th_s, C_s; tol=1e-6)
        catch
            [(Theta=Th_s, C=C_s)]
        end
        n_keep = min(length(reps), 32)
        for idx in 1:n_keep
            rep = reps[idx]
            Th_r = Array{T,3}(rep.Theta)
            C_r = Matrix{T}(rep.C)
            # Nudge eigenvalues slightly away from unit circle if borderline
            if q > 0
                eigs_r = eigvals(_svarma_companion(Th_r))
                if any(z -> abs(abs(z) - 1.0) <= 5e-4, eigs_r)
                    continue
                end
            end
            sig = vcat(vec(Th_r), vec(C_r))
            if !any(s -> maximum(abs, s .- sig) < T(1e-3), seen_sigs)
                push!(seen_sigs, sig)
                push!(seed_pairs, (Vector{T}(mu_s), Array{T,3}(Phi_s), Th_r, C_r))
            end
        end
    end

    add_blaschke_family!(mu_base, Phi_base, Theta_base, C_base)

    # Also add fundamental Wold + JADE/FastICA Blaschke family when distinct
    Z_base = _svarma_ar_residuals(Y, mu_base, Phi_base)
    Th_fund2, L_fund2 = _svarma_gmm_fundamental_init(Z_base, q)
    sh_fund2 = q == 0 ? Z_base : ssvarma_filter_shocks(Z_base, Th_fund2, L_fund2; tol=1e-6).shocks
    for R_rot in _svarma_gmm_ica_rotations(sh_fund2, rng)
        add_blaschke_family!(mu_base, Phi_base, Th_fund2, L_fund2 * R_rot)
    end

    # Extra Haar orthogonal C perturbations if n_starts > 1 and n >= 2
    if n >= 2 && n_starts > 1
        for s_extra in 2:n_starts
            Q_rand = generate_Q(n, T; rng=rng)
            add_blaschke_family!(mu_base, Phi_base, Th_fund2, L_fund2 * Q_rand)
        end
    end

    # 3. Explore each regime basin via Optim.LBFGS() + ForwardDiff gradients
    best_nll = T(Inf)
    best_params = T[]
    best_conv = false
    best_iters = 0

    opts = Optim.Options(iterations=max_iter, g_tol=tol_T, f_reltol=T(1e-12),
                         allow_f_increases=true, show_trace=false)

    for (mu0, Phi0, Th0, C0) in seed_pairs
        target_m2 = q == 0 ? 0 : count(z -> abs(z) > 1.0, eigvals(_svarma_companion(Th0)))
        sh0 = try
            _svarma_ml_filter(_svarma_ml_ar_residuals(Y, mu0, Phi0), Th0, C0; tol=1e-6).shocks
        catch
            continue
        end
        dp0 = _svarma_ml_fit_1d_dist(sh0, dist_canon)
        p0 = _svarma_ml_pack(mu0, Phi0, Th0, C0, dp0)

        neg_ll_fn = p_vec -> _svarma_ml_neg_loglik(p_vec, Y, n, p, q, dist_canon, target_m2)
        v0 = try
            T(neg_ll_fn(p0))
        catch
            T(Inf)
        end
        isfinite(v0) || continue

        res = try
            cfg = ForwardDiff.GradientConfig(neg_ll_fn, p0)
            g! = (G, x) -> ForwardDiff.gradient!(G, neg_ll_fn, x, cfg)
            Optim.optimize(neg_ll_fn, g!, p0, Optim.LBFGS(), opts)
        catch
            nothing
        end
        if res !== nothing
            val = T(Optim.minimum(res))
            if isfinite(val) && val < best_nll
                best_nll = val
                best_params = Vector{T}(Optim.minimizer(res))
                best_conv = Optim.converged(res)
                best_iters = Optim.iterations(res)
            end
        elseif v0 < best_nll
            best_nll = v0
            best_params = p0
            best_conv = false
            best_iters = 0
        end
    end

    if isempty(best_params)
        dp0 = zeros(T, n * k_dp)
        best_params = _svarma_ml_pack(mu_base, Phi_base, Theta_base, C_base, dp0)
        best_nll = T(_svarma_ml_neg_loglik(best_params, Y, n, p, q, dist_canon, -1))
    end

    # 4. Unpack and normalize by Assumption A.7
    mu_raw, Phi_raw, Theta_raw, C_raw, dp_raw = _svarma_ml_unpack(best_params, n, p, q, k_dp)
    mu_hat = Vector{T}(mu_raw)
    Phi_3d = Array{T,3}(Phi_raw)
    Theta_mat = Array{T,3}(Theta_raw)
    C_mat = Matrix{T}(C_raw)
    dp_opt = Vector{T}(dp_raw)

    C_norm, Theta_norm, perm, signs, _ = _svarma_normalize_a7(C_mat, Theta_mat)
    dp_norm, dp_display = _svarma_ml_extract_dist_params(dp_opt, signs, perm, n, dist_canon)
    p_norm = _svarma_ml_pack(mu_hat, Phi_3d, Theta_norm, C_norm, dp_norm)

    # Final filter & classification at normalized optimum
    f_res = ssvarma_filter_shocks(Y, mu_hat, Phi_3d, Theta_norm, C_norm; tol=1e-8)
    reg = fundamentalness_regime(Theta_norm, C_norm; tol=1e-8)
    loglik = -best_nll

    # 5. Covariance / Standard Errors: Hessian (:ml) or Huber-White Sandwich (:pml)
    eff_se = se === :auto ? (method === :ml ? :hessian : :sandwich) : se
    T_eff = size(f_res.shocks, 1)
    k_coef = n + p * n^2 + q * n^2 + n^2
    d_total = length(p_norm)

    obs_fn = p_vec -> _svarma_ml_obs_loglik(p_vec, Y, n, p, q, dist_canon)
    H_inv = if eff_se in (:hessian, :sandwich)
        H_raw = try
            ForwardDiff.hessian(p_vec -> -sum(obs_fn(p_vec)), p_norm)
        catch
            zeros(T, d_total, d_total)
        end
        isempty(H_raw) || all(iszero, H_raw) ? zeros(T, d_total, d_total) : _spd_inv(Matrix{T}(H_raw))
    else
        zeros(T, d_total, d_total)
    end

    J_meat = if eff_se in (:sandwich, :opg)
        G_t = try
            ForwardDiff.jacobian(obs_fn, p_norm)
        catch
            zeros(T, T_eff, d_total)
        end
        if all(iszero, G_t) || any(!isfinite, G_t)
            zeros(T, d_total, d_total)
        else
            active_cols = [j for j in 1:d_total if sum(abs2, @view(G_t[1:(end - 1), j])) > eps(T)]
            bw = if isempty(active_cols)
                2 * q + 2
            else
                min(max(2 * q + 2, optimal_bandwidth_nw(Matrix{T}(@view G_t[:, active_cols]); kernel=:bartlett)), 6)
            end
            Matrix{T}(lrcov(Matrix{T}(G_t); kernel=:bartlett, bandwidth=bw, demean=false)) .* T(T_eff)
        end
    else
        zeros(T, d_total, d_total)
    end

    vcov_full = if eff_se === :hessian
        H_inv
    elseif eff_se === :sandwich
        V = H_inv * J_meat * H_inv
        Matrix{T}((V .+ transpose(V)) ./ T(2))
    else
        all(iszero, J_meat) ? zeros(T, d_total, d_total) : _spd_inv(J_meat)
    end

    vcov_coef = Matrix{T}(vcov_full[1:k_coef, 1:k_coef])
    vcov_coef = Matrix{T}((vcov_coef .+ transpose(vcov_coef)) ./ T(2))
    se_coef = T[sqrt(max(vcov_coef[i, i], zero(T))) for i in 1:k_coef]

    n_params_tot = d_total
    aic_val = -T(2) * loglik + T(2) * T(n_params_tot)
    bic_val = -T(2) * loglik + log(T(T_eff)) * T(n_params_tot)

    re_vec = T[T(real(z)) for z in reg.eigenvalues]
    im_vec = T[T(imag(z)) for z in reg.eigenvalues]
    dists_vec = fill(distribution === :pml ? :pearson_iv : dist_canon, n)
    vn_vec = varnames === nothing ? default_var_names(n) : String[string(v) for v in varnames]
    sn_vec = shock_names === nothing ? default_shock_names(n) : String[string(s) for s in shock_names]

    return SVARMAModel{T}(
        Y, p, q, mu_hat, Phi_3d, Theta_norm, C_norm, f_res.shocks,
        re_vec, im_vec, reg.regime, reg.gmr_region, dists_vec, dp_display,
        loglik, aic_val, bic_val, vn_vec, sn_vec, best_conv, method,
        vcov_coef, se_coef,
    )
end

# =============================================================================
# Public API: estimate_ssvarma
# =============================================================================

"""
    estimate_ssvarma(Y::AbstractMatrix, p::Int, q::Int=1;
                     method::Symbol=:pml, distribution::Symbol=:student_t,
                     k_iv::Int=p+3, se::Symbol=:auto, max_iter::Int=500,
                     tol::Real=1e-6, n_starts::Int=3,
                     varnames=nothing, shock_names=nothing,
                     rng=Random.default_rng(), seed=nothing) -> SVARMAModel{T}

Estimate a non-Gaussian Strong Structural VARMA(p, q) model following
Gouriéroux, Monfort & Renne (2020):

```math
Y_t = \\mu + \\sum_{k=1}^{p} \\Phi_k Y_{t-k} + \\varepsilon_t + \\sum_{j=1}^{q} \\Theta_j \\varepsilon_{t-j}, \\qquad \\varepsilon_t = C \\eta_t
```

where:
- `Y_t` is the ``n \\times 1`` vector of endogenous observables at time ``t``
- `\\mu` is the ``n \\times 1`` intercept vector
- `\\Phi_k` are the ``n \\times n`` autoregressive lag matrices (``k = 1, \\ldots, p``)
- `\\Theta_j` are the ``n \\times n`` moving-average lag matrices (``j = 1, \\ldots, q``)
- `C` is the ``n \\times n`` structural impact matrix normalized by GMR Assumption A.7
- `\\eta_t` are mutually and serially independent zero-mean unit-variance non-Gaussian shocks

Two estimation procedures (`method`) are supported:
- `:pml` (default) or `:ml` — maximizes the GMR (2020, eq. 4.16) truncated log-likelihood
  with non-fundamental Schur Jacobian ``-T_{\\text{eff}} (\\log |\\det C| + \\sum_{|\\lambda_i(\\Theta)| > 1} \\log |\\lambda_i(\\Theta)|)``
  across every Blaschke root-flipped regime basin (`blaschke_representations`), initialized
  from 2SLS-GMM and marginal shock MLE.
- `:gmm` — two-step semi-parametric 2SLS-GMM estimator (GMR §4.2): Step 1 estimates
  ``(\\mu, \\Phi_1, \\ldots, \\Phi_p)`` by 2SLS with lagged instruments ``Y_{t-q-1}, \\ldots, Y_{t-q-k_{\\text{iv}}}``;
  Step 2 estimates ``(C_0, \\ldots, C_q, \\kappa_3, \\kappa_4)`` from order-2/3/4 cumulant moments
  across non-symmetric direction pairs with exact first-step-error-adjusted Newey–West HAC
  optimal weighting ``\\Omega^* = Q_0^{-1}`` and nonparametric KDE shock densities.

# Keyword Arguments
- `method::Symbol=:pml` — `:pml`, `:ml`, or `:gmm`
- `distribution::Symbol=:student_t` — marginal density for `:pml`/`:ml` (`:student_t`, `:mixture_normal_asym`, `:mixture_normal`, `:pearson_iv`, `:pml`, `:skew_normal`)
- `k_iv::Int=p+3` — number of lagged instrument lags for Step 1 2SLS
- `se::Symbol=:auto` — standard error type (`:auto` selects `:sandwich` for `:pml`, `:hessian` for `:ml`, first-step-adjusted GMM covariance for `:gmm`; `:opg` also supported)
- `max_iter::Int=500` — maximum optimizer iterations per regime start
- `tol::Real=1e-6` — gradient convergence tolerance
- `n_starts::Int=3` — additional orthogonal rotations per regime for multi-start search
- `varnames` — optional length-`n` vector of variable names
- `shock_names` — optional length-`n` vector of structural shock names
- `rng::AbstractRNG=Random.default_rng()` — random number generator
- `seed::Union{Integer,Nothing}=nothing` — optional seed overriding `rng`

# References
- Gouriéroux, C., Monfort, A., & Renne, J.-P. (2020). Identification and Estimation in
  Non-Fundamental Structural VARMA Models. *Review of Economic Studies*, 87(4), 1915–1953.
"""
function estimate_ssvarma(Y::AbstractMatrix{T}, p::Int, q::Int=1;
                          method::Symbol=:pml,
                          distribution::Symbol=:student_t,
                          k_iv::Int=p + 3,
                          se::Symbol=:auto,
                          max_iter::Int=500,
                          tol::Real=1e-6,
                          n_starts::Int=3,
                          varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                          shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                          rng::AbstractRNG=Random.default_rng(),
                          seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(Y, "Y")
    p >= 0 || throw(ArgumentError("AR order p must be >= 0, got $p"))
    q >= 0 || throw(ArgumentError("MA order q must be >= 0, got $q"))
    k_iv >= p || throw(ArgumentError("k_iv ($k_iv) must be >= p ($p)"))
    method in (:pml, :ml, :gmm) ||
        throw(ArgumentError("method must be :pml, :ml, or :gmm (got :$method)"))

    T_obs, n = size(Y)
    T_min = max(p + q + 4, p + n * p + 3)
    T_obs > T_min || throw(ArgumentError(
        "Insufficient observations (T=$T_obs) for SVARMA($p, $q) with n=$n (need T > $T_min)"))

    rng_eff = _resolve_repro_rng(rng, seed)
    Y_mat = Matrix{T}(Y)

    if method === :gmm
        return _svarma_gmm_estimate(Y_mat, p, q; k_iv=k_iv, max_iter=max_iter, tol=tol,
                                    varnames=varnames, shock_names=shock_names,
                                    rng=rng_eff).model
    end

    return _svarma_ml_estimate(Y_mat, p, q; method=method, distribution=distribution,
                               k_iv=k_iv, se=se, max_iter=max_iter, tol=tol,
                               n_starts=n_starts, varnames=varnames,
                               shock_names=shock_names, rng=rng_eff)
end

function estimate_ssvarma(Y::AbstractMatrix{<:Real}, p::Int, q::Int=1; kwargs...)
    return estimate_ssvarma(Float64.(Y), p, q; kwargs...)
end
