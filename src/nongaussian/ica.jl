# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
ICA-based SVAR identification: FastICA, JADE, SOBI, dCov, HSIC.

Nonparametric identification via non-Gaussianity (Lewis 2025, Section 4).
Recovers structural shocks under the assumption that at most one shock is Gaussian
(Darmois-Skitovich theorem; Comon 1994). The unmixing matrix W satisfies
ε_t = W u_t where ε_t are independent non-Gaussian shocks.

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
- Hyvärinen, A. (1999). "Fast and robust fixed-point algorithms for independent component analysis."
- Cardoso, J.-F. & Souloumiac, A. (1993). "Blind beamforming for non-Gaussian signals."
- Belouchrani, A. et al. (1997). "A blind source separation technique using second-order statistics."
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of non-Gaussian SVAR."
- Székely, G. J., Rizzo, M. L. & Bakirov, N. K. (2007). "Measuring and testing dependence by correlation of distances."
- Gretton, A. et al. (2005). "Measuring statistical dependence with Hilbert-Schmidt norms."
"""

using LinearAlgebra, Statistics, Random
import Optim

# =============================================================================
# Result Type
# =============================================================================

"""
    ICASVARResult{T} <: AbstractNonGaussianSVAR

Result from ICA-based SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n): u_t = B₀ ε_t
- `W::Matrix{T}` — unmixing matrix (n × n): ε_t = W u_t
- `Q::Matrix{T}` — rotation matrix for `compute_Q` integration
- `shocks::Matrix{T}` — recovered structural shocks (T_eff × n)
- `method::Symbol` — `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`, or `:cvm`
- `converged::Bool`
- `iterations::Int`
- `objective::T` — final objective value
- `shock_names::Vector{String}` — labels for the n structural shocks
- `vcov::Matrix{T}` — plug-in asymptotic covariance of ``\\operatorname{vec}(B_0)`` (n² × n²)
- `se::Matrix{T}` — asymptotic standard errors for `B0` (n × n)
"""
struct ICASVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    W::Matrix{T}
    Q::Matrix{T}
    shocks::Matrix{T}
    method::Symbol
    converged::Bool
    iterations::Int
    objective::T
    shock_names::Vector{String}
    vcov::Matrix{T}
    se::Matrix{T}
end

function ICASVARResult{T}(B0, W, Q, shocks, method, converged, iterations,
                          objective, shock_names::Vector{String}) where {T<:AbstractFloat}
    n1, n2 = size(B0)
    ICASVARResult{T}(Matrix{T}(B0), Matrix{T}(W), Matrix{T}(Q), Matrix{T}(shocks),
                     method, converged, iterations, T(objective), shock_names,
                     zeros(T, 0, 0), fill(T(NaN), n1, n2))
end

function ICASVARResult(B0, W, Q, shocks, method, converged, iterations,
                       objective, shock_names::Vector{String})
    T = float(eltype(B0))
    ICASVARResult{T}(B0, W, Q, shocks, method, converged, iterations, objective, shock_names)
end

function ICASVARResult{T}(B0, W, Q, shocks, method, converged, iterations,
                          objective) where {T<:AbstractFloat}
    n = size(B0, 2)
    ICASVARResult{T}(B0, W, Q, shocks, method, converged, iterations, T(objective),
                     _default_shock_names(n))
end

function ICASVARResult(B0, W, Q, shocks, method, converged, iterations, objective)
    T = float(eltype(B0))
    ICASVARResult{T}(B0, W, Q, shocks, method, converged, iterations, objective)
end

StatsAPI.vcov(r::ICASVARResult) = r.vcov
function StatsAPI.stderror(r::ICASVARResult{T}) where {T}
    isempty(r.vcov) ? vec(r.se) : T[sqrt(max(r.vcov[i, i], zero(T))) for i in 1:size(r.vcov, 1)]
end

function _transform_statid_vcov(result::ICASVARResult{T}, perm::Vector{Int},
                                signs::Vector{Int}, n::Int, Q_new) where {T<:AbstractFloat}
    V = result.vcov
    size(V) == (n * n, n * n) || return V
    P_signed = zeros(T, n, n)
    @inbounds for j in 1:n
        P_signed[perm[j], j] = T(signs[j])
    end
    A = kron(transpose(P_signed), Matrix{T}(I, n, n))
    Matrix{T}(Symmetric(A * V * transpose(A)))
end

function Base.show(io::IO, r::ICASVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Method"     string(r.method);
        "Variables"  n;
        "Shocks"     join(r.shock_names, ", ");
        "Converged"  r.converged ? "Yes" : "No";
        "Iterations" r.iterations;
        "Objective"  _fmt(r.objective)
    ]
    _pretty_table(io, spec;
        title = "ICA-SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    se_disp = size(r.se) == (n, n) ? r.se : fill(T(NaN), n, n)
    _show_B0_ses(io, r.B0, se_disp)
end

"""Compute GMR (2017) / Keweloh plug-in GMM sandwich covariance for ICA B₀ (n² × n²)."""
function _ica_cvm_sandwich_vcov(B0::Matrix{T}, U::AbstractMatrix{T},
                                 method::Symbol) where {T<:AbstractFloat}
    T_eff, n = size(U)
    if T_eff < 4 || n < 1
        return zeros(T, n * n, n * n), fill(T(NaN), n, n)
    end
    U_mat = Matrix{T}(U)
    if n == 1
        w1 = one(T) / max(abs(B0[1, 1]), sqrt(eps(T)))
        e1 = U_mat[:, 1] .* w1
        s2 = e1 .^ 2 .- one(T)
        v11 = max(mean(abs2, s2) * B0[1, 1]^2 / T(4 * T_eff), sqrt(eps(T)))
        return reshape(T[v11], 1, 1), reshape(T[sqrt(v11)], 1, 1)
    end

    use_lag = (method === :sobi && T_eff > 12)
    T_m = use_lag ? (T_eff - 1) : T_eff
    n_cov = (n * (n + 1)) ÷ 2
    n_pairs = n * (n - 1)
    K_mom = n_cov + 2 * n_pairs + (use_lag ? n_pairs : 0)

    function moment_matrix(b::AbstractVector{Tb}) where {Tb<:Real}
        B_m = reshape(b, n, n)
        W_m = Matrix{Tb}(robust_inv(B_m; silent=true))
        E_m = Tb.(U_mat) * transpose(W_m)

        # Compute tanh(E_m) and centered τ_t = tanh(ε_t) - mean(tanh(ε))
        E_off = use_lag ? @view(E_m[2:T_eff, :]) : E_m
        T_rows = size(E_off, 1)
        Tau = tanh.(E_off)
        tau_bar = vec(mean(Tau; dims=1))
        Tau_c = Tau .- transpose(tau_bar)

        G = Matrix{Tb}(undef, T_rows, K_mom)
        col = 0
        # 1. Second-moment identification: ε_i ε_j - δ_ij
        @inbounds for i in 1:n, j in i:n
            col += 1
            d_ij = i == j ? one(Tb) : zero(Tb)
            for t in 1:T_rows
                G[t, col] = E_off[t, i] * E_off[t, j] - d_ij
            end
        end
        # 2. Bounded odd non-Gaussian independence contrast: (tanh(ε_i) - μ_τ,i) ε_j
        @inbounds for i in 1:n, j in 1:n
            i == j && continue
            col += 1
            for t in 1:T_rows
                G[t, col] = Tau_c[t, i] * E_off[t, j]
            end
        end
        # 3. Skewness / asymmetric cross-moments: (ε_i² - 1)(tanh(ε_j) - μ_τ,j)
        @inbounds for i in 1:n, j in 1:n
            i == j && continue
            col += 1
            for t in 1:T_rows
                G[t, col] = (E_off[t, i]^2 - one(Tb)) * Tau_c[t, j]
            end
        end
        # 4. Lagged cross-autocovariance for SOBI temporal identification
        if use_lag
            E_lag = @view E_m[1:(T_eff - 1), :]
            @inbounds for i in 1:n, j in 1:n
                i == j && continue
                col += 1
                for t in 1:T_rows
                    G[t, col] = E_off[t, i] * E_lag[t, j]
                end
            end
        end
        return G
    end

    b_hat = vec(Matrix{T}(B0))
    G0 = moment_matrix(b_hat)
    G0c = G0 .- mean(G0; dims=1)
    Omega_raw = Matrix{T}(Symmetric((transpose(G0c) * G0c) / T(T_m)))
    # Equilibrated SPD inverse so differing moment scales never drop modes
    d_o = T[sqrt(max(Omega_raw[k, k], sqrt(eps(T)))) for k in 1:K_mom]
    d_o_inv = Diagonal(one(T) ./ d_o)
    Omega_inv = Matrix{T}(Symmetric(d_o_inv * _spd_inv(Matrix{T}(d_o_inv * Omega_raw * d_o_inv)) * d_o_inv))

    D = Matrix{T}(ForwardDiff.jacobian(b -> vec(mean(moment_matrix(b); dims=1)), b_hat))
    H_info = Matrix{T}(Symmetric(transpose(D) * Omega_inv * D))

    F_H = eigen(Symmetric(H_info))
    eig_max = maximum(F_H.values; init=zero(T))
    eig_floor = max(T(1e-3), T(1e-8) * eig_max)
    inv_eigs = T[v > eig_floor ? one(T) / v : one(T) / eig_floor for v in F_H.values]
    V_mat = (F_H.vectors * Diagonal(inv_eigs) * transpose(F_H.vectors)) ./ T(T_m)
    V_sym = Matrix{T}((V_mat .+ transpose(V_mat)) ./ T(2))

    se_mat = reshape(T[sqrt(max(V_sym[i, i], zero(T))) for i in 1:(n * n)], n, n)
    _finite_se!(se_mat)
    return V_sym, se_mat
end

function _ica_cvm_finalize_result(B0::AbstractMatrix{T}, W_full::AbstractMatrix{T},
                                  Q::AbstractMatrix{T}, shocks::AbstractMatrix{T},
                                  method::Symbol, converged::Bool, iterations::Int,
                                  objective::T, U::AbstractMatrix{T}) where {T<:AbstractFloat}
    B0_m = Matrix{T}(B0)
    W_m = Matrix{T}(W_full)
    Q_m = Matrix{T}(Q)
    S_m = Matrix{T}(shocks)
    vcov_B0, se_B0 = try
        _ica_cvm_sandwich_vcov(B0_m, U, method)
    catch
        n = size(B0_m, 1)
        zeros(T, 0, 0), fill(T(NaN), n, n)
    end
    return ICASVARResult{T}(B0_m, W_m, Q_m, S_m, method, converged, iterations,
                            objective, _default_shock_names(size(B0_m, 2)), vcov_B0, se_B0)
end

# =============================================================================
# Contrast Functions for FastICA
# =============================================================================

"""Log-cosh contrast: G(u) = log(cosh(u)), g(u) = tanh(u), g'(u) = 1 - tanh²(u)."""
function _contrast_logcosh(u::T) where {T<:AbstractFloat}
    t = tanh(u)
    (log(cosh(u)), t, one(T) - t^2)
end

"""Exponential contrast: G(u) = -exp(-u²/2), g(u) = u·exp(-u²/2), g'(u) = (1-u²)·exp(-u²/2)."""
function _contrast_exp(u::T) where {T<:AbstractFloat}
    e = exp(-u^2 / 2)
    (-e, u * e, (one(T) - u^2) * e)
end

"""Kurtosis contrast: G(u) = u⁴/4, g(u) = u³, g'(u) = 3u²."""
function _contrast_kurtosis(u::T) where {T<:AbstractFloat}
    (u^4 / 4, u^3, 3 * u^2)
end

_get_contrast(sym::Symbol) = sym == :logcosh ? _contrast_logcosh :
                              sym == :exp ? _contrast_exp :
                              sym == :kurtosis ? _contrast_kurtosis :
                              throw(ArgumentError("Unknown contrast: $sym"))

# =============================================================================
# FastICA Algorithms
# =============================================================================

"""FastICA deflation: extract components one at a time."""
function _fastica_deflation(Z::Matrix{T}, n::Int; contrast::Symbol=:logcosh,
                            max_iter::Int=200, tol::T=T(1e-6),
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    g_func = _get_contrast(contrast)
    W = zeros(T, n, n)
    total_iter = 0

    for p in 1:n
        w = randn(rng, T, n)
        w /= norm(w)

        for iter in 1:max_iter
            total_iter += 1
            # w_new = E[Z g(w'Z)] - E[g'(w'Z)] w
            proj = Z * w  # T_obs × 1
            g_vals = similar(proj)
            gp_vals = similar(proj)
            @inbounds for i in 1:T_obs
                _, g, gp = g_func(proj[i])
                g_vals[i] = g
                gp_vals[i] = gp
            end

            w_new = (Z' * g_vals) / T_obs - mean(gp_vals) * w

            # Orthogonalize against previous components
            for k in 1:(p-1)
                w_new -= dot(w_new, @view(W[k, :])) * @view(W[k, :])
            end
            w_new /= norm(w_new)

            converged = abs(abs(dot(w, w_new)) - one(T)) < tol
            w = w_new
            converged && break
        end
        W[p, :] = w
    end
    W, total_iter
end

"""FastICA symmetric: extract all components simultaneously."""
function _fastica_symmetric(Z::Matrix{T}, n::Int; contrast::Symbol=:logcosh,
                            max_iter::Int=200, tol::T=T(1e-6),
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    g_func = _get_contrast(contrast)

    # Initialize W with random orthogonal matrix
    W = Matrix{T}(qr(randn(rng, T, n, n)).Q)
    total_iter = 0

    for iter in 1:max_iter
        total_iter = iter
        W_old = copy(W)

        for p in 1:n
            w = @view W[p, :]
            proj = Z * w
            g_vals = similar(proj)
            gp_vals = similar(proj)
            @inbounds for i in 1:T_obs
                _, g, gp = g_func(proj[i])
                g_vals[i] = g
                gp_vals[i] = gp
            end
            W[p, :] = (Z' * g_vals) / T_obs - mean(gp_vals) * w
        end

        # Symmetric decorrelation: W = (W W')^{-1/2} W
        F = eigen(Symmetric(W * W'))
        D_inv_sqrt = Diagonal(one(T) ./ sqrt.(max.(F.values, eps(T))))
        W = F.vectors * D_inv_sqrt * F.vectors' * W

        # Check convergence
        max_change = maximum(abs.(abs.(diag(W * W_old')) .- one(T)))
        max_change < tol && break
    end
    W, total_iter
end

# =============================================================================
# Public API: FastICA
# =============================================================================

"""
    identify_fastica(model::VARModel; contrast=:logcosh, approach=:deflation,
                     max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR via FastICA (Hyvärinen 1999).

Recovers independent non-Gaussian structural shocks by maximizing non-Gaussianity
of the recovered sources.

Arguments:
- `contrast` — non-Gaussianity measure: `:logcosh` (default, robust), `:exp`, `:kurtosis`
- `approach` — `:deflation` (one-by-one) or `:symmetric` (simultaneous)
- `max_iter` — maximum iterations per component
- `tol` — convergence tolerance

**Reference**: Hyvärinen (1999)
"""
function identify_fastica(model::VARModel{T}; contrast::Symbol=:logcosh,
                          approach::Symbol=:deflation, max_iter::Int=200,
                          tol::T=T(1e-6),
                          seed::Union{Integer,Nothing}=nothing,
                          rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    if approach == :deflation
        W_ica, iters = _fastica_deflation(Z, n; contrast=contrast, max_iter=max_iter, tol=tol, rng=rng)
    else
        W_ica, iters = _fastica_symmetric(Z, n; contrast=contrast, max_iter=max_iter, tol=tol, rng=rng)
    end

    # Full unmixing in original space
    W_full = W_ica * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Compute objective (sum of negentropies)
    g_func = _get_contrast(contrast)
    obj = zero(T)
    for j in 1:n
        s = @view shocks[:, j]
        obj += abs(mean(x -> g_func(x)[1], s) - mean(x -> g_func(x)[1], randn(rng, T, length(s))))
    end

    _ica_cvm_finalize_result(B0, W_full, Q, shocks, :fastica, iters < max_iter * n, iters, obj, model.U)
end

# =============================================================================
# JADE: Joint Approximate Diagonalization of Eigenmatrices
# =============================================================================

"""Compute fourth-order cumulant matrices for JADE."""
function _jade_cumulant_matrices(Z::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(Z)
    matrices = Matrix{T}[]

    for i in 1:n, j in i:n
        C = zeros(T, n, n)
        @inbounds for t in 1:T_obs
            z = @view Z[t, :]
            C .+= z[i] * z[j] * (z * z') / T_obs
        end
        # Subtract Gaussian cumulant: δ_{ij} I + e_i e_j' + e_j e_i'
        C[i, j] -= one(T)
        C[j, i] -= one(T)
        for k in 1:n
            C[k, k] -= (i == j ? one(T) : zero(T))
        end
        push!(matrices, C)
    end
    matrices
end

"""Joint diagonalization via Jacobi rotations (Cardoso & Souloumiac 1993)."""
function _joint_diagonalization(matrices::Vector{Matrix{T}}; max_iter::Int=100,
                                tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = size(matrices[1], 1)
    V = Matrix{T}(I, n, n)
    M = [copy(m) for m in matrices]

    total_iter = 0
    for iter in 1:max_iter
        total_iter = iter
        max_rotation = zero(T)

        for p in 1:n-1, q in (p+1):n
            # Compute optimal Givens rotation angle
            h = zeros(T, 3)
            for m in M
                ip, iq = m[p, p] - m[q, q], m[p, q] + m[q, p]
                h[1] += ip * iq
                h[2] += ip^2 - iq^2
                h[3] += 2 * ip * iq  # redundant with h[1] but keeps notation clear
            end

            # Cardoso–Souloumiac (1993) Jacobi angle: quarter-angle, not half.
            # atan2(2·h₁, h₂)/2 is a fixed point of the off-diagonal criterion
            # (no rotation); the joint-diagonalization optimum is atan2/4.
            theta = atan(2 * h[1], h[2]) / 4

            # Apply rotation
            c, s = cos(theta), sin(theta)
            if abs(s) > tol
                max_rotation = max(max_rotation, abs(s))
                for m in M
                    # Rotate rows
                    for k in 1:n
                        a, b = m[p, k], m[q, k]
                        m[p, k] = c * a + s * b
                        m[q, k] = -s * a + c * b
                    end
                    # Rotate columns
                    for k in 1:n
                        a, b = m[k, p], m[k, q]
                        m[k, p] = c * a + s * b
                        m[k, q] = -s * a + c * b
                    end
                end

                # Update V
                for k in 1:n
                    a, b = V[k, p], V[k, q]
                    V[k, p] = c * a + s * b
                    V[k, q] = -s * a + c * b
                end
            end
        end

        max_rotation < tol && break
    end

    V, total_iter
end

"""
    identify_jade(model::VARModel; max_iter=100, tol=1e-6) -> ICASVARResult

Identify SVAR via JADE (Joint Approximate Diagonalization of Eigenmatrices).

Uses fourth-order cumulant matrices and joint diagonalization via Jacobi rotations.

**Reference**: Cardoso & Souloumiac (1993)
"""
function identify_jade(model::VARModel{T}; max_iter::Int=100,
                        tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    cum_matrices = _jade_cumulant_matrices(Z)
    V, iters = _joint_diagonalization(cum_matrices; max_iter=max_iter, tol=tol)

    W_full = V' * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Objective: sum of squared off-diagonals of rotated cumulant matrices
    obj = zero(T)
    for m in cum_matrices
        Rm = V' * m * V
        for i in 1:n, j in 1:n
            i != j && (obj += Rm[i, j]^2)
        end
    end

    _ica_cvm_finalize_result(B0, W_full, Q, shocks, :jade, iters < max_iter, iters, obj, model.U)
end

# =============================================================================
# SOBI: Second-Order Blind Identification
# =============================================================================

"""Compute autocovariance matrix at lag τ."""
function _sobi_autocovariance(Z::Matrix{T}, lag::Int) where {T<:AbstractFloat}
    T_obs, n = size(Z)
    R = zeros(T, n, n)
    T_eff = T_obs - lag
    @inbounds for t in 1:T_eff
        z_t = @view Z[t + lag, :]
        z_0 = @view Z[t, :]
        R .+= z_t * z_0'
    end
    R / T_eff
end

"""
    identify_sobi(model::VARModel; lags=1:12, max_iter=100, tol=1e-6) -> ICASVARResult

Identify SVAR via SOBI (Second-Order Blind Identification).

Uses autocovariance matrices at multiple lags and joint diagonalization.
Exploits temporal structure rather than higher-order statistics.

**Reference**: Belouchrani et al. (1997)
"""
function identify_sobi(model::VARModel{T}; lags::AbstractRange=1:12,
                        max_iter::Int=100, tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    auto_covs = [_sobi_autocovariance(Z, lag) for lag in lags]
    V, iters = _joint_diagonalization(auto_covs; max_iter=max_iter, tol=tol)

    W_full = V' * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    # Objective: sum of squared off-diagonals
    obj = zero(T)
    for m in auto_covs
        Rm = V' * m * V
        for i in 1:n, j in 1:n
            i != j && (obj += Rm[i, j]^2)
        end
    end

    _ica_cvm_finalize_result(B0, W_full, Q, shocks, :sobi, iters < max_iter, iters, obj, model.U)
end

# =============================================================================
# Distance Covariance
# =============================================================================

"""
Compute distance covariance between two vectors, filling the preallocated
T_obs x T_obs scratch buffers `A`, `B`, `C` in place. Arithmetic is identical
to the allocating form (same fill values, same `mean`/`sum` reduction paths),
so results are bit-for-bit unchanged while the objective loop reuses one set
of buffers across all pairs and evaluations.
"""
function _distance_covariance!(A::Matrix{T}, B::Matrix{T}, C::Matrix{T},
                               x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)

    # Compute distance matrices (in place)
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = abs(x[i] - x[j])
        B[i, j] = abs(y[i] - y[j])
    end

    # Double-center
    A_row = mean(A, dims=2)
    A_col = mean(A, dims=1)
    A_grand = mean(A)
    B_row = mean(B, dims=2)
    B_col = mean(B, dims=1)
    B_grand = mean(B)

    @inbounds for i in 1:n, j in 1:n
        A[i, j] = A[i, j] - A_row[i] - A_col[j] + A_grand
        B[i, j] = B[i, j] - B_row[i] - B_col[j] + B_grand
    end

    # dCov² = (1/n²) Σᵢⱼ Aᵢⱼ Bᵢⱼ
    C .= A .* B
    dcov2 = sum(C) / n^2
    max(dcov2, zero(T))
end

"""Compute distance covariance between two vectors."""
function _distance_covariance(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    A = Matrix{T}(undef, n, n)
    B = Matrix{T}(undef, n, n)
    C = Matrix{T}(undef, n, n)
    _distance_covariance!(A, B, C, x, y)
end

"""Objective: sum of pairwise distance covariances (to minimize). The buffered
method reuses the scratch matrices `A`, `B`, `C` across pairs."""
function _dcov_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int,
                         A::Matrix{T}, B::Matrix{T}, C::Matrix{T}) where {T<:AbstractFloat}
    Q = _givens_to_orthogonal(angles, n)
    S = Z * Q'  # rotated sources

    obj = zero(T)
    for i in 1:n-1, j in (i+1):n
        obj += _distance_covariance!(A, B, C, @view(S[:, i]), @view(S[:, j]))
    end
    obj
end

function _dcov_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    A = Matrix{T}(undef, T_obs, T_obs)
    B = Matrix{T}(undef, T_obs, T_obs)
    C = Matrix{T}(undef, T_obs, T_obs)
    _dcov_objective(angles, Z, n, A, B, C)
end

"""
    identify_dcov(model::VARModel; max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR by minimizing pairwise distance covariance between recovered shocks.

Distance covariance (Székely et al. 2007) is zero iff the variables are independent,
making it a natural criterion for ICA.

**Reference**: Matteson & Tsay (2017)
"""
function identify_dcov(model::VARModel{T}; max_iter::Int=200,
                        tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    n_angles = n * (n - 1) ÷ 2
    angles0 = zeros(T, n_angles)

    # Preallocate the T_obs x T_obs scratch once; reused by every objective evaluation
    T_obs = size(Z, 1)
    Abuf = Matrix{T}(undef, T_obs, T_obs)
    Bbuf = Matrix{T}(undef, T_obs, T_obs)
    Cbuf = Matrix{T}(undef, T_obs, T_obs)
    result = Optim.optimize(a -> _dcov_objective(a, Z, n, Abuf, Bbuf, Cbuf), angles0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, g_tol=tol))

    Q_opt = _givens_to_orthogonal(Optim.minimizer(result), n)
    W_full = Q_opt * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    _ica_cvm_finalize_result(B0, W_full, Q, shocks, :dcov, Optim.converged(result),
                             Optim.iterations(result), T(Optim.minimum(result)), model.U)
end

# =============================================================================
# HSIC (Hilbert-Schmidt Independence Criterion)
# =============================================================================

"""
Compute the HSIC statistic with preallocated scratch. `H` is the (parameter-
invariant) centering matrix `I - ones(n,n)/n`, hoisted out of the objective
loop; `K`, `L`, `Kc`, `Lc`, `tmp`, `P` are reusable n x n buffers. The
products are evaluated with the same `mul!` calls/associations as the
allocating form `H*K*H` / `tr(Kc*Lc)`, so results are bit-for-bit unchanged.
"""
function _hsic_statistic!(K::Matrix{T}, L::Matrix{T}, Kc::Matrix{T}, Lc::Matrix{T},
                          tmp::Matrix{T}, P::Matrix{T}, H::Matrix{T},
                          x::AbstractVector{T}, y::AbstractVector{T};
                          sigma::T=T(1.0)) where {T<:AbstractFloat}
    n = length(x)
    s2 = 2 * sigma^2

    # Gaussian kernel matrices (in place)
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = exp(-(x[i] - x[j])^2 / s2)
        L[i, j] = exp(-(y[i] - y[j])^2 / s2)
    end

    # Center kernel matrices: Kc = (H*K)*H, Lc = (H*L)*H
    mul!(tmp, H, K)
    mul!(Kc, tmp, H)
    mul!(tmp, H, L)
    mul!(Lc, tmp, H)

    mul!(P, Kc, Lc)
    tr(P) / (n - 1)^2
end

"""Compute HSIC statistic between two vectors using Gaussian kernel."""
function _hsic_statistic(x::AbstractVector{T}, y::AbstractVector{T};
                          sigma::T=T(1.0)) where {T<:AbstractFloat}
    n = length(x)
    H = I - ones(T, n, n) / n
    K = Matrix{T}(undef, n, n)
    L = Matrix{T}(undef, n, n)
    Kc = Matrix{T}(undef, n, n)
    Lc = Matrix{T}(undef, n, n)
    tmp = Matrix{T}(undef, n, n)
    P = Matrix{T}(undef, n, n)
    _hsic_statistic!(K, L, Kc, Lc, tmp, P, H, x, y; sigma=sigma)
end

"""Objective: sum of pairwise HSIC (to minimize). The buffered method reuses
the hoisted centering matrix `H` and six n x n scratch buffers across pairs."""
function _hsic_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int,
                         K::Matrix{T}, L::Matrix{T}, Kc::Matrix{T}, Lc::Matrix{T},
                         tmp::Matrix{T}, P::Matrix{T}, H::Matrix{T};
                         sigma::T=T(1.0)) where {T<:AbstractFloat}
    Q = _givens_to_orthogonal(angles, n)
    S = Z * Q'

    obj = zero(T)
    for i in 1:n-1, j in (i+1):n
        obj += _hsic_statistic!(K, L, Kc, Lc, tmp, P, H,
                                @view(S[:, i]), @view(S[:, j]); sigma=sigma)
    end
    obj
end

function _hsic_objective(angles::AbstractVector{T}, Z::Matrix{T}, n::Int;
                          sigma::T=T(1.0)) where {T<:AbstractFloat}
    T_obs = size(Z, 1)
    H = I - ones(T, T_obs, T_obs) / T_obs
    bufs = ntuple(_ -> Matrix{T}(undef, T_obs, T_obs), 6)
    _hsic_objective(angles, Z, n, bufs..., H; sigma=sigma)
end

"""
    identify_hsic(model::VARModel; kernel=:gaussian, sigma=1.0,
                  max_iter=200, tol=1e-6) -> ICASVARResult

Identify SVAR by minimizing pairwise HSIC between recovered shocks.

HSIC with a characteristic kernel (Gaussian) is zero iff variables are independent.

**Reference**: Gretton et al. (2005)
"""
function identify_hsic(model::VARModel{T}; kernel::Symbol=:gaussian,
                        sigma::T=T(1.0), max_iter::Int=200,
                        tol::T=T(1e-6),
                        seed::Union{Integer,Nothing}=nothing,
                        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    n = nvars(model)
    Z, W_white, dewhiten = _whiten(model.U)

    # Use median heuristic for bandwidth if not specified (seeded subsample)
    if sigma == one(T)
        dists = T[]
        T_obs = size(Z, 1)
        n_sample = min(T_obs, 200)
        idx = randperm(rng, T_obs)[1:n_sample]
        for i in idx, j in idx
            i < j && push!(dists, norm(@view(Z[i, :]) .- @view(Z[j, :])))
        end
        sigma = max(median(dists), eps(T))
    end

    n_angles = n * (n - 1) ÷ 2
    angles0 = zeros(T, n_angles)

    # Hoist the invariant centering matrix and the T_obs x T_obs scratch buffers
    # out of the objective — reused by every evaluation and pair
    T_obs = size(Z, 1)
    H = I - ones(T, T_obs, T_obs) / T_obs
    bufs = ntuple(_ -> Matrix{T}(undef, T_obs, T_obs), 6)
    result = Optim.optimize(a -> _hsic_objective(a, Z, n, bufs..., H; sigma=sigma), angles0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, g_tol=tol))

    Q_opt = _givens_to_orthogonal(Optim.minimizer(result), n)
    W_full = Q_opt * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)

    _ica_cvm_finalize_result(B0, W_full, Q, shocks, :hsic, Optim.converged(result),
                             Optim.iterations(result), T(Optim.minimum(result)), model.U)
end

# =============================================================================
# Cramér–von Mises Empirical-Copula ICA (Herwartz & Plödt 2016)
# =============================================================================

function _ica_cvm_objective(S::AbstractMatrix{T}, U_pit::Matrix{T},
                            ord_buf::Vector{Int}) where {T<:AbstractFloat}
    T_eff, n = size(S)
    inv_Tp1 = one(T) / T(T_eff + 1)

    @inbounds for j in 1:n
        col_j = @view S[:, j]
        sortperm!(ord_buf, col_j)
        # Average tied ranks if any exact ties occur
        idx = 1
        while idx <= T_eff
            v0 = col_j[ord_buf[idx]]
            end_idx = idx
            while end_idx < T_eff && col_j[ord_buf[end_idx + 1]] == v0
                end_idx += 1
            end
            avg_rank = T(idx + end_idx) / T(2) * inv_Tp1
            for k in idx:end_idx
                U_pit[ord_buf[k], j] = avg_rank
            end
            idx = end_idx + 1
        end
    end

    # Genest & Rémillard (2004) / Herwartz & Plödt (2016) Möbius empirical-copula CvM statistic
    # with Anderson–Darling tail weighting K_AD(a,b) = -log((1 - min(a,b)) * max(a,b)) - 1
    # (or uniform K(a,b) = 1/3 + (a^2 + b^2)/2 - max(a,b)).
    tot = zero(T)
    @inbounds for t in 1:T_eff
        p_d = one(T)
        s1_d = zero(T)
        for j in 1:n
            u = U_pit[t, j]
            kv = -log((one(T) - u) * u) - one(T)
            p_d *= (one(T) + kv)
            s1_d += kv
        end
        tot += p_d - one(T) - s1_d

        acc = zero(T)
        for s in (t + 1):T_eff
            p_ts = one(T)
            s1_ts = zero(T)
            for j in 1:n
                a = U_pit[t, j]
                b = U_pit[s, j]
                lo = a < b ? a : b
                hi = a > b ? a : b
                kv = -log((one(T) - lo) * hi) - one(T)
                p_ts *= (one(T) + kv)
                s1_ts += kv
            end
            acc += p_ts - one(T) - s1_ts
        end
        tot += T(2) * acc
    end

    return max(zero(T), tot / T(T_eff))
end

"""
    identify_cvm(model::VARModel{T}; max_iter::Int=200, n_starts::Int=3, tol::Real=1e-6,
                 rng::AbstractRNG=Random.default_rng(),
                 seed::Union{Integer,Nothing}=nothing) -> ICASVARResult{T}

Identify structural SVAR shocks via **Cramér–von Mises (CvM) empirical-copula ICA**
(Herwartz & Plödt 2016; Herwartz 2018).

Minimizes the Möbius-decomposed Anderson–Darling-weighted empirical copula Cramér–von Mises
independence discrepancy over Givens rotations ``Q(\\theta) \\in SO(n)`` applied to
whitened VAR residuals ``\\hat{\\varepsilon}(\\theta) = Z \\, Q(\\theta)``:

```math
\\mathcal{M}_T(\\theta) = \\frac{1}{T} \\sum_{t=1}^T \\sum_{s=1}^T \\left[
\\prod_{j=1}^n \\bigl(1 + K(u_{t,j}, u_{s,j})\\bigr) - 1 - \\sum_{j=1}^n K(u_{t,j}, u_{s,j})
\\right],
```

where:
- ``u_{t,j} = \\operatorname{rank}(\\hat{\\varepsilon}_{t,j}(\\theta)) / (T + 1)`` are
  empirical probability integral transforms,
- ``K(a, b) = -\\log\\bigl((1 - \\min(a,b))\\max(a,b)\\bigr) - 1`` is the tail-weighted
  Anderson–Darling copula kernel ``\\int_0^1 \\frac{(\\mathbb{I}(a \\le x) - x)(\\mathbb{I}(b \\le x) - x)}{x(1-x)} dx``,
- ``Z`` is the whitened VAR residual matrix and ``Q(\\theta)`` is parameterized by Givens
  rotations.

# Arguments
- `model::VARModel{T}`: estimated reduced-form VAR model
- `max_iter::Int=200`: maximum Nelder–Mead iterations per start
- `n_starts::Int=3`: number of additional random starting rotations (augmented by identity, FastICA, and JADE warm starts)
- `tol::Real=1e-6`: optimizer convergence tolerance
- `rng::AbstractRNG=Random.default_rng()`: random number generator
- `seed::Union{Integer,Nothing}=nothing`: optional random seed

# Example
```julia
sim = dgp_var(; T=500, n=2, seed=1)
m = estimate_var(sim.Y, 1)
res = identify_cvm(m; seed=1)
```
"""
function identify_cvm(model::VARModel{T};
                      max_iter::Int=200,
                      n_starts::Int=3,
                      tol::Real=1e-6,
                      rng::AbstractRNG=Random.default_rng(),
                      seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    max_iter < 1 && throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    n_starts < 1 && throw(ArgumentError("n_starts must be ≥ 1, got $n_starts"))
    rng_eff = _resolve_repro_rng(rng, seed)

    U = model.U
    T_eff, n = size(U)
    Z, W_white, _ = _whiten(U)

    if n == 1
        W_ica = ones(T, 1, 1)
        W_full = W_ica * W_white
        B0, Q, shocks = _ica_to_svar(W_full, model)
        return _ica_cvm_finalize_result(B0, W_full, Q, shocks, :cvm, true, 0, zero(T), U)
    end

    U_pit = Matrix{T}(undef, T_eff, n)
    S_buf = Matrix{T}(undef, T_eff, n)
    ord_buf = Vector{Int}(undef, T_eff)

    function cvm_loss(angles::AbstractVector)
        Q_rot = _givens_to_orthogonal(T.(angles), n)
        mul!(S_buf, Z, Q_rot)
        return Float64(_ica_cvm_objective(S_buf, U_pit, ord_buf))
    end

    n_angles = (n * (n - 1)) ÷ 2
    starts = Vector{Float64}[]

    # 1. Exact coarse grid scan when n == 2 guarantees global optimum across piecewise rank surface
    if n == 2
        grid_th = range(-pi / 4, pi / 4; length=73)
        grid_vals = Float64[cvm_loss([th]) for th in grid_th]
        top_idx = partialsortperm(grid_vals, 1:min(4, length(grid_vals)))
        for idx in top_idx
            push!(starts, [Float64(grid_th[idx])])
        end
    else
        push!(starts, zeros(Float64, n_angles))
    end

    # 2. FastICA & JADE warm starts
    try
        W_f, _ = _fastica_symmetric(Z, n; contrast=:logcosh, max_iter=150, tol=T(1e-6), rng=rng_eff)
        push!(starts, Float64.(_q_to_so_givens(Matrix{T}(transpose(W_f)), n)))
    catch
    end
    try
        cum_mats = _jade_cumulant_matrices(Z)
        V_j, _ = _joint_diagonalization(cum_mats; max_iter=100, tol=T(1e-6))
        push!(starts, Float64.(_q_to_so_givens(Matrix{T}(V_j), n)))
    catch
    end

    # 3. Random SO(n) starts
    for _ in 1:n_starts
        A_r = randn(rng_eff, T, n, n)
        Q_r = Matrix{T}(qr(A_r).Q)
        if det(Q_r) < zero(T)
            Q_r[:, 1] .*= -one(T)
        end
        push!(starts, Float64.(_q_to_so_givens(Q_r, n)))
    end

    best_angles = copy(starts[1])
    best_obj = Inf
    best_conv = false
    total_iters = 0

    opts = Optim.Options(iterations=max_iter, f_reltol=Float64(tol), g_tol=Float64(tol))
    for st in starts
        if n == 2
            a0 = st[1]
            res_b = Optim.optimize(x -> cvm_loss([x]), a0 - pi / 18, a0 + pi / 18, Optim.Brent();
                                   iterations=max_iter, rel_tol=Float64(tol))
            val = Optim.minimum(res_b)
            total_iters += Optim.iterations(res_b)
            if val < best_obj
                best_obj = val
                best_angles = [Optim.minimizer(res_b)]
                best_conv = Optim.converged(res_b)
            end
        else
            res_nm = Optim.optimize(cvm_loss, st, Optim.NelderMead(), opts)
            val = Optim.minimum(res_nm)
            total_iters += Optim.iterations(res_nm)
            if val < best_obj
                best_obj = val
                best_angles = copy(Optim.minimizer(res_nm))
                best_conv = Optim.converged(res_nm)
            end
        end
    end

    Q_rot = _givens_to_orthogonal(T.(best_angles), n)
    W_ica = Matrix{T}(transpose(Q_rot))
    W_full = W_ica * W_white
    B0, Q, shocks = _ica_to_svar(W_full, model)
    return _ica_cvm_finalize_result(B0, W_full, Q, shocks, :cvm, best_conv,
                                    total_iters, T(best_obj), U)
end
