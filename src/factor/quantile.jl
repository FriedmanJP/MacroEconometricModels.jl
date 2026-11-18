# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Quantile Factor Model (QFM) via Iterative Quantile Regression (IQR) and Convolution Smoothing.

Implements Chen, Dolado & Gonzalo (2021):
``Q_{\\tau}(X_{it} \\mid f_t) = \\lambda_i(\\tau)' f_t(\\tau)``

Estimated by alternating batched quantile regressions (`_qreg_batch!`) or
convolution-smoothed L-BFGS optimization (`method=:smoothed_iqr`) with canonical
CDG (§2.1) factor/loading normalization:
``F'F / T = I_r, \\qquad \\Lambda'\\Lambda / N = \\mathrm{diag}(\\sigma_1^2 \\ge \\dots \\ge \\sigma_r^2 \\ge 0).``

Supports CDG (2021) Theorem 1 Powell/Hall–Sheather asymptotic standard errors
(with optional Newey–West HAC adjustment) and wild quantile bootstrap standard
errors with orthogonal Procrustes alignment.

References:
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models.
  *Econometrica*, 89(2), 875–910.
- Fernandes, M., Guerre, E., & Horta, E. (2021). Smoothing quantile regressions.
  *Journal of Business & Economic Statistics*, 39(1), 338–357.
- He, Y., Kong, X., Yu, L., & Zhang, X. (2023). Quantile factor analysis for
  large-dimensional time series with smoothed quantile regression.
  *Journal of Econometrics*, 232(2), 367–390.
- Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related inference
  processes for quantile regression. *Journal of the American Statistical Association*,
  94(448), 1296–1310.
"""

using LinearAlgebra, Statistics, StatsAPI, Random
import SpecialFunctions

# =============================================================================
# Quantile Factor Model Type
# =============================================================================

"""
    QuantileFactorModel{T<:AbstractFloat} <: AbstractFactorModel

Quantile factor model (Chen, Dolado & Gonzalo 2021) estimated at quantile `tau` via
Iterative Quantile Regression (`:iqr`) or Convolution-Smoothed QFA (`:smoothed_iqr`).

# Fields
- `X::Matrix{T}` — raw data panel (`T_obs × N`)
- `tau::T` — fitted quantile level in `(0, 1)`
- `factors::Matrix{T}` — normalized quantile factors (`T_obs × r`, `F'F / T_obs = I_r`)
- `loadings::Matrix{T}` — normalized quantile loadings (`N × r`, `Λ'Λ / N = diag(sigma_sq)`)
- `sigma_sq::Vector{T}` — factor strengths (`diag(Λ'Λ / N)`, descending)
- `common_component::Matrix{T}` — fitted common component (`T_obs × N`, in raw `X` units)
- `residuals::Matrix{T}` — `X - common_component` (in raw `X` units)
- `objective::T` — average check-function loss `(NT)⁻¹ Σ ρ_τ(X_proc - FΛ')`
- `pseudo_r2::Vector{T}` — Koenker–Machado per-series `1 - V̂_i(τ) / Ṽ_i(τ)`
- `r::Int` — number of quantile factors
- `method::Symbol` — estimation method (`:iqr` or `:smoothed_iqr`)
- `standardized::Bool` — whether `X` was standardized prior to estimation
- `converged::Bool` — whether the winning solver run satisfied the stopping rule
- `iterations::Int` — outer IQR or L-BFGS iterations performed by the winning run
- `se_loadings::Union{Nothing,Matrix{T}}` — optional loading SEs (`N × r`) when `se != :none`
- `se_factors::Union{Nothing,Matrix{T}}` — optional factor SEs (`T_obs × r`) when `se != :none`
- `varnames::Vector{String}` — variable names (`length N`)
- `manifest::Union{ReproManifest,Nothing}` — reproducibility metadata
"""
struct QuantileFactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    tau::T
    factors::Matrix{T}
    loadings::Matrix{T}
    sigma_sq::Vector{T}
    common_component::Matrix{T}
    residuals::Matrix{T}
    objective::T
    pseudo_r2::Vector{T}
    r::Int
    method::Symbol
    standardized::Bool
    converged::Bool
    iterations::Int
    se_loadings::Union{Nothing,Matrix{T}}
    se_factors::Union{Nothing,Matrix{T}}
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

# 17-arg constructor (all non-manifest fields positional; manifest optional kwarg)
QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                       objective, pseudo_r2, r, method, standardized, converged, iterations,
                       se_loadings, se_factors, varnames;
                       manifest=nothing) where {T<:AbstractFloat} =
    QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                           objective, pseudo_r2, r, method, standardized, converged, iterations,
                           se_loadings, se_factors, varnames, manifest)

# 15-arg backward-compatible positional constructor (with varnames)
QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                       objective, pseudo_r2, r, method, standardized, converged, iterations,
                       varnames::Vector{String};
                       se_loadings=nothing, se_factors=nothing,
                       manifest=nothing) where {T<:AbstractFloat} =
    QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                           objective, pseudo_r2, r, method, standardized, converged, iterations,
                           se_loadings, se_factors, varnames, manifest)

# 14-arg positional constructor with keyword defaults
QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                       objective, pseudo_r2, r, method, standardized, converged, iterations;
                       se_loadings=nothing, se_factors=nothing,
                       varnames::Vector{String}=["Var $i" for i in 1:size(X, 2)],
                       manifest=nothing) where {T<:AbstractFloat} =
    QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                           objective, pseudo_r2, r, method, standardized, converged, iterations,
                           se_loadings, se_factors, varnames, manifest)

QuantileFactorModel(X::Matrix{T}, tau::T, factors::Matrix{T}, loadings::Matrix{T},
                    sigma_sq::Vector{T}, common_component::Matrix{T}, residuals::Matrix{T},
                    objective::T, pseudo_r2::Vector{T}, r::Int, method::Symbol,
                    standardized::Bool, converged::Bool, iterations::Int;
                    kwargs...) where {T<:AbstractFloat} =
    QuantileFactorModel{T}(X, tau, factors, loadings, sigma_sq, common_component, residuals,
                           objective, pseudo_r2, r, method, standardized, converged, iterations;
                           kwargs...)

# Full keyword constructor
function QuantileFactorModel(;
    X::AbstractMatrix{T},
    tau::Real,
    factors::AbstractMatrix{T},
    loadings::AbstractMatrix{T},
    sigma_sq::AbstractVector{T},
    common_component::AbstractMatrix{T},
    residuals::AbstractMatrix{T},
    objective::Real,
    pseudo_r2::AbstractVector{T},
    r::Int,
    method::Symbol=:iqr,
    standardized::Bool=false,
    converged::Bool=true,
    iterations::Int=0,
    se_loadings::Union{Nothing,AbstractMatrix{T}}=nothing,
    se_factors::Union{Nothing,AbstractMatrix{T}}=nothing,
    varnames::Vector{String}=["Var $i" for i in 1:size(X, 2)],
    manifest::Union{ReproManifest,Nothing}=nothing,
) where {T<:AbstractFloat}
    QuantileFactorModel{T}(
        Matrix{T}(X), T(tau), Matrix{T}(factors), Matrix{T}(loadings),
        Vector{T}(sigma_sq), Matrix{T}(common_component), Matrix{T}(residuals),
        T(objective), Vector{T}(pseudo_r2), r, method, standardized, converged, iterations,
        se_loadings === nothing ? nothing : Matrix{T}(se_loadings),
        se_factors === nothing ? nothing : Matrix{T}(se_factors),
        varnames, manifest,
    )
end

# =============================================================================
# Standardization, Normalization & Check-Loss Helpers
# =============================================================================

"""Standardize columns of `X`, returning `(X_proc, center, scale)`.

When `robust_scale=true`, centers by column median and scales by normalized IQR
`(q₇₅ − q₂₅) / 1.3489795003921634` with MAD and SD fallbacks, clamped below at `T(1e-8)`.
When `robust_scale=false`, uses mean and SD clamped below at `T(1e-8)`."""
function _qfm_standardize(X::Matrix{T}, robust_scale::Bool) where {T<:AbstractFloat}
    _, N = size(X)
    center = Vector{T}(undef, N)
    scale = Vector{T}(undef, N)
    iqr_norm = T(1.3489795003921634)
    mad_norm = T(1.482602218505602)
    floor_s = T(1e-8)

    @inbounds for i in 1:N
        col = @view X[:, i]
        if robust_scale
            c_i = T(Statistics.median(col))
            q75 = T(Statistics.quantile(col, 0.75))
            q25 = T(Statistics.quantile(col, 0.25))
            s_i = (q75 - q25) / iqr_norm
            if !isfinite(s_i) || s_i <= floor_s
                mad_s = mad_norm * T(Statistics.median(abs.(col .- c_i)))
                s_i = mad_s
                if !isfinite(s_i) || s_i <= floor_s
                    sd_s = T(Statistics.std(col))
                    s_i = isfinite(sd_s) ? sd_s : one(T)
                end
            end
            center[i] = c_i
            scale[i] = max(s_i, floor_s)
        else
            c_i = T(Statistics.mean(col))
            sd_s = T(Statistics.std(col))
            center[i] = c_i
            scale[i] = max(isfinite(sd_s) ? sd_s : one(T), floor_s)
        end
    end
    X_proc = (X .- center') ./ scale'
    return X_proc, center, scale
end

"""Canonical Chen–Dolado–Gonzalo (2021, §2.1) normalization of `(F_raw, Λ_raw)`.

Enforces `F'F / T_obs = I_r`, `Λ'Λ / N = diag(σ₁² ≥ … ≥ σ_r² ≥ 0)`, and a
deterministic sign convention (first entry of `Λ[:, j]` with `|entry| > 1e-8`,
or largest-magnitude entry if all tiny, is positive), preserving `F * Λ'`."""
function _qfm_normalize(F_raw::AbstractMatrix{T}, Λ_raw::AbstractMatrix{T}) where {T<:AbstractFloat}
    T_obs, r = size(F_raw)
    N = size(Λ_raw, 1)
    sqrt_T = sqrt(T(T_obs))
    sqrt_N = sqrt(T(N))

    Q_f, R_f = qr(F_raw ./ sqrt_T)
    F1 = sqrt_T .* Matrix(Q_f)
    Λ1 = Λ_raw * R_f'

    sv = svd(Λ1 ./ sqrt_N)
    F_norm = F1 * sv.V
    Λ_norm = (sqrt_N .* sv.U) .* sv.S'
    sigma_sq = sv.S .^ 2

    thresh = T(1e-8)
    @inbounds for j in 1:r
        col = @view Λ_norm[:, j]
        idx = findfirst(x -> abs(x) > thresh, col)
        i = idx === nothing ? argmax(abs.(col)) : idx
        if Λ_norm[i, j] < zero(T)
            Λ_norm[:, j] .*= -one(T)
            F_norm[:, j] .*= -one(T)
        end
    end

    return F_norm, Λ_norm, sigma_sq
end

"""Zero-allocation check-function loss `Σ_t ρ_τ(X[t, j] - dot(F[t, :], lam))`."""
function _qfm_col_check_loss(X::Matrix{T}, F::Matrix{T}, lam::AbstractVector{T},
                             j::Int, tau::T) where {T<:AbstractFloat}
    T_obs, r = size(F)
    acc = zero(T)
    @inbounds for t in 1:T_obs
        pred = zero(T)
        for k in 1:r
            pred += F[t, k] * lam[k]
        end
        u = X[t, j] - pred
        acc += u * (tau - (u < zero(T) ? one(T) : zero(T)))
    end
    return acc
end

"""Average QFM objective `(NT)⁻¹ Σ_{i,t} ρ_τ(X_proc[t, i] - (F * Λ')[t, i])`."""
function _qfm_objective(X_proc::Matrix{T}, F::Matrix{T}, Λ::Matrix{T},
                        tau::T) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    total = zero(T)
    @inbounds for i in 1:N
        total += _qfm_col_check_loss(X_proc, F, @view(Λ[i, :]), i, tau)
    end
    return total / T(T_obs * N)
end

"""Koenker–Machado per-series pseudo-R²: `1 - V̂_i(τ) / Ṽ_i(τ)` on `X_proc`."""
function _qfm_pseudo_r2(X_proc::Matrix{T}, F::Matrix{T}, Λ::Matrix{T},
                        tau::T) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    pr2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        v_hat = _qfm_col_check_loss(X_proc, F, @view(Λ[i, :]), i, tau)
        col = @view X_proc[:, i]
        q_i = T(Statistics.quantile(col, tau))
        v_tilde = zero(T)
        for t in 1:T_obs
            u = col[t] - q_i
            v_tilde += u * (tau - (u < zero(T) ? one(T) : zero(T)))
        end
        pr2[i] = v_tilde > zero(T) ? one(T) - v_hat / v_tilde : zero(T)
    end
    return pr2
end

# =============================================================================
# Initialization & Monotone IQR Solver
# =============================================================================

"""Initialize `T_obs × r` orthonormal factors (`F'F / T_obs = I_r`)."""
function _qfm_init_factors(X_proc::Matrix{T}, r::Int, tau::T, init::Symbol,
                           local_rng::AbstractRNG) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    sqrt_T = sqrt(T(T_obs))

    if init === :pca
        sv = svd(X_proc)
        return sqrt_T .* sv.U[:, 1:r]
    elseif init === :quantile_svd
        # Robust Winsorized location + tail-asymmetry scale SVD (immune to Cauchy outliers)
        Z_loc = Matrix{T}(undef, T_obs, N)
        Z_scale = Matrix{T}(undef, T_obs, N)
        @inbounds for i in 1:N
            col = @view X_proc[:, i]
            med = T(Statistics.median(col))
            lo = T(Statistics.quantile(col, 0.05))
            hi = T(Statistics.quantile(col, 0.95))
            for t in 1:T_obs
                c = clamp(col[t], lo, hi)
                Z_loc[t, i] = c
                Z_scale[t, i] = abs(c - med)
            end
        end
        w = T(0.5) * abs(T(2) * tau - one(T))
        Z = w > T(1e-6) ? hcat(Z_loc, w .* Z_scale) : Z_loc
        sv = svd(Z)
        return sqrt_T .* sv.U[:, 1:r]
    else
        # :random
        F_raw = randn(local_rng, T, T_obs, r)
        return sqrt_T .* Matrix(qr(F_raw).Q)
    end
end

"""Run one monotone IQR trajectory from initial orthonormal factor matrix `F_init`."""
function _qfm_iqr_single(X_proc::Matrix{T}, X_proc_t::Matrix{T}, F_init::Matrix{T},
                         tau::T, max_iter::Int, tol_eff::T,
                         threaded::Bool) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    r = size(F_init, 2)

    F_cur = copy(F_init)
    F_prev = Matrix{T}(undef, T_obs, r)
    F_cand = Matrix{T}(undef, T_obs, r)
    Lambda_cur = zeros(T, N, r)
    sigma_sq_cur = zeros(T, r)

    Lambda_cand_t = Matrix{T}(undef, r, N)
    F_cand_t = Matrix{T}(undef, r, T_obs)

    obj_trace = Vector{T}()
    sizehint!(obj_trace, min(max_iter, 64))
    obj_prev = T(Inf)
    converged = false
    iterations = 0
    sub_tol = sqrt(tol_eff)
    inv_sqrt_Tr = one(T) / sqrt(T(T_obs * r))

    for k in 1:max_iter
        copyto!(F_prev, F_cur)

        # 1. Update loadings Λ given F_cur
        _qreg_batch!(Lambda_cand_t, F_cur, X_proc, tau; threaded=threaded)
        updated = false
        if k == 1
            @inbounds for i in 1:N
                cand = @view Lambda_cand_t[:, i]
                if all(isfinite, cand)
                    for j in 1:r
                        Lambda_cur[i, j] = cand[j]
                    end
                end
            end
            updated = true
        else
            @inbounds for i in 1:N
                cand = @view Lambda_cand_t[:, i]
                all(isfinite, cand) || continue
                old_loss = _qfm_col_check_loss(X_proc, F_cur, @view(Lambda_cur[i, :]), i, tau)
                new_loss = _qfm_col_check_loss(X_proc, F_cur, cand, i, tau)
                if new_loss <= old_loss
                    for j in 1:r
                        Lambda_cur[i, j] = cand[j]
                    end
                    updated = true
                end
            end
        end

        # 2. Update factors F given Lambda_cur
        _qreg_batch!(F_cand_t, Lambda_cur, X_proc_t, tau; threaded=threaded)
        copyto!(F_cand, F_cur)
        @inbounds for t in 1:T_obs
            cand = @view F_cand_t[:, t]
            all(isfinite, cand) || continue
            old_loss = _qfm_col_check_loss(X_proc_t, Lambda_cur, @view(F_cur[t, :]), t, tau)
            new_loss = _qfm_col_check_loss(X_proc_t, Lambda_cur, cand, t, tau)
            if new_loss <= old_loss
                for j in 1:r
                    F_cand[t, j] = cand[j]
                end
                updated = true
            end
        end

        # 3. Canonical CDG normalization (preserves F * Λ' identically)
        if k == 1 || updated
            F_cur, Lambda_cur, sigma_sq_cur = _qfm_normalize(F_cand, Lambda_cur)
        end

        # 4. Record check-loss objective and check convergence (k >= 2)
        obj_cur = _qfm_objective(X_proc, F_cur, Lambda_cur, tau)
        push!(obj_trace, obj_cur)
        iterations = k

        if k >= 2
            obj_diff = abs(obj_prev - obj_cur)
            sub_diff = norm(F_cur - F_prev) * inv_sqrt_Tr
            if obj_diff <= tol_eff * max(one(T), obj_prev) && sub_diff <= sub_tol
                converged = true
                break
            end
        end
        obj_prev = obj_cur
    end

    return (
        factors = F_cur,
        loadings = Lambda_cur,
        sigma_sq = sigma_sq_cur,
        objective = obj_trace[end],
        converged = converged,
        iterations = iterations,
        obj_trace = obj_trace,
    )
end

# =============================================================================
# Convolution-Smoothed QFA Helpers (Fernandes–Guerre–Horta 2021; He et al. 2023)
# =============================================================================

"""Scalar convolution-smoothed check loss `l_{h,τ}(u)` and derivative `l'_{h,τ}(u)`.

Supports `:gaussian`, `:logistic` (overflow-safe), and `:epanechnikov` (exact `C¹`
piecewise polynomial integral). Generic over `Real` so `ForwardDiff.Dual` propagates."""
@inline function _qfm_smooth_loss_and_deriv(u::U, tau::Real, h::Real, kernel::Symbol) where {U<:Real}
    v = u / h
    tau_v = oftype(v, tau)
    one_v = one(v)
    if kernel === :gaussian
        half_v = oftype(v, 0.5)
        inv_sqrt2 = inv(sqrt(oftype(v, 2)))
        inv_sqrt2pi = inv(sqrt(oftype(v, 2) * oftype(v, pi)))
        Phi = half_v * SpecialFunctions.erfc(-v * inv_sqrt2)
        phi = exp(-half_v * v * v) * inv_sqrt2pi
        deriv = Phi - (one_v - tau_v)
        val = u * deriv + h * phi
        return val, deriv
    elseif kernel === :logistic
        if v >= zero(v)
            emv = exp(-v)
            sig = one_v / (one_v + emv)
            val = u * tau_v + h * log1p(emv)
            deriv = sig - (one_v - tau_v)
            return val, deriv
        else
            ev = exp(v)
            sig = ev / (one_v + ev)
            val = u * (tau_v - one_v) + h * log1p(ev)
            deriv = sig - (one_v - tau_v)
            return val, deriv
        end
    else
        # :epanechnikov — exact C¹ piecewise polynomial integral of (3/4)(1 - z²) 1(|z| ≤ 1)
        if v >= one_v
            return u * tau_v, tau_v
        elseif v <= -one_v
            dm = tau_v - one_v
            return u * dm, dm
        else
            v2 = v * v
            half_v = oftype(v, 0.5)
            val = u * (tau_v - half_v) +
                  (h * oftype(v, 0.0625)) * (oftype(v, 3) + oftype(v, 6) * v2 - v2 * v2)
            deriv = (tau_v - half_v) + oftype(v, 0.75) * v - oftype(v, 0.25) * v2 * v
            return val, deriv
        end
    end
end

"""Convolution-smoothed QFM objective `(NT)⁻¹ Σ l_{h,τ}(X_proc - F Λ')`.

Generic over `AbstractMatrix` and element types so `ForwardDiff.gradient` works directly."""
function _qfm_smooth_objective(X_proc::AbstractMatrix, F::AbstractMatrix,
                               Λ::AbstractMatrix, tau::Real, h::Real, kernel::Symbol)
    T_obs, N = size(X_proc)
    r = size(F, 2)
    z0 = zero(promote_type(eltype(X_proc), eltype(F), eltype(Λ), typeof(tau), typeof(h)))
    total = z0
    @inbounds for i in 1:N
        for t in 1:T_obs
            pred = z0
            for k in 1:r
                pred += F[t, k] * Λ[i, k]
            end
            u = X_proc[t, i] - pred
            val, _ = _qfm_smooth_loss_and_deriv(u, tau, h, kernel)
            total += val
        end
    end
    return total / (T_obs * N)
end

"""Analytical gradients `(∂M̃_{NT}/∂F, ∂M̃_{NT}/∂Λ)` of the smoothed QFM objective."""
function _qfm_smooth_grad(X_proc::AbstractMatrix{T}, F::AbstractMatrix{T},
                          Λ::AbstractMatrix{T}, tau::T, h::T, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    r = size(F, 2)
    U_buf = Matrix{T}(undef, T_obs, N)
    G_buf = Matrix{T}(undef, T_obs, N)
    grad_F = Matrix{T}(undef, T_obs, r)
    grad_L = Matrix{T}(undef, N, r)
    _qfm_smooth_fg!(grad_F, grad_L, X_proc, F, Λ, tau, h, kernel, U_buf, G_buf)
    return grad_F, grad_L
end

"""In-place smoothed objective and optional gradients `(grad_F, grad_Λ)` using pre-allocated buffers."""
function _qfm_smooth_fg!(grad_F::Union{Nothing,AbstractMatrix{T}},
                         grad_L::Union{Nothing,AbstractMatrix{T}},
                         X_proc::AbstractMatrix{T},
                         F::AbstractMatrix{T},
                         Λ::AbstractMatrix{T},
                         tau::T, h::T, kernel::Symbol,
                         U_buf::Matrix{T}, G_buf::Matrix{T}) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    inv_NT = one(T) / T(T_obs * N)
    copyto!(U_buf, X_proc)
    mul!(U_buf, F, transpose(Λ), -one(T), one(T))

    need_grad = grad_F !== nothing || grad_L !== nothing
    total = zero(T)
    if need_grad
        @inbounds for i in 1:N
            for t in 1:T_obs
                val, deriv = _qfm_smooth_loss_and_deriv(U_buf[t, i], tau, h, kernel)
                total += val
                G_buf[t, i] = deriv * inv_NT
            end
        end
        if grad_F !== nothing
            mul!(grad_F, G_buf, Λ, -one(T), zero(T))
        end
        if grad_L !== nothing
            mul!(grad_L, transpose(G_buf), F, -one(T), zero(T))
        end
    else
        @inbounds for i in 1:N
            for t in 1:T_obs
                val, _ = _qfm_smooth_loss_and_deriv(U_buf[t, i], tau, h, kernel)
                total += val
            end
        end
    end
    return total * inv_NT
end

"""Rule-of-thumb smoothing bandwidth `h = max(0.05, std(resid_init) * (log(NT)/min(N,T))^0.25)`."""
function _qfm_smooth_bandwidth(X_proc::Matrix{T}, F_init::Matrix{T},
                               bandwidth::Union{Symbol,Real}) where {T<:AbstractFloat}
    if bandwidth isa Real
        return T(bandwidth)
    end
    T_obs, N = size(X_proc)
    Λ_init = (X_proc' * F_init) / T(T_obs)
    resid_init = X_proc - F_init * Λ_init'
    sd_r = T(Statistics.std(resid_init))
    if !isfinite(sd_r)
        sd_r = zero(T)
    end
    rate = (log(T(N * T_obs)) / T(min(N, T_obs)))^T(0.25)
    return max(T(0.05), sd_r * rate)
end

"""Run one convolution-smoothed QFA L-BFGS trajectory from initial orthonormal `F_init`."""
function _qfm_smooth_single(X_proc::Matrix{T}, F_init::Matrix{T}, tau::T, h::T,
                            kernel::Symbol, max_iter::Int, tol_eff::T) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    r = size(F_init, 2)

    # Promote Float32 internally to Float64 so Optim.LBFGS line-search never stalls
    S = promote_type(T, Float64)
    X_opt = S === T ? X_proc : S.(X_proc)
    F0_opt = S === T ? F_init : S.(F_init)
    tau_opt = S(tau)
    h_opt = S(h)
    tol_opt = max(S(tol_eff), S(10) * eps(S))

    sqrt_T = sqrt(S(T_obs))
    sqrt_N = sqrt(S(N))
    inv_sqrt_T = one(S) / sqrt_T
    inv_sqrt_N = one(S) / sqrt_N

    U_buf = Matrix{S}(undef, T_obs, N)
    G_buf = Matrix{S}(undef, T_obs, N)

    # 1. Warm-start Λ holding F0_opt fixed
    Λ0_opt = (X_opt' * F0_opt) / S(T_obs)
    if !all(isfinite, Λ0_opt)
        fill!(Λ0_opt, zero(S))
    end
    L_tilde0 = vec(Λ0_opt .* inv_sqrt_N)
    L_mat_buf = Matrix{S}(undef, N, r)
    gL_buf = Matrix{S}(undef, N, r)

    function f_L(theta_L)
        copyto!(L_mat_buf, reshape(theta_L, N, r))
        L_mat_buf .*= sqrt_N
        return _qfm_smooth_fg!(nothing, nothing, X_opt, F0_opt, L_mat_buf,
                               tau_opt, h_opt, kernel, U_buf, G_buf)
    end
    function g_L!(G_vec, theta_L)
        copyto!(L_mat_buf, reshape(theta_L, N, r))
        L_mat_buf .*= sqrt_N
        _qfm_smooth_fg!(nothing, gL_buf, X_opt, F0_opt, L_mat_buf,
                        tau_opt, h_opt, kernel, U_buf, G_buf)
        gL_buf .*= sqrt_N
        copyto!(G_vec, vec(gL_buf))
        return G_vec
    end

    warm_iters = min(max_iter, 40)
    res_L = Optim.optimize(f_L, g_L!, L_tilde0, Optim.LBFGS(),
                           Optim.Options(iterations=warm_iters, g_tol=max(tol_opt, S(1e-6)),
                                         show_trace=false))
    L_tilde_warm = Optim.minimizer(res_L)

    # 2. Joint L-BFGS over scaled parameters θ = [vec(F / √T); vec(Λ / √N)]
    nF = T_obs * r
    theta0 = Vector{S}(undef, nF + N * r)
    copyto!(@view(theta0[1:nF]), vec(F0_opt .* inv_sqrt_T))
    copyto!(@view(theta0[(nF + 1):end]), L_tilde_warm)

    F_mat_buf = Matrix{S}(undef, T_obs, r)
    gF_buf = Matrix{S}(undef, T_obs, r)

    function f_joint(theta)
        copyto!(F_mat_buf, reshape(@view(theta[1:nF]), T_obs, r))
        F_mat_buf .*= sqrt_T
        copyto!(L_mat_buf, reshape(@view(theta[(nF + 1):end]), N, r))
        L_mat_buf .*= sqrt_N
        return _qfm_smooth_fg!(nothing, nothing, X_opt, F_mat_buf, L_mat_buf,
                               tau_opt, h_opt, kernel, U_buf, G_buf)
    end
    function g_joint!(G_vec, theta)
        copyto!(F_mat_buf, reshape(@view(theta[1:nF]), T_obs, r))
        F_mat_buf .*= sqrt_T
        copyto!(L_mat_buf, reshape(@view(theta[(nF + 1):end]), N, r))
        L_mat_buf .*= sqrt_N
        _qfm_smooth_fg!(gF_buf, gL_buf, X_opt, F_mat_buf, L_mat_buf,
                        tau_opt, h_opt, kernel, U_buf, G_buf)
        gF_buf .*= sqrt_T
        gL_buf .*= sqrt_N
        copyto!(@view(G_vec[1:nF]), vec(gF_buf))
        copyto!(@view(G_vec[(nF + 1):end]), vec(gL_buf))
        return G_vec
    end

    res = Optim.optimize(f_joint, g_joint!, theta0, Optim.LBFGS(),
                         Optim.Options(iterations=max_iter, g_tol=tol_opt, show_trace=false))
    theta_hat = Optim.minimizer(res)
    F_raw = sqrt_T .* reshape(theta_hat[1:nF], T_obs, r)
    L_raw = sqrt_N .* reshape(theta_hat[(nF + 1):end], N, r)

    F_norm_S, L_norm_S, sigma_sq_S = _qfm_normalize(F_raw, L_raw)
    F_norm = T === S ? F_norm_S : T.(F_norm_S)
    L_norm = T === S ? L_norm_S : T.(L_norm_S)
    sigma_sq = T === S ? sigma_sq_S : T.(sigma_sq_S)
    obj_check = _qfm_objective(X_proc, F_norm, L_norm, tau)

    return (
        factors = F_norm,
        loadings = L_norm,
        sigma_sq = sigma_sq,
        objective = obj_check,
        smooth_obj = T(Optim.minimum(res)),
        converged = Optim.converged(res),
        iterations = Optim.iterations(res),
        obj_trace = T[obj_check],
    )
end

# =============================================================================
# Standard Errors (CDG Asymptotic Sandwich & Wild Quantile Bootstrap)
# =============================================================================

"""Adaptive residual scale for Powell kernel density bandwidth (Silverman guard for heavy tails)."""
function _qfm_se_resid_scale(v::AbstractVector{T}) where {T<:AbstractFloat}
    sd = T(Statistics.std(v))
    q75 = T(Statistics.quantile(v, 0.75))
    q25 = T(Statistics.quantile(v, 0.25))
    iqr_s = (q75 - q25) / T(1.3489795003921634)
    if isfinite(sd) && sd > zero(T) && isfinite(iqr_s) && iqr_s > zero(T)
        return min(sd, iqr_s)
    elseif isfinite(sd) && sd > zero(T)
        return sd
    elseif isfinite(iqr_s) && iqr_s > zero(T)
        return iqr_s
    else
        return zero(T)
    end
end

"""Safe Newey–West automatic lag selection across columns of score matrix `S`."""
function _qfm_se_nw_bandwidth(S::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, k = size(S)
    (n < 4 || k == 0) && return 0
    acc = 0
    cnt = 0
    @inbounds for j in 1:k
        col = @view S[:, j]
        r_lag = @view col[1:(n - 1)]
        if dot(r_lag, r_lag) > eps(T)
            acc += optimal_bandwidth_nw(col; kernel=:bartlett)
            cnt += 1
        end
    end
    return cnt == 0 ? 0 : round(Int, acc / cnt)
end

"""Chen–Dolado–Gonzalo (2021, Theorem 1) Powell/Hall–Sheather asymptotic SEs.

Computes `se_loadings` (`N × r`) with optional Newey–West HAC across time `t`
controlled by `hac_bandwidth`, and cross-sectional `se_factors` (`T_obs × r`).
Operates in standardized `X_proc` units whenever `standardized == true` and
excludes exact LP-interpolated zero residuals from Powell kernel density sums
when sufficient non-zero residuals exist."""
function _qfm_se_asymptotic(X_proc::Matrix{T}, F::Matrix{T}, Λ::Matrix{T},
                            tau::T, hac_bandwidth::Union{Symbol,Int}) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    r = size(F, 2)
    resid_proc = X_proc - F * Λ'

    h_T = _hall_sheather_bandwidth(T_obs, tau, T(0.05))
    h_N = _hall_sheather_bandwidth(N, tau, T(0.05))
    inv_sqrt2pi = inv(sqrt(T(2) * T(pi)))
    bw_floor = max(T(1e-8), sqrt(eps(T)))
    ztol_base = max(T(1e-7), sqrt(eps(T)))

    se_loadings = Matrix{T}(undef, N, r)
    D_t_vec = Vector{T}(undef, T_obs)
    S_i = Matrix{T}(undef, T_obs, r)

    for i in 1:N
        u_i = @view resid_proc[:, i]
        sc_i = _qfm_se_resid_scale(u_i)
        bw_i = max(h_T * sc_i, bw_floor)
        inv_bw = one(T) / bw_i
        ztol_i = ztol_base * max(sc_i, one(T))

        n_nz = 0
        @inbounds for t in 1:T_obs
            if abs(u_i[t]) > ztol_i
                n_nz += 1
            end
        end
        skip_zeros = n_nz >= max(r + 2, div(T_obs, 2))
        denom_T = T(skip_zeros ? n_nz : T_obs)

        @inbounds for t in 1:T_obs
            if skip_zeros && abs(u_i[t]) <= ztol_i
                D_t_vec[t] = zero(T)
            else
                z = u_i[t] * inv_bw
                D_t_vec[t] = exp(-T(0.5) * z * z) * inv_bw * inv_sqrt2pi
            end
            psi = tau - (u_i[t] < zero(T) ? one(T) : zero(T))
            for k in 1:r
                S_i[t, k] = psi * F[t, k]
            end
        end
        Phi_i = ((F .* D_t_vec)' * F) / denom_T
        Phi_i_sym = Symmetric((Phi_i + Phi_i') / T(2))
        Phi_i_inv = Matrix{T}(robust_inv(Phi_i_sym; silent=true))

        bw_hac = hac_bandwidth === :auto ? _qfm_se_nw_bandwidth(S_i) : hac_bandwidth
        Omega_i = if bw_hac == 0
            (S_i' * S_i) / T(T_obs)
        else
            Matrix{T}(_psd_project(_hac_meat(S_i, bw_hac, :bartlett))) / T(T_obs)
        end
        V_i = (Phi_i_inv * Omega_i * Phi_i_inv) / T(T_obs)
        @inbounds for k in 1:r
            se_loadings[i, k] = sqrt(max(V_i[k, k], zero(T)))
        end
    end

    se_factors = Matrix{T}(undef, T_obs, r)
    D_i_vec = Vector{T}(undef, N)
    S_t = Matrix{T}(undef, N, r)

    for t in 1:T_obs
        v_t = @view resid_proc[t, :]
        sc_t = _qfm_se_resid_scale(v_t)
        bw_t = max(h_N * sc_t, bw_floor)
        inv_bw = one(T) / bw_t
        ztol_t = ztol_base * max(sc_t, one(T))

        n_nz = 0
        @inbounds for i in 1:N
            if abs(v_t[i]) > ztol_t
                n_nz += 1
            end
        end
        skip_zeros = n_nz >= max(r + 2, div(N, 2))
        denom_N = T(skip_zeros ? n_nz : N)

        @inbounds for i in 1:N
            if skip_zeros && abs(v_t[i]) <= ztol_t
                D_i_vec[i] = zero(T)
            else
                z = v_t[i] * inv_bw
                D_i_vec[i] = exp(-T(0.5) * z * z) * inv_bw * inv_sqrt2pi
            end
            psi = tau - (v_t[i] < zero(T) ? one(T) : zero(T))
            for k in 1:r
                S_t[i, k] = psi * Λ[i, k]
            end
        end
        Psi_t = ((Λ .* D_i_vec)' * Λ) / denom_N
        Psi_t_sym = Symmetric((Psi_t + Psi_t') / T(2))
        Psi_t_inv = Matrix{T}(robust_inv(Psi_t_sym; silent=true))

        Upsilon_t = (S_t' * S_t) / T(N)
        V_t = (Psi_t_inv * Upsilon_t * Psi_t_inv) / T(N)
        @inbounds for k in 1:r
            se_factors[t, k] = sqrt(max(V_t[k, k], zero(T)))
        end
    end

    return se_loadings, se_factors
end

"""Orthogonal Procrustes alignment of `(F_b, Λ_b)` to reference `F_ref`."""
function _qfm_se_procrustes_align(F_b::Matrix{T}, Λ_b::Matrix{T},
                                  F_ref::Matrix{T}) where {T<:AbstractFloat}
    sv = svd(F_b' * F_ref)
    Q = sv.U * sv.V'
    return F_b * Q, Λ_b * Q
end

"""Asymmetric wild quantile bootstrap SEs (Feng, He & Hu 2011) with Procrustes alignment."""
function _qfm_se_bootstrap(X_proc::Matrix{T}, F_hat::Matrix{T}, Λ_hat::Matrix{T},
                           tau::T, n_boot::Int, method::Symbol, h_smooth::T,
                           kernel::Symbol, max_iter::Int, tol_eff::T,
                           threaded::Bool, rng_use::AbstractRNG) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    r = size(F_hat, 2)
    fit_proc = F_hat * Λ_hat'
    abs_resid = abs.(X_proc - fit_proc)
    ztol_base = max(T(1e-7), sqrt(eps(T)))

    # Precompute per-column non-zero |residual| pools so LP-interpolated zeros are not pinned across draws
    col_ztols = Vector{T}(undef, N)
    nz_pools = Vector{Vector{T}}(undef, N)
    for i in 1:N
        col = @view abs_resid[:, i]
        sc_i = _qfm_se_resid_scale(col)
        ztol_i = ztol_base * max(sc_i, one(T))
        col_ztols[i] = ztol_i
        nz_pools[i] = T[x for x in col if x > ztol_i]
    end

    w_neg = -T(2) * tau
    w_pos = T(2) * (one(T) - tau)

    boot_seeds = rand(rng_use, UInt64, n_boot)
    F_draws = Array{T,3}(undef, T_obs, r, n_boot)
    L_draws = Array{T,3}(undef, N, r, n_boot)

    function _run_boot_rep!(b::Int, X_b::Matrix{T}, X_bt::Matrix{T})
        local_rng = Random.Xoshiro(boot_seeds[b])
        @inbounds for i in 1:N
            pool = nz_pools[i]
            ztol_i = col_ztols[i]
            has_pool = !isempty(pool)
            n_pool = length(pool)
            for t in 1:T_obs
                ar = abs_resid[t, i]
                if ar <= ztol_i && has_pool
                    ar = pool[rand(local_rng, 1:n_pool)]
                end
                w = rand(local_rng, T) < tau ? w_neg : w_pos
                X_b[t, i] = fit_proc[t, i] + w * ar
            end
        end
        res_b = if method === :iqr
            transpose!(X_bt, X_b)
            _qfm_iqr_single(X_b, X_bt, F_hat, tau, max_iter, tol_eff, false)
        else
            _qfm_smooth_single(X_b, F_hat, tau, h_smooth, kernel, max_iter, tol_eff)
        end
        F_al, L_al = _qfm_se_procrustes_align(res_b.factors, res_b.loadings, F_hat)
        F_draws[:, :, b] = F_al
        L_draws[:, :, b] = L_al
        return nothing
    end

    if threaded && Threads.nthreads() > 1 && n_boot > 1
        n_workers = min(n_boot, Threads.nthreads())
        X_pool = [Matrix{T}(undef, T_obs, N) for _ in 1:n_workers]
        Xt_pool = [Matrix{T}(undef, N, T_obs) for _ in 1:n_workers]
        base, rem = divrem(n_boot, n_workers)
        Threads.@threads for c in 1:n_workers
            X_b = X_pool[c]
            X_bt = Xt_pool[c]
            start_b = (c - 1) * base + min(c - 1, rem) + 1
            len_b = base + (c <= rem ? 1 : 0)
            for b in start_b:(start_b + len_b - 1)
                _run_boot_rep!(b, X_b, X_bt)
            end
        end
    else
        X_b = Matrix{T}(undef, T_obs, N)
        X_bt = Matrix{T}(undef, N, T_obs)
        for b in 1:n_boot
            _run_boot_rep!(b, X_b, X_bt)
        end
    end

    se_loadings = Matrix{T}(undef, N, r)
    se_factors = Matrix{T}(undef, T_obs, r)
    @inbounds for j in 1:r
        for i in 1:N
            se_loadings[i, j] = T(Statistics.std(@view(L_draws[i, j, :])))
        end
        for t in 1:T_obs
            se_factors[t, j] = T(Statistics.std(@view(F_draws[t, j, :])))
        end
    end
    return se_loadings, se_factors
end

# =============================================================================
# Public Estimation API
# =============================================================================

"""
    _qfm_iqr_trace(X, r, tau=0.5; ...) -> (QuantileFactorModel{T}, Vector{T})

Internal entry point returning both the fitted [`QuantileFactorModel`](@ref) and the
outer-iteration check-loss trajectory `obj_trace` of the winning start.
"""
function _qfm_iqr_trace(X::AbstractMatrix{T}, r::Int, tau::Real=0.5;
                        standardize::Bool=false,
                        robust_scale::Bool=true,
                        init::Symbol=:pca,
                        n_starts::Int=3,
                        max_iter::Int=200,
                        tol::Real=1e-7,
                        threaded::Bool=true,
                        varnames::Union{Nothing,Vector{String}}=nothing,
                        method::Symbol=:iqr,
                        bandwidth::Union{Symbol,Real}=:auto,
                        kernel::Symbol=:gaussian,
                        se::Symbol=:none,
                        hac_bandwidth::Union{Symbol,Int}=:auto,
                        n_boot::Int=200,
                        rng::AbstractRNG=Random.default_rng(),
                        seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)
    tau_T = T(tau)
    (isfinite(tau) && zero(T) < tau_T < one(T)) ||
        throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
    validate_option(init, "init", (:pca, :quantile_svd, :random))
    validate_option(method, "method", (:iqr, :smoothed_iqr))
    validate_option(kernel, "kernel", (:gaussian, :logistic, :epanechnikov))
    validate_option(se, "se", (:none, :asymptotic, :bootstrap))
    if bandwidth isa Symbol
        bandwidth === :auto || throw(ArgumentError(
            "bandwidth must be :auto or a positive Real, got :$bandwidth"))
    else
        (!(bandwidth isa Bool) && isfinite(bandwidth) && bandwidth > 0) ||
            throw(ArgumentError("bandwidth must be :auto or a positive Real, got $bandwidth"))
    end
    if hac_bandwidth isa Symbol
        hac_bandwidth === :auto || throw(ArgumentError(
            "hac_bandwidth must be :auto or a non-negative Int, got :$hac_bandwidth"))
    else
        hac_bandwidth >= 0 || throw(ArgumentError(
            "hac_bandwidth must be :auto or a non-negative Int, got $hac_bandwidth"))
    end
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("tol must be > 0, got $tol"))
    if se === :bootstrap
        n_boot >= 2 || throw(ArgumentError("n_boot must be >= 2 when se=:bootstrap, got $n_boot"))
    else
        n_boot >= 1 || throw(ArgumentError("n_boot must be >= 1, got $n_boot"))
    end

    vn = something(varnames, ["Var $i" for i in 1:N])
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $N columns"))

    rng_use = _resolve_repro_rng(rng, seed)
    X_orig = Matrix{T}(X)

    if standardize
        X_proc, center, scale = _qfm_standardize(X_orig, robust_scale)
    else
        X_proc = X_orig
        center = zeros(T, N)
        scale = ones(T, N)
    end
    X_proc_t = Matrix{T}(X_proc')

    tol_eff = max(T(tol), T(10) * eps(T))
    seeds = rand(rng_use, UInt64, n_starts)

    h_smooth = zero(T)
    best = nothing
    for s in 1:n_starts
        local_rng = Random.Xoshiro(seeds[s])
        start_init = s == 1 ? init : :random
        F_init = _qfm_init_factors(X_proc, r, tau_T, start_init, local_rng)
        if method === :iqr
            res = _qfm_iqr_single(X_proc, X_proc_t, F_init, tau_T, max_iter, tol_eff, threaded)
            if best === nothing || res.objective < best.objective
                best = res
            end
        else
            if s == 1
                h_smooth = _qfm_smooth_bandwidth(X_proc, F_init, bandwidth)
            end
            res = _qfm_smooth_single(X_proc, F_init, tau_T, h_smooth, kernel, max_iter, tol_eff)
            if best === nothing || res.smooth_obj < best.smooth_obj
                best = res
            end
        end
    end

    fit_proc = best.factors * best.loadings'
    common_component = standardize ? (fit_proc .* scale' .+ center') : fit_proc
    resid = X_orig - common_component
    pr2 = _qfm_pseudo_r2(X_proc, best.factors, best.loadings, tau_T)

    se_loadings, se_factors = if se === :asymptotic
        _qfm_se_asymptotic(X_proc, best.factors, best.loadings, tau_T, hac_bandwidth)
    elseif se === :bootstrap
        _qfm_se_bootstrap(X_proc, best.factors, best.loadings, tau_T, n_boot, method,
                          h_smooth, kernel, max_iter, tol_eff, threaded, rng_use)
    else
        (nothing, nothing)
    end

    model = QuantileFactorModel{T}(
        X_orig, tau_T, best.factors, best.loadings, best.sigma_sq,
        common_component, resid, best.objective, pr2, r, method, standardize,
        best.converged, best.iterations, se_loadings, se_factors, vn, nothing,
    )
    model = _with_manifest(model, capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "r" => r,
            "tau" => Float64(tau),
            "init" => String(init),
            "n_starts" => n_starts,
            "method" => String(method),
            "kernel" => String(kernel),
            "se" => String(se),
            "n_boot" => n_boot,
        )))
    return model, best.obj_trace
end

@float_fallback _qfm_iqr_trace X

"""
    estimate_qfm(X, r, tau=0.5; standardize=false, robust_scale=true, init=:pca,
                 n_starts=3, max_iter=200, tol=1e-7, threaded=true,
                 varnames=nothing, method=:iqr, bandwidth=:auto, kernel=:gaussian,
                 se=:none, hac_bandwidth=:auto, n_boot=200,
                 rng=Random.default_rng(), seed=nothing) -> QuantileFactorModel

Estimate a Quantile Factor Model (Chen, Dolado & Gonzalo 2021) at quantile level `tau`
using Iterative Quantile Regression (`:iqr`) or Convolution-Smoothed QFA (`:smoothed_iqr`).

```math
Q_{\\tau}(X_{it} \\mid f_t) = \\lambda_i(\\tau)' f_t(\\tau)
```

where:
- ``X_{it}`` is the panel observation for unit ``i`` at time ``t``
- ``\\tau \\in (0, 1)`` is the quantile level
- ``\\lambda_i(\\tau) \\in \\mathbb{R}^r`` is the quantile loading vector for unit ``i``
- ``f_t(\\tau) \\in \\mathbb{R}^r`` is the quantile factor vector at time ``t``

# Arguments
- `X::AbstractMatrix` — data panel (`T_obs × N`)
- `r::Int` — number of quantile factors (`1 ≤ r ≤ min(T_obs, N)`)
- `tau::Real = 0.5` — quantile level strictly in `(0, 1)`

# Keyword Arguments
- `standardize::Bool = false` — standardize columns before fitting (common component is
  returned in raw `X` units)
- `robust_scale::Bool = true` — when `standardize=true`, use median and normalized IQR
  (with MAD and SD fallbacks) instead of mean and SD
- `init::Symbol = :pca` — start-1 initialization (`:pca`, `:quantile_svd`, or `:random`)
- `n_starts::Int = 3` — number of multi-start initializations (starts `2..n_starts` use
  reproducible random orthonormal Gaussian factors)
- `max_iter::Int = 200` — maximum outer alternating IQR or L-BFGS iterations
- `tol::Real = 1e-7` — convergence tolerance on relative objective / gradient norm
- `threaded::Bool = true` — parallelize batched quantile regressions and bootstrap draws across threads
- `varnames::Union{Nothing,Vector{String}} = nothing` — variable names (default `"Var 1"`, …)
- `method::Symbol = :iqr` — `:iqr` (exact alternating IPM) or `:smoothed_iqr` (convolution-smoothed L-BFGS)
- `bandwidth::Union{Symbol,Real} = :auto` — smoothing bandwidth `h` for `method=:smoothed_iqr`
  (`:auto` uses `max(0.05, std(resid_init) * (log(N * T_obs) / min(N, T_obs))^0.25)`)
- `kernel::Symbol = :gaussian` — smoothing kernel (`:gaussian`, `:logistic`, or `:epanechnikov`)
- `se::Symbol = :none` — standard errors (`:none`, `:asymptotic` CDG Theorem 1 sandwich, or
  `:bootstrap` asymmetric wild quantile bootstrap with Procrustes alignment)
- `hac_bandwidth::Union{Symbol,Int} = :auto` — Newey–West Bartlett lag truncation across `t`
  for `se_loadings` (`:auto` selects via Andrews 1991 plug-in; `0` disables HAC lags)
- `n_boot::Int = 200` — bootstrap replications when `se=:bootstrap`
- `rng::AbstractRNG = Random.default_rng()` — RNG for random starts and bootstrap draws
- `seed::Union{Integer,Nothing} = nothing` — integer seed overriding `rng` via `Xoshiro(seed)`

# Returns
A [`QuantileFactorModel`](@ref) containing normalized factors (`F'F / T = I_r`), loadings
(`Λ'Λ / N = diag(σ₁² ≥ … ≥ σ_r²)`), common component, residuals, check-loss objective,
Koenker–Machado pseudo-R², optional `se_loadings` and `se_factors`, and reproducibility manifest.

# Example
```julia
X = randn(200, 80)
qfm = estimate_qfm(X, 2, 0.9; method=:smoothed_iqr, se=:asymptotic, seed=42)
report(qfm)
```

# References
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models.
  *Econometrica*, 89(2), 875–910.
- Fernandes, M., Guerre, E., & Horta, E. (2021). Smoothing quantile regressions.
  *Journal of Business & Economic Statistics*, 39(1), 338–357.
- He, Y., Kong, X., Yu, L., & Zhang, X. (2023). Quantile factor analysis for
  large-dimensional time series with smoothed quantile regression.
  *Journal of Econometrics*, 232(2), 367–390.
"""
function estimate_qfm(X::AbstractMatrix{T}, r::Int, tau::Real=0.5;
                      standardize::Bool=false,
                      robust_scale::Bool=true,
                      init::Symbol=:pca,
                      n_starts::Int=3,
                      max_iter::Int=200,
                      tol::Real=1e-7,
                      threaded::Bool=true,
                      varnames::Union{Nothing,Vector{String}}=nothing,
                      method::Symbol=:iqr,
                      bandwidth::Union{Symbol,Real}=:auto,
                      kernel::Symbol=:gaussian,
                      se::Symbol=:none,
                      hac_bandwidth::Union{Symbol,Int}=:auto,
                      n_boot::Int=200,
                      rng::AbstractRNG=Random.default_rng(),
                      seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    model, _ = _qfm_iqr_trace(X, r, tau;
                              standardize=standardize,
                              robust_scale=robust_scale,
                              init=init,
                              n_starts=n_starts,
                              max_iter=max_iter,
                              tol=tol,
                              threaded=threaded,
                              varnames=varnames,
                              method=method,
                              bandwidth=bandwidth,
                              kernel=kernel,
                              se=se,
                              hac_bandwidth=hac_bandwidth,
                              n_boot=n_boot,
                              rng=rng,
                              seed=seed)
    return model
end

@float_fallback estimate_qfm X

# =============================================================================
# Accessors & StatsAPI Interface
# =============================================================================

"""Estimated quantile factor matrix (`T_obs × r`, normalized `F'F / T_obs = I_r`)."""
factors(m::QuantileFactorModel) = m.factors

"""Estimated quantile loading matrix (`N × r`, normalized `Λ'Λ / N = diag(sigma_sq)`)."""
loadings(m::QuantileFactorModel) = m.loadings

"""Predicted conditional quantile common component (`T_obs × N`, in raw `X` units)."""
StatsAPI.predict(m::QuantileFactorModel) = m.common_component

"""Fitted conditional quantile common component (`T_obs × N`, in raw `X` units)."""
StatsAPI.fitted(m::QuantileFactorModel) = m.common_component

"""Residuals: `X - common_component`."""
StatsAPI.residuals(m::QuantileFactorModel) = m.residuals

"""Koenker–Machado pseudo-R² for each variable (`length N`)."""
StatsAPI.r2(m::QuantileFactorModel) = m.pseudo_r2

"""Number of time-series observations."""
StatsAPI.nobs(m::QuantileFactorModel) = size(m.X, 1)

"""Effective degrees of freedom: `N*r + T*r - r²`."""
StatsAPI.dof(m::QuantileFactorModel) = size(m.X, 2) * m.r + size(m.X, 1) * m.r - m.r^2

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::QuantileFactorModel{T}) where {T}
    Tobs, N = size(m.X)
    has_se = m.se_loadings !== nothing || m.se_factors !== nothing
    spec = Any[
        "Quantile (τ)"  _fmt(m.tau; digits=3);
        "Factors"       m.r;
        "Variables"     N;
        "Observations"  Tobs;
        "Method"        string(m.method);
        "Standardized"  m.standardized ? "Yes" : "No";
        "Check loss"    _fmt(m.objective; digits=4);
        "Converged"     m.converged ? "Yes" : "No";
        "Iterations"    m.iterations
    ]
    if has_se
        se_str = (m.manifest !== nothing && haskey(m.manifest.settings, "se")) ?
                 string(m.manifest.settings["se"]) : "computed"
        spec = vcat(spec, Any["Std. errors" se_str])
    end
    _pretty_table(io, spec;
        title = "Quantile Factor Model (r=$(m.r), τ=$(_fmt(m.tau; digits=3)))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    n_show = min(m.r, 5)
    tot_sig = sum(m.sigma_sq)
    sig_data = Matrix{Any}(undef, n_show, 4)
    cum_share = zero(T)
    for j in 1:n_show
        share = tot_sig > zero(T) ? m.sigma_sq[j] / tot_sig : zero(T)
        cum_share += share
        sig_data[j, 1] = "Factor $j"
        sig_data[j, 2] = _fmt(m.sigma_sq[j]; digits=4)
        sig_data[j, 3] = _fmt_pct(share)
        sig_data[j, 4] = _fmt_pct(cum_share)
    end
    _pretty_table(io, sig_data;
        title = "Factor Strength (Λ'Λ / N)",
        column_labels = ["", "σ²", "Share", "Cumulative"],
        alignment = [:l, :r, :r, :r],
    )

    if has_se
        se_sum = Matrix{Any}(undef, n_show, 5)
        for j in 1:n_show
            se_sum[j, 1] = "Factor $j"
            if m.se_loadings !== nothing
                sl = @view m.se_loadings[:, j]
                se_sum[j, 2] = _fmt(Statistics.median(sl); digits=4)
                se_sum[j, 3] = _fmt(maximum(sl); digits=4)
            else
                se_sum[j, 2] = "—"
                se_sum[j, 3] = "—"
            end
            if m.se_factors !== nothing
                sf = @view m.se_factors[:, j]
                se_sum[j, 4] = _fmt(Statistics.median(sf); digits=4)
                se_sum[j, 5] = _fmt(maximum(sf); digits=4)
            else
                se_sum[j, 4] = "—"
                se_sum[j, 5] = "—"
            end
        end
        _pretty_table(io, se_sum;
            title = "Standard Errors Summary",
            column_labels = ["", "Med SE(Λ)", "Max SE(Λ)", "Med SE(F)", "Max SE(F)"],
            alignment = [:l, :r, :r, :r, :r],
        )
    end

    n_factors_show = min(m.r, 3)
    n_top = min(5, N)
    names = m.varnames
    degen_tol = sqrt(eps(T))
    nd = Distributions.Normal()
    for f in 1:n_factors_show
        loadings_f = m.loadings[:, f]
        sorted_idx = sortperm(abs.(loadings_f); rev=true)
        top_idx = sorted_idx[1:n_top]
        if m.se_loadings !== nothing
            se_f = @view m.se_loadings[:, f]
            load_data = Matrix{Any}(undef, n_top, 5)
            for (row, idx) in enumerate(top_idx)
                est = loadings_f[idx]
                sei = se_f[idx]
                load_data[row, 1] = names[idx]
                load_data[row, 2] = _fmt(est)
                load_data[row, 3] = _fmt(sei)
                if !isfinite(est) || !isfinite(sei) || sei <= degen_tol * max(abs(est), one(T))
                    load_data[row, 4] = "—"
                    load_data[row, 5] = "—"
                else
                    zval = est / sei
                    pval = T(2) * (one(T) - T(Distributions.cdf(nd, abs(zval))))
                    load_data[row, 4] = _fmt(zval; digits=3)
                    load_data[row, 5] = _format_pvalue(pval)
                end
            end
            _pretty_table(io, load_data;
                title = "Top Loadings — Factor $f",
                column_labels = ["Variable", "Loading", "Std.Err.", "z", "P>|z|"],
                alignment = [:l, :r, :r, :r, :r],
            )
        else
            load_data = Matrix{Any}(undef, n_top, 2)
            for (row, idx) in enumerate(top_idx)
                load_data[row, 1] = names[idx]
                load_data[row, 2] = _fmt(loadings_f[idx])
            end
            _pretty_table(io, load_data;
                title = "Top Loadings — Factor $f",
                column_labels = ["Variable", "Loading"],
                alignment = [:l, :r],
            )
        end
    end

    mean_pr2 = T(Statistics.mean(m.pseudo_r2))
    med_pr2 = T(Statistics.median(m.pseudo_r2))
    println(io, "Pseudo-R² (Koenker–Machado): mean = ",
            _fmt(mean_pr2; digits=4), ", median = ", _fmt(med_pr2; digits=4))
    _manifest_footer(io, m.manifest)
end

"""
    report(m::QuantileFactorModel; repro=false)

Print a comprehensive summary of the fitted quantile factor model.
Pass `repro=true` to include the reproducibility manifest footer.
"""
report(m::QuantileFactorModel; repro::Bool=false) = report(stdout, m; repro=repro)
report(io::IO, m::QuantileFactorModel; repro::Bool=false) =
    (show(IOContext(io, :repro => repro), m); nothing)
