# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Quantile Factor Model (QFM) via Iterative Quantile Regression (IQR).

Implements Chen, Dolado & Gonzalo (2021):
``Q_{\\tau}(X_{it} \\mid f_t) = \\lambda_i(\\tau)' f_t(\\tau)``

Estimated by alternating batched quantile regressions (`_qreg_batch!`) with
non-increasing check-loss guards and canonical CDG (§2.1) factor/loading
normalization:
``F'F / T = I_r, \\qquad \\Lambda'\\Lambda / N = \\mathrm{diag}(\\sigma_1^2 \\ge \\dots \\ge \\sigma_r^2 \\ge 0).``

References:
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models.
  *Econometrica*, 89(2), 875–910.
- Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related inference
  processes for quantile regression. *Journal of the American Statistical Association*,
  94(448), 1296–1310.
"""

using LinearAlgebra, Statistics, StatsAPI, Random

# =============================================================================
# Quantile Factor Model Type
# =============================================================================

"""
    QuantileFactorModel{T<:AbstractFloat} <: AbstractFactorModel

Quantile factor model (Chen, Dolado & Gonzalo 2021) estimated at quantile `tau` via
Iterative Quantile Regression (`:iqr`).

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
- `method::Symbol` — estimation method (`:iqr`)
- `standardized::Bool` — whether `X` was standardized prior to estimation
- `converged::Bool` — whether the winning IQR run satisfied the stopping rule
- `iterations::Int` — outer IQR iterations performed by the winning run
- `se_loadings::Union{Nothing,Matrix{T}}` — optional asymptotic loading SEs
- `se_factors::Union{Nothing,Matrix{T}}` — optional asymptotic factor SEs
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
                        rng::AbstractRNG=Random.default_rng(),
                        seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)
    tau_T = T(tau)
    (isfinite(tau) && zero(T) < tau_T < one(T)) ||
        throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))
    validate_option(init, "init", (:pca, :quantile_svd, :random))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("tol must be > 0, got $tol"))

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

    best = nothing
    for s in 1:n_starts
        local_rng = Random.Xoshiro(seeds[s])
        start_init = s == 1 ? init : :random
        F_init = _qfm_init_factors(X_proc, r, tau_T, start_init, local_rng)
        res = _qfm_iqr_single(X_proc, X_proc_t, F_init, tau_T, max_iter, tol_eff, threaded)
        if best === nothing || res.objective < best.objective
            best = res
        end
    end

    fit_proc = best.factors * best.loadings'
    common_component = standardize ? (fit_proc .* scale' .+ center') : fit_proc
    resid = X_orig - common_component
    pr2 = _qfm_pseudo_r2(X_proc, best.factors, best.loadings, tau_T)

    model = QuantileFactorModel{T}(
        X_orig, tau_T, best.factors, best.loadings, best.sigma_sq,
        common_component, resid, best.objective, pr2, r, :iqr, standardize,
        best.converged, best.iterations, nothing, nothing, vn, nothing,
    )
    model = _with_manifest(model, capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "r" => r,
            "tau" => Float64(tau),
            "init" => String(init),
            "n_starts" => n_starts,
        )))
    return model, best.obj_trace
end

@float_fallback _qfm_iqr_trace X

"""
    estimate_qfm(X, r, tau=0.5; standardize=false, robust_scale=true, init=:pca,
                 n_starts=3, max_iter=200, tol=1e-7, threaded=true,
                 varnames=nothing, rng=Random.default_rng(), seed=nothing) -> QuantileFactorModel

Estimate a Quantile Factor Model (Chen, Dolado & Gonzalo 2021) at quantile level `tau`
using Iterative Quantile Regression (`:iqr`).

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
- `max_iter::Int = 200` — maximum outer alternating quantile-regression iterations
- `tol::Real = 1e-7` — convergence tolerance on relative objective change and factor
  subspace change
- `threaded::Bool = true` — parallelize batched quantile regressions across threads
- `varnames::Union{Nothing,Vector{String}} = nothing` — variable names (default `"Var 1"`, …)
- `rng::AbstractRNG = Random.default_rng()` — RNG for random starts
- `seed::Union{Integer,Nothing} = nothing` — integer seed overriding `rng` via `Xoshiro(seed)`

# Returns
A [`QuantileFactorModel`](@ref) containing normalized factors (`F'F / T = I_r`), loadings
(`Λ'Λ / N = diag(σ₁² ≥ … ≥ σ_r²)`), common component, residuals, check-loss objective,
Koenker–Machado pseudo-R², and reproducibility manifest.

# Example
```julia
X = randn(200, 80)
qfm = estimate_qfm(X, 2, 0.9; init=:quantile_svd, seed=42)
report(qfm)
```

# References
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models.
  *Econometrica*, 89(2), 875–910.
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

    n_factors_show = min(m.r, 3)
    n_top = min(5, N)
    names = m.varnames
    for f in 1:n_factors_show
        loadings_f = m.loadings[:, f]
        sorted_idx = sortperm(abs.(loadings_f); rev=true)
        top_idx = sorted_idx[1:n_top]
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
