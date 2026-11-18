# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Forni, Hallin, Lippi & Zaffaroni (2015, 2017) One-Sided Infinite-Dimensional GDFM.

Implements the random-permutation blockwise VAR estimator for infinite-dimensional
factor spaces (generalized dynamic factor models without finite static factor rank `r`).

References:
- Forni, M., Hallin, M., Lippi, M., & Zaffaroni, P. (2015). Dynamic factor models with
  infinite-dimensional factor spaces: One-sided representations. Journal of Econometrics,
  185(2), 359–371.
- Forni, M., Hallin, M., Lippi, M., & Zaffaroni, P. (2017). Dynamic factor models with
  infinite-dimensional factor spaces: Asymptotic analysis. Journal of Econometrics,
  199(1), 74–92.
"""

using LinearAlgebra, Statistics, Random, StatsAPI

# =============================================================================
# FHLZModel Type
# =============================================================================

"""
    FHLZModel{T<:AbstractFloat} <: AbstractFactorModel

One-sided infinite-dimensional Generalized Dynamic Factor Model estimated via
Forni, Hallin, Lippi & Zaffaroni (2015, 2017) random-permutation blockwise VARs.

# Fields
- `X::Matrix{T}`: Original panel data (`T_obs × N`)
- `q::Int`: Number of dynamic common shocks
- `common_component::Matrix{T}`: One-sided common component `χ̂_t` (`T_eff × N`, `T_eff = T_obs - p_max`)
- `idiosyncratic::Matrix{T}`: Idiosyncratic component `X[(p_max+1):end, :] - common_component` (`T_eff × N`)
- `innovations::Matrix{T}`: Orthonormal one-sided common shocks `û_t` (`T_eff × q`)
- `impulse_filters::Array{T,3}`: One-sided MA filters `B̂_0, …, B̂_{H-1}` (`H_irf × N × q`)
- `p_blocks::Matrix{Int}`: Selected block VAR lag orders (`n_perm × n_blocks`)
- `n_perm::Int`: Number of cross-sectional permutations averaged
- `bandwidth::Int`: Lag-window truncation parameter `M`
- `kernel::Symbol`: Spectral lag-window kernel (`:bartlett`, `:parzen`, `:tukey`)
- `standardized::Bool`: Whether estimation standardized `X` prior to filtering
- `variance_explained::Vector{T}`: Share of spectral trace explained by each of the `q` dynamic factors
- `varnames::Vector{String}`: Panel variable names (length `N`)
- `manifest::Union{ReproManifest,Nothing}`: Reproducibility manifest
"""
struct FHLZModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    q::Int
    common_component::Matrix{T}     # T_eff × N one-sided χ̂_t (T_eff = T_obs - p_max)
    idiosyncratic::Matrix{T}        # X[(p_max+1):end, :] - common_component
    innovations::Matrix{T}          # T_eff × q orthonormal one-sided common shocks û_t
    impulse_filters::Array{T,3}     # H_irf × N × q one-sided MA coefficients B̂_0..B̂_{H-1}
    p_blocks::Matrix{Int}           # n_perm × n_blocks selected block VAR lag orders
    n_perm::Int
    bandwidth::Int
    kernel::Symbol
    standardized::Bool
    variance_explained::Vector{T}   # length q
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

# =============================================================================
# Public Estimator
# =============================================================================

"""
    estimate_gdfm_fhlz(X, q; kwargs...) -> FHLZModel

Estimate the one-sided infinite-dimensional Generalized Dynamic Factor Model of
Forni, Hallin, Lippi & Zaffaroni (2015, 2017) using random-permutation blockwise
Yule–Walker VARs, with optional EM missing-data and ragged-edge imputation:

```math
X_t = \\chi_t + \\xi_t, \\qquad \\chi_t = \\sum_{h=0}^{\\infty} B_h u_{t-h}
```

where:
- ``X_t`` is the ``N \\times 1`` observed panel vector at time ``t``
- ``\\chi_t`` is the ``N \\times 1`` one-sided common component
- ``\\xi_t`` is the ``N \\times 1`` weakly correlated idiosyncratic component
- ``B_h`` is the ``N \\times q`` impulse-response filter matrix at lag ``h``
- ``u_t`` is the ``q \\times 1`` vector of orthonormal dynamic common innovations

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T_obs × N`)
- `q::Int`: Number of dynamic common shocks (`1 ≤ q ≤ min(T_obs, N)`)

# Keyword Arguments
- `bandwidth::Int=0`: Spectral lag-window truncation `M` (`0` uses `max(3, round(0.5√T))`)
- `kernel::Symbol=:bartlett`: Spectral kernel (`:bartlett`, `:parzen`, `:tukey`)
- `p_max::Int=5`: Maximum block VAR lag order (`≥ 1`)
- `lag_criterion::Symbol=:bic`: Block lag selection criterion (`:bic` or `:aic`)
- `n_perm::Int=30`: Number of cross-sectional permutations to average (`≥ 1`)
- `irf_horizon::Int=40`: Number of one-sided MA coefficients `B̂_0..B̂_{H-1}` (`≥ 1`)
- `standardize::Bool=true`: Standardize series before estimation, then unstandardize outputs
- `threaded::Bool=true`: Parallelize across permutations using `Threads.@threads`
- `varnames::Union{Nothing,Vector{String}}=nothing`: Panel variable names (length `N`)
- `seed::Union{Integer,Nothing}=nothing`: Optional RNG seed recorded in `manifest`
- `rng::AbstractRNG=Random.default_rng()`: RNG for permutation draws
- `allow_missing::Bool=false`: Allow `NaN` entries (interior or ragged-edge) and impute via
  cross-sectional EM-PCA warm-start followed by EM-FHLZ iteration (`Inf`/`-Inf` and
  all-`NaN` rows or columns are rejected)
- `em_max_iter::Int=50`: Maximum EM iterations when `allow_missing=true` and `X` contains `NaN`
- `em_tol::Real=1e-4`: Relative convergence tolerance on missing-cell updates
  ``\\max |\\Delta \\hat{\\chi}_{\\text{missing}}|``

# Returns
[`FHLZModel`](@ref) storing one-sided common/idiosyncratic components, orthonormal
innovations, impulse filters, and (when `allow_missing=true`) the converged finite
EM-imputed panel in `model.X`.
"""
function estimate_gdfm_fhlz(
    X::AbstractMatrix{T},
    q::Int;
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    p_max::Int=5,
    lag_criterion::Symbol=:bic,
    n_perm::Int=30,
    irf_horizon::Int=40,
    standardize::Bool=true,
    threaded::Bool=true,
    varnames::Union{Nothing,Vector{String}}=nothing,
    seed::Union{Integer,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    allow_missing::Bool=false,
    em_max_iter::Int=50,
    em_tol::Real=1e-4,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    if allow_missing
        _gdfm_validate_missing_data(X, "X")
    else
        _validate_data(X, "X")
    end
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))
    validate_option(lag_criterion, "lag_criterion", (:bic, :aic))
    p_max >= 1 || throw(ArgumentError("p_max must be >= 1, got $p_max"))
    n_perm >= 1 || throw(ArgumentError("n_perm must be >= 1, got $n_perm"))
    irf_horizon >= 1 || throw(ArgumentError("irf_horizon must be >= 1, got $irf_horizon"))
    em_max_iter >= 1 || throw(ArgumentError("em_max_iter must be >= 1, got $em_max_iter"))
    em_tol > 0 || throw(ArgumentError("em_tol must be > 0, got $em_tol"))

    p_max_use = min(p_max, T_obs - 2)
    T_eff = T_obs - p_max_use
    T_eff >= max(2, q) || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for p_max=$p_max_use and q=$q (need T - p_max >= max(2, q))"))

    vn = something(varnames, ["Var $i" for i in 1:N])
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $N columns"))

    rng = _resolve_repro_rng(rng, seed)
    M_bw = bandwidth > 0 ? bandwidth : _select_lag_window(T_obs)
    # Pre-generate per-permutation seeds sequentially for thread and EM determinism
    seeds = rand(rng, UInt64, n_perm)

    has_missing = allow_missing && any(isnan, X)
    X_orig = if has_missing
        _fhlz_em_impute(
            X, q, M_bw, kernel, p_max_use, lag_criterion,
            n_perm, irf_horizon, standardize, threaded, seeds,
            em_max_iter, T(em_tol))
    else
        Matrix{T}(X)
    end

    core = _fhlz_core(
        X_orig, q, M_bw, kernel, p_max_use, lag_criterion,
        n_perm, irf_horizon, standardize, threaded, seeds)
    idiosyncratic = X_orig[(p_max_use + 1):end, :] - core.common_component

    model = FHLZModel{T}(
        X_orig, q, core.common_component, idiosyncratic, core.U_final, core.B_final,
        core.p_blocks, n_perm, M_bw, kernel, standardize, core.var_explained, vn, nothing,
    )
    settings = Dict{String,Any}(
        "q" => q,
        "bandwidth" => M_bw,
        "kernel" => String(kernel),
        "p_max" => p_max_use,
        "lag_criterion" => String(lag_criterion),
        "n_perm" => n_perm,
        "irf_horizon" => irf_horizon,
        "standardize" => standardize,
        "threaded" => threaded,
    )
    allow_missing && (settings["allow_missing"] = true)
    return _with_manifest(model, capture_manifest(; seed=seed, settings=settings))
end

@float_fallback estimate_gdfm_fhlz X

# =============================================================================
# Internal Helpers
# =============================================================================

"""Core complete-panel FHLZ estimator returning common component, innovations, filters, and spectra."""
function _fhlz_core(
    X_orig::Matrix{T},
    q::Int,
    M_bw::Int,
    kernel::Symbol,
    p_max_use::Int,
    lag_criterion::Symbol,
    n_perm::Int,
    irf_horizon::Int,
    standardize::Bool,
    threaded::Bool,
    seeds::Vector{UInt64},
) where {T<:AbstractFloat}
    T_obs, N = size(X_orig)
    μ = vec(mean(X_orig; dims=1))
    σ = max.(vec(std(X_orig; dims=1)), T(1e-10))
    X_proc = standardize ? (X_orig .- μ') ./ σ' : copy(X_orig)

    # Step 1: Lag-window spectral estimation, common spectral density, and Γ_χ(0..p_max_use)
    frequencies, spectral_X = _fhlz_lag_window_spectrum(X_proc, M_bw, kernel, p_max_use)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    evals_q = max.(eigenvalues[1:q, :], zero(T))
    spectral_chi = _compute_common_spectral_density(loadings, evals_q)
    var_explained = _compute_variance_explained(eigenvalues, q)

    Γ_chi_pos = Vector{Matrix{T}}(undef, p_max_use + 1)
    @inbounds for k in 0:p_max_use
        Gk = _gamma_from_spectrum(spectral_chi, frequencies, k)
        if k == 0
            Gk = Matrix{T}((Gk + Gk') / T(2))
        end
        Γ_chi_pos[k + 1] = Gk
    end

    n_blocks = max(1, div(N, q + 1))
    PermResultT = NamedTuple{
        (:p_blocks_m, :chi_m, :U_m, :B_m),
        Tuple{Vector{Int},Matrix{T},Matrix{T},Array{T,3}},
    }
    perm_results = Vector{PermResultT}(undef, n_perm)

    # Steps 3–7: Blockwise VAR Yule–Walker, causal filtering, whitening, and MA inversion
    if threaded && Threads.nthreads() > 1 && n_perm > 1
        Threads.@threads for m in 1:n_perm
            perm_results[m] = _fhlz_run_permutation(
                X_proc, Γ_chi_pos, q, p_max_use, lag_criterion, irf_horizon, seeds[m])
        end
    else
        for m in 1:n_perm
            perm_results[m] = _fhlz_run_permutation(
                X_proc, Γ_chi_pos, q, p_max_use, lag_criterion, irf_horizon, seeds[m])
        end
    end

    # Step 8: Procrustes alignment to permutation 1, averaging, re-orthonormalization, unstandardizing
    p_blocks = Matrix{Int}(undef, n_perm, n_blocks)
    U_ref = perm_results[1].U_m
    chi_acc = copy(perm_results[1].chi_m)
    U_acc = copy(U_ref)
    B_acc = copy(perm_results[1].B_m)
    p_blocks[1, :] .= perm_results[1].p_blocks_m

    @inbounds for m in 2:n_perm
        res_m = perm_results[m]
        p_blocks[m, :] .= res_m.p_blocks_m
        sv = svd(res_m.U_m' * U_ref)
        O_m = sv.U * sv.Vt
        chi_acc .+= res_m.chi_m
        mul!(U_acc, res_m.U_m, O_m, one(T), one(T))
        for h in 1:irf_horizon
            mul!(@view(B_acc[h, :, :]), @view(res_m.B_m[h, :, :]), O_m, one(T), one(T))
        end
    end

    inv_perm = one(T) / T(n_perm)
    chi_bar = chi_acc .* inv_perm
    U_bar = U_acc .* inv_perm
    B_bar = B_acc .* inv_perm

    # Re-orthonormalize averaged innovations and adjust impulse filters cleanly
    L_bar, L_bar_inv = _fhlz_symmetric_whitening(U_bar)
    U_final = U_bar * L_bar_inv
    B_final = similar(B_bar)
    @inbounds for h in 1:irf_horizon
        B_final[h, :, :] = @view(B_bar[h, :, :]) * L_bar
    end

    if standardize
        common_component = chi_bar .* σ' .+ μ'
        @inbounds for h in 1:irf_horizon
            B_final[h, :, :] .*= σ
        end
    else
        common_component = chi_bar
    end

    (
        common_component = common_component,
        U_final = U_final,
        B_final = B_final,
        p_blocks = p_blocks,
        var_explained = var_explained,
        frequencies = frequencies,
        spectral_X = spectral_X,
        spectral_chi = spectral_chi,
        μ = μ,
        σ = σ,
    )
end

"""EM missing-data imputation for `estimate_gdfm_fhlz(..., allow_missing=true)`."""
function _fhlz_em_impute(
    X::AbstractMatrix{T},
    q::Int,
    M_bw::Int,
    kernel::Symbol,
    p_max_use::Int,
    lag_criterion::Symbol,
    n_perm::Int,
    irf_horizon::Int,
    standardize::Bool,
    threaded::Bool,
    seeds::Vector{UInt64},
    em_max_iter::Int,
    em_tol::T,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    X_cur, obs_mask, miss_mask, scale_obs = _gdfm_init_em_pca(
        X, min(2 * q, max(1, N - 1)), standardize; tol=min(em_tol, T(1e-4)))
    X_raw = Matrix{T}(X)
    L_em = min(max(p_max_use, 2), max(T_obs - 1, 0))
    t_bnd_start = max(1, T_obs - 2)
    wk_rows = Int[]
    @inbounds for t in 1:T_obs
        if (t <= p_max_use || t >= t_bnd_start) && any(!, @view(obs_mask[t, :]))
            push!(wk_rows, t)
        end
    end

    for _ in 1:em_max_iter
        prev_miss = X_cur[miss_mask]
        core = _fhlz_core(
            X_cur, q, M_bw, kernel, p_max_use, lag_criterion,
            n_perm, irf_horizon, standardize, threaded, seeds)

        X_next = copy(X_cur)
        @inbounds for t in (p_max_use + 1):(t_bnd_start - 1)
            t_eff = t - p_max_use
            for j in 1:N
                obs_mask[t, j] || (X_next[t, j] = core.common_component[t_eff, j])
            end
        end

        if !isempty(wk_rows)
            k_max_wk = min(max(T_obs - 1, 0), 2 * L_em)
            Γ_X_pos, Γ_chi_pos = _gdfm_spectral_autocovariances(
                core.spectral_X, core.spectral_chi, core.frequencies, :lag_window, T_obs, k_max_wk)
            σ_use = standardize ? core.σ : ones(T, N)
            X_obs_proc = (X_raw .- core.μ') ./ σ_use'
            for t in wk_rows
                t_lo = max(1, t - L_em)
                t_hi = min(T_obs, t + L_em)
                chi_bnd, _ = _gdfm_wk_project(
                    X_obs_proc, obs_mask, Γ_X_pos, Γ_chi_pos, t_lo, t_hi, t)
                @inbounds for j in 1:N
                    obs_mask[t, j] || (X_next[t, j] = chi_bnd[j] * σ_use[j] + core.μ[j])
                end
            end
        end

        X_cur .= X_next
        new_miss = X_cur[miss_mask]
        denom = max(maximum(abs, new_miss), scale_obs, T(1e-10))
        maximum(abs.(new_miss .- prev_miss)) / denom < em_tol && break
    end
    X_cur
end

"""Evaluate lag-window spectrum with truncation `M_bw` on a frequency grid of `max(M_bw, min_grid) + 1` points to prevent DFT lag aliasing when `M_bw < p_max`."""
function _fhlz_lag_window_spectrum(X::AbstractMatrix{T}, M_bw::Int, kernel::Symbol,
    min_grid::Int) where {T<:AbstractFloat}
    M_bw >= min_grid && return _estimate_spectral_density_lagwindow(X, M_bw, kernel)
    T_obs, N = size(X)
    M_trunc = max(M_bw, 0)
    M_grid = max(M_trunc, min_grid, 1)
    n_freq = M_grid + 1
    frequencies = [T(π * h / M_grid) for h in 0:M_grid]

    Xc = X .- mean(X; dims=1)
    Gamma = Vector{Matrix{T}}(undef, M_trunc + 1)
    @inbounds for k in 0:M_trunc
        if k == 0
            Gamma[1] = (Xc' * Xc) / T_obs
        else
            Gamma[k + 1] = (Xc[1:(T_obs - k), :]' * Xc[(k + 1):T_obs, :]) / T_obs
        end
    end

    spectral = Array{Complex{T},3}(undef, N, N, n_freq)
    twoπ = T(2π)
    @inbounds for h in 1:n_freq
        θ = frequencies[h]
        S = zeros(Complex{T}, N, N)
        S .+= Gamma[1]
        for k in 1:M_trunc
            u = k / max(M_trunc, 1)
            w = kernel == :bartlett ? (1 - u) :
                kernel == :parzen ? (u <= 0.5 ? 1 - 6u^2 + 6u^3 : 2(1 - u)^3) :
                0.5 * (1 + cos(π * u))
            e = cis(-k * θ)
            Gk = Gamma[k + 1]
            S .+= w * (Gk * e + Gk' * conj(e))
        end
        S ./= twoπ
        spectral[:, :, h] = (S + S') / 2
    end
    frequencies, spectral
end

"""Lag-`k` autocovariance `Γ(k) = E[X_t X_{t-k}']` from non-negative lag list `Γ_pos`."""
@inline function _fhlz_gamma_at(Γ_pos::Vector{Matrix{T}}, k::Int) where {T<:AbstractFloat}
    k >= 0 ? Γ_pos[k + 1] : Matrix{T}(transpose(Γ_pos[-k + 1]))
end

"""Partition permutation `perm` into `max(1, div(N, q + 1))` blocks of size `q + 1` (remainder absorbed into last block)."""
function _fhlz_make_blocks(perm::Vector{Int}, q::Int)
    N = length(perm)
    bsize = q + 1
    n_blocks = max(1, div(N, bsize))
    blocks = Vector{Vector{Int}}(undef, n_blocks)
    @inbounds for b in 1:n_blocks
        lo = (b - 1) * bsize + 1
        hi = b < n_blocks ? b * bsize : N
        blocks[b] = perm[lo:hi]
    end
    blocks
end

"""Symmetric square root and inverse square root of `Σ_U = (U'U) / T_eff`: returns `(Σ_U^{1/2}, Σ_U^{-1/2})`."""
function _fhlz_symmetric_whitening(U::AbstractMatrix{T}) where {T<:AbstractFloat}
    T_eff, q = size(U)
    Σ_raw = (U' * U) / T(T_eff)
    Σ_sym = Symmetric((Σ_raw + Σ_raw') / T(2))
    E = eigen(Σ_sym)
    scale_floor = max(tr(Σ_sym) / T(q), one(T))
    floor_val = max(T(1e-12), T(10) * eps(T)) * scale_floor
    vals = max.(E.values, floor_val)
    vecs = E.vectors
    sqrt_vals = sqrt.(vals)
    inv_sqrt_vals = one(T) ./ sqrt_vals
    L_sqrt = vecs * Diagonal(sqrt_vals) * vecs'
    L_inv = vecs * Diagonal(inv_sqrt_vals) * vecs'
    Matrix{T}((L_sqrt + L_sqrt') / T(2)), Matrix{T}((L_inv + L_inv') / T(2))
end

"""Fit block VAR via Yule–Walker with ridge regularization, select lag order via BIC/AIC on top-`q` innovation eigenvalues, and enforce causal stability."""
function _fhlz_fit_block_var(
    Γ_pos::Vector{Matrix{T}},
    idx_b::Vector{Int},
    q::Int,
    p_max_use::Int,
    T_obs::Int,
    lag_criterion::Symbol,
) where {T<:AbstractFloat}
    n_b = length(idx_b)
    Γ_b = Vector{Matrix{T}}(undef, p_max_use + 1)
    G0 = Γ_pos[1][idx_b, idx_b]
    Γ_b[1] = Matrix{T}((G0 + G0') / T(2))
    @inbounds for k in 1:p_max_use
        Γ_b[k + 1] = Matrix{T}(Γ_pos[k + 1][idx_b, idx_b])
    end

    tr_scale = max(tr(Γ_b[1]) / T(n_b), T(1e-10))
    # Regularize the (n_b - q) rank-deficient subspace per excess lag so Bartlett-taper
    # estimation error does not invert near-zero Toeplitz eigenvalues at higher p_c.
    reg = max(T(8e-3), T(100) * eps(T)) * tr_scale
    eig_floor = max(T(1e-12), T(10) * eps(T)) * tr_scale
    pen = lag_criterion === :bic ? log(T(T_obs)) : T(2)
    k_top = min(q, n_b)

    best_score = T(Inf)
    best_p = 1
    best_Phi = Matrix{T}[]

    for p_c in 1:p_max_use
        m_dim = p_c * n_b
        T_p = Matrix{T}(undef, m_dim, m_dim)
        R_rhs = Matrix{T}(undef, n_b, m_dim)
        @inbounds for j in 1:p_c
            cj = ((j - 1) * n_b + 1):(j * n_b)
            R_rhs[:, cj] .= Γ_b[j + 1]
            for i in 1:p_c
                ri = ((i - 1) * n_b + 1):(i * n_b)
                T_p[ri, cj] .= _fhlz_gamma_at(Γ_b, j - i)
            end
        end
        T_p_sym = Matrix{T}((T_p + T_p') / T(2))
        T_reg = Symmetric(T_p_sym + reg * I)

        A_stack = try
            cholesky(T_reg) \ R_rhs'
        catch
            robust_inv(T_p_sym + reg * I; silent=true) * R_rhs'
        end
        Phi_all = Matrix{T}(A_stack')

        Σ_r_raw = Γ_b[1] - Phi_all * R_rhs' - R_rhs * Phi_all' + Phi_all * T_p_sym * Phi_all'
        Σ_r = Symmetric((Σ_r_raw + Σ_r_raw') / T(2))
        evals_r = sort(real.(eigvals(Σ_r)); rev=true)
        λ_top = max.(evals_r[1:k_top], eig_floor)
        score = sum(log, λ_top) + T(p_c * n_b * n_b) * pen / T(T_obs)

        if score < best_score || p_c == 1
            best_score = score
            best_p = p_c
            best_Phi = [Matrix{T}(Phi_all[:, ((k - 1) * n_b + 1):(k * n_b)]) for k in 1:p_c]
        end
    end

    # Enforce causal stability on the selected block VAR companion matrix
    comp_dim = best_p * n_b
    F_b = zeros(T, comp_dim, comp_dim)
    @inbounds for k in 1:best_p
        F_b[1:n_b, ((k - 1) * n_b + 1):(k * n_b)] .= best_Phi[k]
    end
    if best_p > 1
        @inbounds for i in 1:(best_p - 1)
            row_rng = (i * n_b + 1):((i + 1) * n_b)
            col_rng = ((i - 1) * n_b + 1):(i * n_b)
            F_b[row_rng, col_rng] .= Matrix{T}(I, n_b, n_b)
        end
    end

    max_mod = maximum(abs, eigvals(F_b))
    if !isfinite(max_mod)
        for k in 1:best_p
            fill!(best_Phi[k], zero(T))
        end
    elseif max_mod >= T(0.995)
        damp = T(0.99) / max_mod
        @inbounds for k in 1:best_p
            best_Phi[k] .*= damp^k
        end
    end

    best_p, best_Phi
end

"""Left-multiply `M` by lag-`k` block-diagonal filter `A_k^{(m)}` (`A_0 = I`, `A_k = -Φ_k` for `k ≥ 1`)."""
function _fhlz_left_apply_A(
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    p_blocks_m::Vector{Int},
    k::Int,
    M::AbstractMatrix{T},
) where {T<:AbstractFloat}
    k == 0 && return Matrix{T}(M)
    N = size(M, 1)
    out = zeros(T, N, size(M, 2))
    @inbounds for b in eachindex(blocks)
        k <= p_blocks_m[b] || continue
        idx_b = blocks[b]
        mul!(view(out, idx_b, :), Phi_blocks[b][k], view(M, idx_b, :), -one(T), zero(T))
    end
    out
end

"""Population common covariance `Γ_{Z,χ}^{(m)}(0) = Σ_{k=0}^{p_max} Σ_{l=0}^{p_max} A_k^{(m)} Γ_χ(l - k) A_l^{(m)'}`."""
function _fhlz_common_filtered_cov(
    Γ_pos::Vector{Matrix{T}},
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    p_blocks_m::Vector{Int},
    p_max_use::Int,
    N::Int,
) where {T<:AbstractFloat}
    p_eff = min(p_max_use, maximum(p_blocks_m))
    Γ_Zχ = zeros(T, N, N)
    @inbounds for k in 0:p_eff
        # k == l term: A_k Γ_χ(0) A_k'
        right_k = Matrix{T}(transpose(_fhlz_left_apply_A(blocks, Phi_blocks, p_blocks_m, k, Γ_pos[1])))
        term_kk = _fhlz_left_apply_A(blocks, Phi_blocks, p_blocks_m, k, right_k)
        Γ_Zχ .+= term_kk
        for l in (k + 1):p_eff
            # Off-diagonal (k, l): A_k Γ_χ(l - k) A_l' + (A_k Γ_χ(l - k) A_l')'
            # Recall (A_l Γ_χ(k - l))' = Γ_χ(k - l)' A_l' = Γ_χ(l - k) A_l'
            Γ_kl = _fhlz_gamma_at(Γ_pos, k - l)
            right_l = Matrix{T}(transpose(_fhlz_left_apply_A(blocks, Phi_blocks, p_blocks_m, l, Γ_kl)))
            term_kl = _fhlz_left_apply_A(blocks, Phi_blocks, p_blocks_m, k, right_l)
            Γ_Zχ .+= term_kl .+ transpose(term_kl)
        end
    end
    Matrix{T}((Γ_Zχ + Γ_Zχ') / T(2))
end

"""Compute causal blockwise VAR filtered residuals `Z` on `(p_max + 1):T_obs`."""
function _fhlz_filtered_residuals(
    X_proc::AbstractMatrix{T},
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    p_max_use::Int,
) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    T_obs > p_max_use || throw(ArgumentError(
        "X_proc must have > p_max=$p_max_use rows, got $T_obs"))
    T_eff = T_obs - p_max_use
    t_idx = (p_max_use + 1):T_obs
    Z = Matrix{T}(undef, T_eff, N)
    @inbounds for b in eachindex(blocks)
        idx_b = blocks[b]
        p_b = length(Phi_blocks[b])
        Zb = copy(@view X_proc[t_idx, idx_b])
        for k in 1:p_b
            mul!(Zb, @view(X_proc[t_idx .- k, idx_b]), transpose(Phi_blocks[b][k]), -one(T), one(T))
        end
        Z[:, idx_b] .= Zb
    end
    Z
end

"""Reconstruct strictly causal one-sided common component `χ̂_{b,t} = R_b u_t + Σ_{k=1}^{min(t-1,p_b)} Φ_k^{(b)} χ̂_{b,t-k}`."""
function _fhlz_reconstruct_causal(
    U_m::AbstractMatrix{T},
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    R_m::AbstractMatrix{T},
) where {T<:AbstractFloat}
    T_eff = size(U_m, 1)
    N = size(R_m, 1)
    chi_hat = Matrix{T}(undef, T_eff, N)
    @inbounds for b in eachindex(blocks)
        idx_b = blocks[b]
        n_b = length(idx_b)
        p_b = length(Phi_blocks[b])
        R_b = @view R_m[idx_b, :]
        chi_b = U_m * transpose(R_b)
        buf = Vector{T}(undef, n_b)
        for t in 1:T_eff
            for k in 1:min(t - 1, p_b)
                mul!(buf, Phi_blocks[b][k], @view(chi_b[t - k, :]))
                for i in 1:n_b
                    chi_b[t, i] += buf[i]
                end
            end
        end
        chi_hat[:, idx_b] .= chi_b
    end
    chi_hat
end

"""
    _fhlz_apply_causal_filter(X_proc, blocks, Phi_blocks, V_q, W_whiten, R_m, p_max) -> (chi_hat, U_m)

Pure causal application of fixed FHLZ blockwise filter parameters `(Phi_blocks, V_q, W_whiten, R_m)`
to `X_proc` without re-estimating any sample-wide statistics. Used to preserve strict one-sidedness.
"""
function _fhlz_apply_causal_filter(
    X_proc::AbstractMatrix{T},
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    V_q::AbstractMatrix{T},
    W_whiten::AbstractMatrix{T},
    R_m::AbstractMatrix{T},
    p_max_use::Int,
) where {T<:AbstractFloat}
    Z = _fhlz_filtered_residuals(X_proc, blocks, Phi_blocks, p_max_use)
    U_m = (Z * V_q) * W_whiten
    chi_hat = _fhlz_reconstruct_causal(U_m, blocks, Phi_blocks, R_m)
    chi_hat, U_m
end

"""Recursively invert `A^{(b)}(L) B^{(b)}(L) = R_b^{(m)}` for `h = 0..(irf_horizon - 1)`."""
function _fhlz_invert_ma(
    blocks::Vector{Vector{Int}},
    Phi_blocks::Vector{Vector{Matrix{T}}},
    R_m::AbstractMatrix{T},
    irf_horizon::Int,
    N::Int,
    q::Int,
) where {T<:AbstractFloat}
    B_m = zeros(T, irf_horizon, N, q)
    @inbounds for b in eachindex(blocks)
        idx_b = blocks[b]
        n_b = length(idx_b)
        p_b = length(Phi_blocks[b])
        B_m[1, idx_b, :] .= @view(R_m[idx_b, :])
        acc = Matrix{T}(undef, n_b, q)
        for h in 2:irf_horizon
            fill!(acc, zero(T))
            for k in 1:min(h - 1, p_b)
                mul!(acc, Phi_blocks[b][k], @view(B_m[h - k, idx_b, :]), one(T), one(T))
            end
            B_m[h, idx_b, :] .= acc
        end
    end
    B_m
end

"""Execute one permutation of the FHLZ blockwise VAR estimator (returns fixed filter parameters alongside outputs)."""
function _fhlz_fit_permutation(
    X_proc::AbstractMatrix{T},
    Γ_chi_pos::Vector{Matrix{T}},
    q::Int,
    p_max_use::Int,
    lag_criterion::Symbol,
    irf_horizon::Int,
    seed_m::UInt64,
) where {T<:AbstractFloat}
    T_obs, N = size(X_proc)
    perm = randperm(Random.Xoshiro(seed_m), N)
    blocks = _fhlz_make_blocks(perm, q)
    n_blocks = length(blocks)

    p_blocks_m = Vector{Int}(undef, n_blocks)
    Phi_blocks = Vector{Vector{Matrix{T}}}(undef, n_blocks)
    for b in 1:n_blocks
        p_b, Phi_b = _fhlz_fit_block_var(Γ_chi_pos, blocks[b], q, p_max_use, T_obs, lag_criterion)
        p_blocks_m[b] = p_b
        Phi_blocks[b] = Phi_b
    end

    Γ_Zχ = _fhlz_common_filtered_cov(Γ_chi_pos, blocks, Phi_blocks, p_blocks_m, p_max_use, N)
    Ez = eigen(Symmetric(Γ_Zχ))
    idx_z = sortperm(Ez.values; rev=true)
    V_q = Matrix{T}(Ez.vectors[:, idx_z[1:q]])
    @inbounds for j in 1:q
        k_max = argmax(abs.(view(V_q, :, j)))
        V_q[k_max, j] < zero(T) && (V_q[:, j] .*= -one(T))
    end

    Z = _fhlz_filtered_residuals(X_proc, blocks, Phi_blocks, p_max_use)
    U_raw = Z * V_q
    L_whiten_T, W_whiten = _fhlz_symmetric_whitening(U_raw)
    R_m = V_q * L_whiten_T

    U_m = U_raw * W_whiten
    chi_m = _fhlz_reconstruct_causal(U_m, blocks, Phi_blocks, R_m)
    B_m = _fhlz_invert_ma(blocks, Phi_blocks, R_m, irf_horizon, N, q)

    (
        p_blocks_m = p_blocks_m,
        chi_m = chi_m,
        U_m = U_m,
        B_m = B_m,
        blocks = blocks,
        Phi_blocks = Phi_blocks,
        V_q = V_q,
        W_whiten = W_whiten,
        R_m = R_m,
    )
end

function _fhlz_run_permutation(
    X_proc::AbstractMatrix{T},
    Γ_chi_pos::Vector{Matrix{T}},
    q::Int,
    p_max_use::Int,
    lag_criterion::Symbol,
    irf_horizon::Int,
    seed_m::UInt64,
) where {T<:AbstractFloat}
    res = _fhlz_fit_permutation(X_proc, Γ_chi_pos, q, p_max_use, lag_criterion, irf_horizon, seed_m)
    (p_blocks_m=res.p_blocks_m, chi_m=res.chi_m, U_m=res.U_m, B_m=res.B_m)
end

# =============================================================================
# StatsAPI Interface & Utilities
# =============================================================================

"""Predicted values (one-sided common component `T_eff × N`)."""
StatsAPI.predict(m::FHLZModel) = m.common_component

"""Residuals (idiosyncratic component `T_eff × N`)."""
StatsAPI.residuals(m::FHLZModel) = m.idiosyncratic

"""Number of raw observations (`T_obs`)."""
StatsAPI.nobs(m::FHLZModel) = size(m.X, 1)

"""Effective sample size (`T_eff = T_obs - p_max`)."""
effective_nobs(m::FHLZModel) = size(m.common_component, 1)

"""Approximate degrees of freedom (average block VAR parameters + innovation weights + shock realizations)."""
function StatsAPI.dof(m::FHLZModel)
    N = size(m.X, 2)
    T_eff = size(m.innovations, 1)
    n_blocks = size(m.p_blocks, 2)
    avg_nb = N / max(n_blocks, 1)
    var_params = round(Int, mean(sum(m.p_blocks; dims=2)) * avg_nb^2)
    var_params + N * m.q + T_eff * m.q
end

"""R² for each panel variable on the effective sample."""
function StatsAPI.r2(m::FHLZModel{T}) where {T}
    T_obs, N = size(m.X)
    T_eff = size(m.common_component, 1)
    offset = T_obs - T_eff
    X_eff = @view m.X[(offset + 1):end, :]
    [one(T) - var(@view m.idiosyncratic[:, i]) / max(var(@view X_eff[:, i]), T(1e-10)) for i in 1:N]
end

"""Fraction of each variable's variance explained by the one-sided common component."""
function common_variance_share(m::FHLZModel{T}) where {T}
    T_obs, N = size(m.X)
    T_eff = size(m.common_component, 1)
    offset = T_obs - T_eff
    X_eff = @view m.X[(offset + 1):end, :]
    [var(@view m.common_component[:, i]) / max(var(@view X_eff[:, i]), T(1e-10)) for i in 1:N]
end

function Base.show(io::IO, m::FHLZModel{T}) where {T}
    T_obs, N = size(m.X)
    T_eff = size(m.common_component, 1)
    p_max_used = T_obs - T_eff
    mean_p = round(mean(m.p_blocks); digits=2)
    max_p = maximum(m.p_blocks)
    spec = Any[
        "Dynamic factors (q)"  m.q;
        "Variables (N)"        N;
        "Observations (T)"     T_obs;
        "Effective obs"        T_eff;
        "Permutations"         m.n_perm;
        "Block lag (mean/max)" "$mean_p / $max_p (cap $p_max_used)";
        "Kernel"               string(m.kernel);
        "Bandwidth"            m.bandwidth;
        "IRF horizon"          size(m.impulse_filters, 1);
        "Standardized"         m.standardized ? "Yes" : "No"
    ]
    _pretty_table(io, spec;
        title = "FHLZ One-Sided Infinite-r GDFM (q=$(m.q), n_perm=$(m.n_perm))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    n_show = min(m.q, 5)
    cum_var = cumsum(m.variance_explained)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(m.variance_explained[i])
        var_data[i, 3] = _fmt_pct(cum_var[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )
end
