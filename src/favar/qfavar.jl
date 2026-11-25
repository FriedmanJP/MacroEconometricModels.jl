# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Quantile Factor-Augmented VAR (Quantile FAVAR) with Two-Step IQR and Variational Bayes
estimation following Korobilis & Schröder (2025).

Supports:
- Two-step estimation (`method=:twostep`): quantile factor extraction via `estimate_qfm`,
  BBE (#528) purging of `Y_key` from `F̂(τ)`, quantile measurement regression with exact
  unit-impact identity on `Y_key_indices`, and recursive structural QVAR via `estimate_qvar`.
- Variational Bayes estimation (`method=:vb`): Asymmetric Laplace normal-exponential mixture
  CAVI updates over factors, loadings `(Λ(τ), Λ_y(τ))` honoring `#528` BBE anchor and
  key-variable constraints, and structural VAR coefficients with Minnesota-Horseshoe shrinkage.
- Panel-wide Quantile IRFs (`qfavar_panel_irf`) and non-crossing quantile forecasts
  (`qfavar_panel_forecast`).

References:
- Korobilis, D., & Schröder, M. (2025). Monitoring macroeconomic tail risk with quantile
  factor-augmented VARs. *Journal of Econometrics*, forthcoming.
- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects of monetary policy:
  A Factor-Augmented Vector Autoregressive (FAVAR) approach. *Quarterly Journal of Economics*,
  120(1), 387-422.
- Chavleishvili, S., & Manganelli, S. (2024). Forecasting and stress testing with quantile
  vector autoregression. *Journal of Applied Econometrics*, 39(1), 66-85.
"""

# =============================================================================
# Model Types
# =============================================================================

"""
    QuantileFAVAR{T<:AbstractFloat}

Two-step Quantile Factor-Augmented VAR model combining quantile factor extraction
(`estimate_qfm`), BBE (#528) purging of observed key variables `Y_key`, quantile
measurement regressions, and a recursive structural [`QuantileVARModel`](@ref) on
``[\\tilde{F}_t(\\tau), Y_{\\text{key}, t}]``:

```math
Q_\\tau(X_{it} \\mid \\tilde{F}_t(\\tau), Y_{\\text{key}, t}) =
c_i(\\tau) + \\lambda_i(\\tau)' \\tilde{F}_t(\\tau) + \\lambda_{y, i}(\\tau)' Y_{\\text{key}, t}
```

where:
- ``X_{it}`` is panel series ``i`` at time ``t``
- ``\\tilde{F}_t(\\tau)`` is the ``r \\times 1`` vector of slow-moving quantile factors purged of ``Y_{\\text{key}, t}``
- ``Y_{\\text{key}, t}`` is the ``n_{\\text{key}} \\times 1`` vector of observed key variables
- ``\\lambda_i(\\tau)`` and ``\\lambda_{y, i}(\\tau)`` are the factor and direct key-variable quantile loadings

# Fields
- `qvar::QuantileVARModel{T}` — estimated QVAR on ``[\\tilde{F}_t(\\tau), Y_{\\text{key}, t}]``
- `factors_list::Vector{Matrix{T}}` — per-quantile purged factor matrices (``T_{\\text{obs}} \\times r``)
- `loadings_list::Vector{Matrix{T}}` — per-quantile factor loading matrices (``N \\times r``)
- `lambda_y_list::Vector{Matrix{T}}` — per-quantile direct key-variable loading matrices (``N \\times n_{\\text{key}}``)
- `intercept_list::Vector{Vector{T}}` — per-quantile measurement intercepts (length ``N``)
- `qfm_models::Vector{QuantileFactorModel{T}}` — underlying quantile factor models per ``\\tau``
- `taus::Vector{T}` — quantile levels strictly in ``(0, 1)``
- `X_panel::Matrix{T}` — original panel matrix (``T_{\\text{obs}} \\times N``)
- `Y_key::Matrix{T}` — observed key variables (``T_{\\text{obs}} \\times n_{\\text{key}}``)
- `Y_key_indices::Vector{Int}` — column indices of key variables in `X_panel` (empty if external)
- `n_factors::Int` — number of latent quantile factors ``r``
- `n_key::Int` — number of key observed variables ``n_{\\text{key}}``
- `p::Int` — QVAR lag order
- `standardize::Bool` — whether panel series were standardized during factor/measurement estimation
- `panel_varnames::Vector{String}` — panel variable names (length ``N``)
- `varnames::Vector{String}` — augmented VAR variable names (length ``r + n_{\\text{key}}``)
- `manifest::Union{ReproManifest,Nothing}` — reproducibility metadata
"""
struct QuantileFAVAR{T<:AbstractFloat}
    qvar::QuantileVARModel{T}
    factors_list::Vector{Matrix{T}}
    loadings_list::Vector{Matrix{T}}
    lambda_y_list::Vector{Matrix{T}}
    intercept_list::Vector{Vector{T}}
    qfm_models::Vector{QuantileFactorModel{T}}
    taus::Vector{T}
    X_panel::Matrix{T}
    Y_key::Matrix{T}
    Y_key_indices::Vector{Int}
    n_factors::Int
    n_key::Int
    p::Int
    standardize::Bool
    panel_varnames::Vector{String}
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

QuantileFAVAR{T}(qvar, factors_list, loadings_list, lambda_y_list, intercept_list,
                 qfm_models, taus, X_panel, Y_key, Y_key_indices, n_factors, n_key,
                 p, standardize, panel_varnames, varnames;
                 manifest=nothing) where {T<:AbstractFloat} =
    QuantileFAVAR{T}(qvar, factors_list, loadings_list, lambda_y_list, intercept_list,
                     qfm_models, taus, X_panel, Y_key, Y_key_indices, n_factors, n_key,
                     p, standardize, panel_varnames, varnames, manifest)

"""
    VBQuantileFAVAR{T<:AbstractFloat}

Variational Bayes Quantile Factor-Augmented VAR model (Korobilis & Schröder 2025)
using the Asymmetric Laplace normal-exponential mixture representation:

```math
e_{it}(\\tau) = \\theta_\\tau z_{it} + \\sqrt{\\kappa_\\tau^2 \\sigma_i z_{it}} \\, \\xi_{it},
\\qquad
\\theta_\\tau = \\frac{1 - 2\\tau}{\\tau(1 - \\tau)},
\\qquad
\\kappa_\\tau^2 = \\frac{2}{\\tau(1 - \\tau)}
```

where:
- ``e_{it}(\\tau)`` is the Asymmetric Laplace disturbance at quantile ``\\tau \\in (0, 1)``
- ``z_{it} \\sim \\mathrm{Exp}(\\sigma_i^{-1})`` is the latent exponential mixing variable
- ``\\sigma_i`` is the equation scale parameter
- ``\\xi_{it} \\sim \\mathcal{N}(0, 1)`` is a standard normal innovation

# Fields
- `qvar::QuantileVARModel{T}` — posterior-mean recursive structural QVAR across quantiles
- `factors_mean::Vector{Matrix{T}}` — per-quantile variational posterior mean factors (``T_{\\text{obs}} \\times r``)
- `loadings_mean::Vector{Matrix{T}}` — per-quantile posterior mean factor loadings (``N \\times r``)
- `lambda_y_mean::Vector{Matrix{T}}` — per-quantile posterior mean direct ``Y_{\\text{key}}`` loadings (``N \\times n_{\\text{key}}``)
- `intercept_mean::Vector{Vector{T}}` — per-quantile posterior mean measurement intercepts (length ``N``)
- `factor_cov::Vector{Array{T,3}}` — per-quantile variational factor covariance tensors (``T_{\\text{obs}} \\times r \\times r``)
- `var_cov::Vector{Vector{Matrix{T}}}` — per-quantile, per-equation variational VAR coefficient covariance matrices
- `A0_draws::Vector{Array{T,3}}` — per-quantile posterior draws of ``A_0(\\tau)`` (``n_{\\text{draws}} \\times n_{\\text{aug}} \\times n_{\\text{aug}}``)
- `B_draws::Vector{Array{T,3}}` — per-quantile posterior draws of ``B(\\tau)`` (``n_{\\text{draws}} \\times (1 + n_{\\text{aug}} p) \\times n_{\\text{aug}}``)
- `loadings_draws::Vector{Array{T,3}}` — per-quantile posterior draws of ``\\Lambda(\\tau)`` (``n_{\\text{draws}} \\times N \\times r``)
- `lambda_y_draws::Vector{Array{T,3}}` — per-quantile posterior draws of ``\\Lambda_y(\\tau)`` (``n_{\\text{draws}} \\times N \\times n_{\\text{key}}``)
- `taus::Vector{T}` — quantile levels strictly in ``(0, 1)``
- `X_panel::Matrix{T}` — original panel matrix (``T_{\\text{obs}} \\times N``)
- `Y_key::Matrix{T}` — observed key variables (``T_{\\text{obs}} \\times n_{\\text{key}}``)
- `Y_key_indices::Vector{Int}` — column indices of key variables in `X_panel`
- `n_factors::Int` — number of latent factors ``r``
- `n_key::Int` — number of key observed variables ``n_{\\text{key}}``
- `p::Int` — QVAR lag order
- `standardize::Bool` — whether panel series were standardized during estimation
- `vb_converged::Vector{Bool}` — CAVI convergence status per quantile
- `vb_iterations::Vector{Int}` — CAVI iteration count per quantile
- `panel_varnames::Vector{String}` — panel variable names (length ``N``)
- `varnames::Vector{String}` — augmented VAR variable names (length ``r + n_{\\text{key}}``)
- `manifest::Union{ReproManifest,Nothing}` — reproducibility metadata
"""
struct VBQuantileFAVAR{T<:AbstractFloat}
    qvar::QuantileVARModel{T}
    factors_mean::Vector{Matrix{T}}
    loadings_mean::Vector{Matrix{T}}
    lambda_y_mean::Vector{Matrix{T}}
    intercept_mean::Vector{Vector{T}}
    factor_cov::Vector{Array{T,3}}
    var_cov::Vector{Vector{Matrix{T}}}
    A0_draws::Vector{Array{T,3}}
    B_draws::Vector{Array{T,3}}
    loadings_draws::Vector{Array{T,3}}
    lambda_y_draws::Vector{Array{T,3}}
    taus::Vector{T}
    X_panel::Matrix{T}
    Y_key::Matrix{T}
    Y_key_indices::Vector{Int}
    n_factors::Int
    n_key::Int
    p::Int
    standardize::Bool
    vb_converged::Vector{Bool}
    vb_iterations::Vector{Int}
    panel_varnames::Vector{String}
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

VBQuantileFAVAR{T}(qvar, factors_mean, loadings_mean, lambda_y_mean, intercept_mean,
                   factor_cov, var_cov, A0_draws, B_draws, loadings_draws, lambda_y_draws,
                   taus, X_panel, Y_key, Y_key_indices, n_factors, n_key, p, standardize,
                   vb_converged, vb_iterations, panel_varnames, varnames;
                   manifest=nothing) where {T<:AbstractFloat} =
    VBQuantileFAVAR{T}(qvar, factors_mean, loadings_mean, lambda_y_mean, intercept_mean,
                       factor_cov, var_cov, A0_draws, B_draws, loadings_draws, lambda_y_draws,
                       taus, X_panel, Y_key, Y_key_indices, n_factors, n_key, p, standardize,
                       vb_converged, vb_iterations, panel_varnames, varnames, manifest)

# =============================================================================
# Accessors
# =============================================================================

nvars(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = m.n_factors + m.n_key
nlags(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = m.p
ncoefs(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = 1 + nvars(m) * m.p
effective_nobs(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = size(m.X_panel, 1) - m.p
varnames(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = m.varnames
StatsAPI.nobs(m::Union{QuantileFAVAR,VBQuantileFAVAR}) = size(m.X_panel, 1)

# =============================================================================
# Internal Helpers: Scaling, Alignment & VB CAVI Kernels
# =============================================================================

"""Standardize panel columns returning `(X_std, mu_X, s_X)`."""
function _qfavar_prep_panel(X::Matrix{T}, standardize::Bool) where {T<:AbstractFloat}
    T_obs, N = size(X)
    if !standardize
        return copy(X), zeros(T, N), ones(T, N)
    end
    mu_X = vec(Statistics.mean(X; dims=1))
    s_X = max.(vec(Statistics.std(X; dims=1)), max(T(1e-10), eps(T)))
    X_std = Matrix{T}(undef, T_obs, N)
    @inbounds for j in 1:N, t in 1:T_obs
        X_std[t, j] = (X[t, j] - mu_X[j]) / s_X[j]
    end
    return X_std, mu_X, s_X
end

"""Orthogonal Procrustes alignment of `F_src` (`T × r`) to `F_ref` (`T × r`)."""
function _qfavar_align_factors(F_src::Matrix{T}, F_ref::Matrix{T}) where {T<:AbstractFloat}
    r = size(F_src, 2)
    r == 0 && return copy(F_src)
    sv = svd(F_src' * F_ref)
    Q = sv.U * sv.V'
    return F_src * Q
end

"""Find BBE (2005) anchor block `G = L[anchor, :]` with QR column-pivot fallback; returns `(anchor, G, cond_G, has_anchors)`."""
function _qfavar_find_anchor_block(L::Matrix{T}, Y_key_indices::Vector{Int},
                                    r::Int, N::Int) where {T<:AbstractFloat}
    nonkey = setdiff(1:N, Y_key_indices)
    length(nonkey) >= r || return Int[], Matrix{T}(undef, 0, r), T(Inf), false
    anchor = nonkey[1:r]
    G = Matrix{T}(L[anchor, :])
    cG = cond(G)
    if !isfinite(cG) || cG >= T(1e6)
        F_qr = qr(Matrix{T}(L[nonkey, :]'), ColumnNorm())
        anchor_qr = nonkey[F_qr.p[1:r]]
        G_qr = Matrix{T}(L[anchor_qr, :])
        if isfinite(cond(G_qr)) && cond(G_qr) < cG
            anchor = anchor_qr
            G = G_qr
            cG = cond(G)
        end
    end
    return anchor, G, cG, true
end

"""Orient reference factor columns consistently with BBE non-key panel anchors (positive loading signs)."""
function _qfavar_orient_factors(F_ref::Matrix{T}, L_ref::Matrix{T},
                                Y_key_indices::Vector{Int}, r::Int, N::Int) where {T<:AbstractFloat}
    r == 0 && return copy(F_ref)
    _, G, cG, has_anchors = _qfavar_find_anchor_block(L_ref, Y_key_indices, r, N)
    if has_anchors && isfinite(cG) && cG < T(1e10)
        sv = svd(G)
        return F_ref * (sv.V * sv.U')
    end
    F_out = copy(F_ref)
    for j in 1:r
        if sum(@view(L_ref[:, j])) < zero(T)
            @views F_out[:, j] .= .-F_out[:, j]
        end
    end
    return F_out
end

"""Two-step Quantile FAVAR estimation."""
function _estimate_qfavar_twostep(X_mat::Matrix{T}, Y_key_mat::Matrix{T},
                                  Y_key_indices::Vector{Int}, r::Int, p::Int,
                                  taus_T::Vector{T}, standardize::Bool,
                                  pvn::Vector{String}, aug_varnames::Vector{String},
                                  se::Symbol, n_boot::Int,
                                  rng_use::AbstractRNG,
                                  seed::Union{Integer,Nothing}) where {T<:AbstractFloat}
    T_obs, N = size(X_mat)
    n_key = size(Y_key_mat, 2)
    n_aug = r + n_key
    M = length(taus_T)
    ord = collect(1:n_aug)

    X_std, mu_X, s_X = _qfavar_prep_panel(X_mat, standardize)

    qfm_models = Vector{QuantileFactorModel{T}}(undef, M)
    slot_seeds = rand(rng_use, UInt64, M)
    for m in 1:M
        slot_rng = Random.Xoshiro(slot_seeds[m])
        qfm_models[m] = estimate_qfm(X_mat, r, taus_T[m];
                                     standardize=standardize, robust_scale=false,
                                     se=:none, threaded=false, rng=slot_rng, seed=nothing)
    end

    # Align quantile factors across taus to BBE-oriented reference factors at τ ≈ 0.50 via Procrustes
    idx_ref = argmin(abs.(taus_T .- T(0.5)))
    F_ref = _qfavar_orient_factors(qfm_models[idx_ref].factors, qfm_models[idx_ref].loadings,
                                   Y_key_indices, r, N)

    factors_list = Vector{Matrix{T}}(undef, M)
    loadings_list = Vector{Matrix{T}}(undef, M)
    lambda_y_list = Vector{Matrix{T}}(undef, M)
    intercept_list = Vector{Vector{T}}(undef, M)

    A0_list = Vector{Matrix{T}}(undef, M)
    B_list = Vector{Matrix{T}}(undef, M)
    resid_list = Vector{Matrix{T}}(undef, M)
    se_list = Vector{Matrix{T}}(undef, M)

    for m in 1:M
        tau_m = taus_T[m]
        F_raw_m = m == idx_ref ? copy(F_ref) : _qfavar_align_factors(qfm_models[m].factors, F_ref)

        # #528 purging: remove Y_key span from quantile factors
        F_tilde_m, _ = _remove_double_counting(F_raw_m, Y_key_mat)
        factors_list[m] = F_tilde_m

        # Quantile measurement equation: Q_tau(X_std) = c + F_tilde * Λ' + Y_key * Λ_y'
        W_m = hcat(ones(T, T_obs), F_tilde_m, Y_key_mat)
        Theta_m = Matrix{T}(undef, 1 + r + n_key, N)
        _qreg_batch!(Theta_m, W_m, X_std, tau_m; threaded=false)

        c_m = Vector{T}(Theta_m[1, :])
        L_m = Matrix{T}(Theta_m[2:(1 + r), :]')
        Ly_m = Matrix{T}(Theta_m[(2 + r):end, :]')

        # Enforce exact #528 unit-impact identity on Y_key_indices columns
        for (k, panel_idx) in enumerate(Y_key_indices)
            (1 <= panel_idx <= N) || continue
            L_m[panel_idx, :] .= zero(T)
            Ly_m[panel_idx, :] .= zero(T)
            Ly_m[panel_idx, k] = one(T) / s_X[panel_idx]
            c_m[panel_idx] = -mu_X[panel_idx] / s_X[panel_idx]
        end

        intercept_list[m] = c_m
        loadings_list[m] = L_m
        lambda_y_list[m] = Ly_m

        # Fit quantile VAR on [F_tilde(τ_m), Y_key]
        Y_aug_m = hcat(F_tilde_m, Y_key_mat)
        Y_eff_m, X_lag_m = construct_var_matrices(Y_aug_m, p)
        slot_rng = Random.Xoshiro(rand(rng_use, UInt64))
        A0_m, B_m, R_m, se_m = _qvar_fit_single(Y_eff_m, X_lag_m, n_aug, p, tau_m,
                                                ord, se, n_boot, slot_rng)
        A0_list[m] = A0_m
        B_list[m] = B_m
        resid_list[m] = R_m
        se_list[m] = se_m
    end

    Y_aug_ref = hcat(factors_list[idx_ref], Y_key_mat)
    mani = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "method" => "twostep", "r" => r, "p" => p, "taus" => Float64.(taus_T),
        "standardize" => standardize, "se" => String(se)))
    qvar = QuantileVARModel{T}(A0_list, B_list, resid_list, se_list,
                               taus_T, ord, Y_aug_ref, p, aug_varnames, mani)

    return QuantileFAVAR{T}(
        qvar, factors_list, loadings_list, lambda_y_list, intercept_list,
        qfm_models, taus_T, X_mat, Y_key_mat, Y_key_indices,
        r, n_key, p, standardize, pvn, aug_varnames, mani
    )
end

"""Select BBE (2005) anchor columns from non-key columns of `X_panel`, throwing if fewer than `r` non-key or singular."""
function _qfavar_select_anchors(Lambda_init::Matrix{T}, Y_key_indices::Vector{Int},
                                r::Int, N::Int) where {T<:AbstractFloat}
    anchor, G, cG, has_anchors = _qfavar_find_anchor_block(Lambda_init, Y_key_indices, r, N)
    has_anchors || throw(ArgumentError(
        "VB Quantile FAVAR requires at least r=$r non-key panel columns to anchor factors, " *
        "got $(length(setdiff(1:N, Y_key_indices))) (N=$N, n_key_in_panel=$(length(intersect(1:N, Y_key_indices))))"))
    isfinite(cG) && cG < T(1e10) || throw(ArgumentError(
        "Panel anchor columns $anchor are numerically singular for $r factors (cond=$cG)."))
    return anchor, G
end

"""Run VB CAVI for a single quantile `tau` of the Quantile FAVAR model."""
function _qfavar_vb_single_tau(X_std::Matrix{T}, Y_key_mat::Matrix{T},
                                Y_key_indices::Vector{Int}, mu_X::Vector{T}, s_X::Vector{T},
                                F_init::Matrix{T}, Lambda_init::Matrix{T}, Lambda_y_init::Matrix{T},
                                anchor::Vector{Int}, r::Int, p::Int, tau::T,
                                vb_maxiter::Int, vb_tol::T, n_draws::Int,
                                rng::AbstractRNG) where {T<:AbstractFloat}
    T_obs, N = size(X_std)
    n_key = size(Y_key_mat, 2)
    n_aug = r + n_key
    ord = collect(1:n_aug)
    k_load = 1 + r + n_key

    # Asymmetric Laplace normal-exponential mixture constants (Korobilis & Schröder 2025)
    tt = tau * (one(T) - tau)
    theta_tau = (one(T) - T(2) * tau) / tt
    kappa2_tau = T(2) / tt
    inv_2tt = one(T) / (T(2) * tt)
    sum_mix = theta_tau^2 + T(2) * kappa2_tau

    # Scale-invariant prior precision on [intercept; loadings_F; loadings_Y]
    y_vars = [max(T(Statistics.var(@view(Y_key_mat[:, k]))), T(1e-4)) for k in 1:n_key]
    P_load_0 = Diagonal(vcat(T[T(1e-4)], fill(T(0.05), r), [T(0.05) / yv for yv in y_vars]))

    anchor_pos = zeros(Int, N)
    for (k, a) in enumerate(anchor)
        anchor_pos[a] = k
    end
    key_pos = zeros(Int, N)
    for (k, idx) in enumerate(Y_key_indices)
        (1 <= idx <= N) && (key_pos[idx] = k)
    end
    free_cols = [i for i in 1:N if key_pos[i] == 0]

    F_m = copy(F_init)
    Lambda_m = copy(Lambda_init)
    Lambda_y_m = copy(Lambda_y_init)
    c_m = zeros(T, N)

    # Set exact #528 constraints on key columns and initial intercepts
    for i in 1:N
        if key_pos[i] > 0
            k = key_pos[i]
            Lambda_m[i, :] .= zero(T)
            Lambda_y_m[i, :] .= zero(T)
            Lambda_y_m[i, k] = one(T) / s_X[i]
            c_m[i] = -mu_X[i] / s_X[i]
        else
            r0 = @view(X_std[:, i]) .- F_m * @view(Lambda_m[i, :]) .- Y_key_mat * @view(Lambda_y_m[i, :])
            c_m[i] = T(Statistics.quantile(r0, tau))
        end
    end

    # Variational covariances and latent scale expectations
    V_f = Array{T,3}(undef, T_obs, r, r)
    for t in 1:T_obs
        V_f[t, :, :] .= T(0.05) .* Matrix{T}(I, r, r)
    end
    V_load = [zeros(T, k_load, k_load) for _ in 1:N]
    for i in free_cols
        if anchor_pos[i] > 0
            V_load[i][1, 1] = T(0.05)
        else
            V_load[i] .= T(0.05) .* Matrix{T}(I, k_load, k_load)
        end
    end

    E_inv_sig_x = ones(T, N)
    for i in free_cols
        resid_0 = @view(X_std[:, i]) .- c_m[i] .- F_m * @view(Lambda_m[i, :]) .- Y_key_mat * @view(Lambda_y_m[i, :])
        E_inv_sig_x[i] = one(T) / max(T(Statistics.var(resid_0)), T(1e-3))
    end
    E_inv_z_x = ones(T, T_obs, N)
    E_z_x = ones(T, T_obs, N)

    # Pre-allocated scratch buffers for measurement CAVI
    W_t = Vector{T}(undef, k_load)
    P_f_buf = Matrix{T}(undef, r, r)
    rhs_f_buf = Vector{T}(undef, r)
    P_l_buf = Matrix{T}(undef, k_load, k_load)
    rhs_l_buf = Vector{T}(undef, k_load)
    EWtW = Matrix{T}(undef, k_load, k_load)
    lam_lam = [Matrix{T}(undef, r, r) for _ in 1:N]
    F_prev = similar(F_m)
    L_prev = similar(Lambda_m)
    delta_floor = max(T(1e-6), T(10) * eps(T))
    I_r = Matrix{T}(I, r, r)

    vb_conv = false
    vb_iters = 0

    for it in 1:vb_maxiter
        copyto!(F_prev, F_m)
        copyto!(L_prev, Lambda_m)
        vb_iters = it

        for i in free_cols
            @views lam_lam[i] .= Lambda_m[i, :] * Lambda_m[i, :]' .+ V_load[i][2:(1 + r), 2:(1 + r)]
        end

        # 1. Update q(z_{t, i}^x) and q(σ_i^{x, -1})
        for i in free_cols
            li = @view Lambda_m[i, :]
            lyi = @view Lambda_y_m[i, :]
            ci = c_m[i]
            Vli = V_load[i]
            lam_i_rate = E_inv_sig_x[i] * inv_2tt
            ll_i = lam_lam[i]
            b_sum = zero(T)

            for t in 1:T_obs
                ft = @view F_m[t, :]
                yt = @view Y_key_mat[t, :]
                r_ti = X_std[t, i] - ci - dot(li, ft) - dot(lyi, yt)

                W_t[1] = one(T)
                @views W_t[2:(1 + r)] .= ft
                @views W_t[(2 + r):k_load] .= yt

                var_load = zero(T)
                @inbounds for a in 1:k_load, b in 1:k_load
                    var_load += W_t[a] * Vli[a, b] * W_t[b]
                end
                tr_vf = zero(T)
                @inbounds for a in 1:r, b in 1:r
                    tr_vf += ll_i[a, b] * V_f[t, b, a]
                end

                Delta = max(r_ti * r_ti + var_load + tr_vf, delta_floor)
                e_iz = min(one(T) / (tt * sqrt(Delta)), T(1e6))
                e_z = one(T) / e_iz + one(T) / max(lam_i_rate, T(1e-8))
                E_inv_z_x[t, i] = e_iz
                E_z_x[t, i] = e_z

                b_sum += Delta * e_iz - T(2) * theta_tau * r_ti + sum_mix * e_z
            end
            a_post = one(T) + T(1.5) * T(T_obs)
            b_post = T(0.01) + b_sum / (T(2) * kappa2_tau)
            E_inv_sig_x[i] = clamp(a_post / max(b_post, T(1e-8)), T(1e-4), T(1e4))
        end

        # 2. Update q(f_t)
        for t in 1:T_obs
            copyto!(P_f_buf, I_r)
            fill!(rhs_f_buf, zero(T))
            sig_over_k = one(T) / kappa2_tau
            yt = @view Y_key_mat[t, :]
            for i in free_cols
                e_iz = E_inv_z_x[t, i]
                w_ti = E_inv_sig_x[i] * sig_over_k * e_iz
                P_f_buf .+= w_ti .* lam_lam[i]
                x_net = X_std[t, i] - c_m[i] - dot(@view(Lambda_y_m[i, :]), yt) - theta_tau / e_iz
                @inbounds for a in 1:r
                    rhs_f_buf[a] += w_ti * Lambda_m[i, a] * x_net
                end
            end
            Vf_t = Matrix{T}(robust_inv(Symmetric(T(0.5) .* (P_f_buf .+ P_f_buf')); silent=true))
            @views V_f[t, :, :] .= T(0.5) .* (Vf_t .+ Vf_t')
            @views F_m[t, :] .= Vf_t * rhs_f_buf
        end

        # 3. Update q(c_i, λ_i, λ_{y,i}) honoring #528 anchors and key columns
        for i in free_cols
            sig_k = E_inv_sig_x[i] / kappa2_tau
            ap = anchor_pos[i]
            if ap > 0
                prec_c = P_load_0[1, 1]
                rhs_c = zero(T)
                for t in 1:T_obs
                    e_iz = E_inv_z_x[t, i]
                    w_ti = sig_k * e_iz
                    prec_c += w_ti
                    rhs_c += w_ti * (X_std[t, i] - F_m[t, ap] - theta_tau / e_iz)
                end
                V_load[i] .= zero(T)
                V_load[i][1, 1] = one(T) / max(prec_c, T(1e-8))
                c_m[i] = V_load[i][1, 1] * rhs_c
                Lambda_m[i, :] .= zero(T)
                Lambda_m[i, ap] = one(T)
                Lambda_y_m[i, :] .= zero(T)
            else
                copyto!(P_l_buf, P_load_0)
                fill!(rhs_l_buf, zero(T))
                for t in 1:T_obs
                    e_iz = E_inv_z_x[t, i]
                    w_ti = sig_k * e_iz
                    W_t[1] = one(T)
                    @views W_t[2:(1 + r)] .= F_m[t, :]
                    @views W_t[(2 + r):k_load] .= Y_key_mat[t, :]
                    mul!(EWtW, W_t, transpose(W_t))
                    @views EWtW[2:(1 + r), 2:(1 + r)] .+= V_f[t, :, :]
                    P_l_buf .+= w_ti .* EWtW
                    y_adj = X_std[t, i] - theta_tau / e_iz
                    @inbounds for a in 1:k_load
                        rhs_l_buf[a] += w_ti * W_t[a] * y_adj
                    end
                end
                Vl_i = Matrix{T}(robust_inv(Symmetric(T(0.5) .* (P_l_buf .+ P_l_buf')); silent=true))
                V_load[i] .= T(0.5) .* (Vl_i .+ Vl_i')
                theta_l = V_load[i] * rhs_l_buf
                c_m[i] = theta_l[1]
                @views Lambda_m[i, :] .= theta_l[2:(1 + r)]
                @views Lambda_y_m[i, :] .= theta_l[(2 + r):k_load]
            end
        end

        diff_F = maximum(abs, F_m - F_prev)
        diff_L = maximum(abs, Lambda_m - L_prev)
        if max(diff_F, diff_L) < vb_tol
            vb_conv = true
            break
        end
    end

    # --- Block 2: VAR CAVI with Minnesota-Horseshoe shrinkage on [F_m, Y_key] ---
    Y_aug = hcat(F_m, Y_key_mat)
    Y_eff, X_lag = construct_var_matrices(Y_aug, p)
    T_eff = size(Y_eff, 1)
    k_lag = 1 + n_aug * p

    sig_vars = [max(T(univariate_ar_variance(@view(Y_aug[:, v]))), T(1e-4)) for v in 1:n_aug]

    A0_m = Matrix{T}(I, n_aug, n_aug)
    B_m = zeros(T, k_lag, n_aug)
    resid_m = Matrix{T}(undef, T_eff, n_aug)
    se_m = zeros(T, k_lag, n_aug)
    var_cov_eqs = Vector{Matrix{T}}(undef, n_aug)
    m_beta_eqs = Vector{Vector{T}}(undef, n_aug)
    n_contemp_eqs = Vector{Int}(undef, n_aug)

    for (pos, eq) in enumerate(ord)
        n_contemp = pos - 1
        k_eq = k_lag + n_contemp
        n_contemp_eqs[eq] = n_contemp

        Z_eq = Matrix{T}(undef, T_eff, k_eq)
        @views Z_eq[:, 1] .= one(T)
        if n_contemp > 0
            @inbounds for j in 1:n_contemp
                @views Z_eq[:, 1 + j] .= Y_eff[:, ord[j]]
            end
        end
        @views Z_eq[:, (2 + n_contemp):k_eq] .= X_lag[:, 2:k_lag]
        y_eq = Vector{T}(@view Y_eff[:, eq])

        # Minnesota scales s_m for slopes
        s_vec = ones(T, k_eq)
        for j in 1:n_contemp
            s_vec[1 + j] = sig_vars[eq] / sig_vars[ord[j]]
        end
        for lag in 1:p, v in 1:n_aug
            idx_s = 1 + n_contemp + (lag - 1) * n_aug + v
            s_vec[idx_s] = sig_vars[eq] / (sig_vars[v] * T(lag))
        end

        # Warm-start m_beta with exact quantile regression
        m_beta = Vector{T}(undef, k_eq)
        ws = _QRegWorkspace{T}(T_eff, k_eq)
        _qreg_ipm!(m_beta, Z_eq, y_eq, tau, ws)

        V_beta = T(1e-3) .* Matrix{T}(I, k_eq, k_eq)
        r0_v = y_eq - Z_eq * m_beta
        E_inv_sig_v = one(T) / max(T(Statistics.var(r0_v)), T(1e-4))
        n_slopes = k_eq - 1
        E_inv_lam2 = ones(T, n_slopes)
        E_inv_nu = ones(T, n_slopes)
        E_inv_tau2 = one(T)
        E_inv_xi = one(T)
        E_inv_z_v = Vector{T}(undef, T_eff)
        m_prev = similar(m_beta)
        P0_eq = zeros(T, k_eq, k_eq)
        Z_w = Matrix{T}(undef, T_eff, k_eq)
        Delta_v = Vector{T}(undef, T_eff)

        for _ in 1:vb_maxiter
            copyto!(m_prev, m_beta)

            # 1. Latent z^v and scale σ^v
            lam_rate_v = E_inv_sig_v * inv_2tt
            b_v_sum = zero(T)
            @inbounds for t in 1:T_eff
                zt = @view Z_eq[t, :]
                rt = y_eq[t] - dot(zt, m_beta)
                qv = zero(T)
                for a in 1:k_eq, b in 1:k_eq
                    qv += zt[a] * V_beta[a, b] * zt[b]
                end
                dt = max(rt * rt + qv, delta_floor)
                Delta_v[t] = dt
                e_iz = min(one(T) / (tt * sqrt(dt)), T(1e6))
                e_z = one(T) / e_iz + one(T) / max(lam_rate_v, T(1e-8))
                E_inv_z_v[t] = e_iz
                b_v_sum += dt * e_iz - T(2) * theta_tau * rt + sum_mix * e_z
            end
            a_v_post = one(T) + T(1.5) * T(T_eff)
            b_v_post = T(0.01) + b_v_sum / (T(2) * kappa2_tau)
            E_inv_sig_v = clamp(a_v_post / max(b_v_post, T(1e-8)), T(1e-4), T(1e4))

            # 2. Horseshoe shrinkage updates (Makalic & Schmidt 2016)
            if n_slopes > 0
                sum_hs = zero(T)
                @inbounds for j in 1:n_slopes
                    eb2 = m_beta[1 + j]^2 + V_beta[1 + j, 1 + j]
                    s2 = s_vec[1 + j]^2
                    E_inv_lam2[j] = clamp(one(T) / (E_inv_nu[j] + eb2 * E_inv_tau2 / (T(2) * s2)),
                                          T(1e-6), T(1e5))
                    E_inv_nu[j] = one(T) / (one(T) + E_inv_lam2[j])
                    sum_hs += eb2 * E_inv_lam2[j] / s2
                end
                E_inv_tau2 = clamp((T(n_slopes) + one(T)) / T(2) / (E_inv_xi + T(0.5) * sum_hs),
                                   T(1e-6), T(1e5))
                E_inv_xi = one(T) / (one(T) + E_inv_tau2)
            end

            # 3. Regression posterior q(β_eq)
            fill!(P0_eq, zero(T))
            P0_eq[1, 1] = T(1e-4)
            @inbounds for j in 1:n_slopes
                P0_eq[1 + j, 1 + j] = clamp(E_inv_tau2 * E_inv_lam2[j] / (s_vec[1 + j]^2),
                                            T(1e-6), T(1e5))
            end
            w_scale = E_inv_sig_v / kappa2_tau
            @inbounds for j in 1:k_eq, t in 1:T_eff
                Z_w[t, j] = w_scale * E_inv_z_v[t] * Z_eq[t, j]
            end
            Prec_beta = Symmetric(T(0.5) .* (P0_eq .+ Z_w' * Z_eq .+ (P0_eq .+ Z_w' * Z_eq)'))
            V_beta = Matrix{T}(robust_inv(Prec_beta; silent=true))
            V_beta = T(0.5) .* (V_beta .+ V_beta')
            rhs_beta = w_scale .* (Z_eq' * (E_inv_z_v .* y_eq .- theta_tau))
            m_beta = V_beta * rhs_beta

            if maximum(abs, m_beta - m_prev) < vb_tol
                break
            end
        end

        B_m[1, eq] = m_beta[1]
        for j in 1:n_contemp
            A0_m[eq, ord[j]] = -m_beta[1 + j]
        end
        @views B_m[2:k_lag, eq] .= m_beta[(2 + n_contemp):k_eq]
        @views resid_m[:, eq] .= y_eq .- Z_eq * m_beta

        d_vb = diag(V_beta)
        se_m[1, eq] = sqrt(max(d_vb[1], zero(T)))
        for lag_i in 2:k_lag
            se_m[lag_i, eq] = sqrt(max(d_vb[n_contemp + lag_i], zero(T)))
        end
        var_cov_eqs[eq] = V_beta
        m_beta_eqs[eq] = m_beta
    end

    # --- Sample n_draws variational posterior credible draws ---
    A0_d = Array{T,3}(undef, n_draws, n_aug, n_aug)
    B_d = Array{T,3}(undef, n_draws, k_lag, n_aug)
    L_d = Array{T,3}(undef, n_draws, N, r)
    Ly_d = Array{T,3}(undef, n_draws, N, n_key)

    for eq in 1:n_aug
        L_V = safe_cholesky(var_cov_eqs[eq]; silent=true)
        k_eq = length(m_beta_eqs[eq])
        n_c = n_contemp_eqs[eq]
        pos_eq = findfirst(==(eq), ord)
        z_buf = Vector{T}(undef, k_eq)
        for d in 1:n_draws
            randn!(rng, z_buf)
            b_draw = m_beta_eqs[eq] + L_V * z_buf
            if pos_eq == 1
                A0_d[d, eq, :] .= zero(T)
            else
                A0_d[d, eq, :] .= zero(T)
                for j in 1:n_c
                    A0_d[d, eq, ord[j]] = -b_draw[1 + j]
                end
            end
            A0_d[d, eq, eq] = one(T)
            B_d[d, 1, eq] = b_draw[1]
            @views B_d[d, 2:k_lag, eq] .= b_draw[(2 + n_c):k_eq]
        end
    end

    z_load = Vector{T}(undef, k_load)
    for i in 1:N
        if key_pos[i] > 0 || anchor_pos[i] > 0
            for d in 1:n_draws
                @views L_d[d, i, :] .= Lambda_m[i, :]
                @views Ly_d[d, i, :] .= Lambda_y_m[i, :]
            end
        else
            L_Vl = safe_cholesky(V_load[i]; silent=true)
            mean_i = vcat(c_m[i], Lambda_m[i, :], Lambda_y_m[i, :])
            for d in 1:n_draws
                randn!(rng, z_load)
                th_d = mean_i + L_Vl * z_load
                @views L_d[d, i, :] .= th_d[2:(1 + r)]
                @views Ly_d[d, i, :] .= th_d[(2 + r):k_load]
            end
        end
    end

    return (
        F_m = F_m,
        Lambda_m = Lambda_m,
        Lambda_y_m = Lambda_y_m,
        c_m = c_m,
        V_f = V_f,
        var_cov = var_cov_eqs,
        A0_m = A0_m,
        B_m = B_m,
        resid_m = resid_m,
        se_m = se_m,
        A0_d = A0_d,
        B_d = B_d,
        L_d = L_d,
        Ly_d = Ly_d,
        converged = vb_conv,
        iterations = vb_iters,
    )
end

"""Variational Bayes Quantile FAVAR estimation."""
function _estimate_qfavar_vb(X_mat::Matrix{T}, Y_key_mat::Matrix{T},
                             Y_key_indices::Vector{Int}, r::Int, p::Int,
                             taus_T::Vector{T}, standardize::Bool,
                             pvn::Vector{String}, aug_varnames::Vector{String},
                             vb_maxiter::Int, vb_tol::Real, n_draws::Int,
                             rng_use::AbstractRNG,
                             seed::Union{Integer,Nothing}) where {T<:AbstractFloat}
    T_obs, N = size(X_mat)
    n_key = size(Y_key_mat, 2)
    n_aug = r + n_key
    M = length(taus_T)
    ord = collect(1:n_aug)
    vb_tol_T = max(T(vb_tol), T(10) * eps(T))

    X_std, mu_X, s_X = _qfavar_prep_panel(X_mat, standardize)
    Y_std, _, Y_stds = _qfavar_prep_panel(Y_key_mat, true)

    # Initialize factors via PCA and rotate to BBE (#528) anchor normalization
    fm = estimate_factors(X_mat, r; standardize=standardize)
    F_pca = Matrix{T}(fm.factors)
    W_init = hcat(F_pca, Y_std)
    coef_init = Matrix{T}(robust_inv(W_init' * W_init; silent=true)) * (W_init' * X_std)
    Lambda_init = Matrix{T}(permutedims(coef_init[1:r, :]))
    Lambda_y_init = Matrix{T}(permutedims(coef_init[(r + 1):n_aug, :]))

    anchor, G = _qfavar_select_anchors(Lambda_init, Y_key_indices, r, N)
    F_init = F_pca * G'
    Lambda_init = Lambda_init / G
    Lambda_init[anchor, :] .= Matrix{T}(I, r, r)
    Lambda_y_init[anchor, :] .= zero(T)
    Lambda_y_init ./= Y_stds'

    factors_mean = Vector{Matrix{T}}(undef, M)
    loadings_mean = Vector{Matrix{T}}(undef, M)
    lambda_y_mean = Vector{Matrix{T}}(undef, M)
    intercept_mean = Vector{Vector{T}}(undef, M)
    factor_cov = Vector{Array{T,3}}(undef, M)
    var_cov = Vector{Vector{Matrix{T}}}(undef, M)
    A0_draws = Vector{Array{T,3}}(undef, M)
    B_draws = Vector{Array{T,3}}(undef, M)
    loadings_draws = Vector{Array{T,3}}(undef, M)
    lambda_y_draws = Vector{Array{T,3}}(undef, M)
    vb_converged = Vector{Bool}(undef, M)
    vb_iterations = Vector{Int}(undef, M)

    A0_list = Vector{Matrix{T}}(undef, M)
    B_list = Vector{Matrix{T}}(undef, M)
    resid_list = Vector{Matrix{T}}(undef, M)
    se_list = Vector{Matrix{T}}(undef, M)

    for m in 1:M
        res_m = _qfavar_vb_single_tau(
            X_std, Y_key_mat, Y_key_indices, mu_X, s_X,
            F_init, Lambda_init, Lambda_y_init, anchor,
            r, p, taus_T[m], vb_maxiter, vb_tol_T, n_draws, rng_use
        )
        factors_mean[m] = res_m.F_m
        loadings_mean[m] = res_m.Lambda_m
        lambda_y_mean[m] = res_m.Lambda_y_m
        intercept_mean[m] = res_m.c_m
        factor_cov[m] = res_m.V_f
        var_cov[m] = res_m.var_cov
        A0_draws[m] = res_m.A0_d
        B_draws[m] = res_m.B_d
        loadings_draws[m] = res_m.L_d
        lambda_y_draws[m] = res_m.Ly_d
        vb_converged[m] = res_m.converged
        vb_iterations[m] = res_m.iterations

        A0_list[m] = res_m.A0_m
        B_list[m] = res_m.B_m
        resid_list[m] = res_m.resid_m
        se_list[m] = res_m.se_m
    end

    idx_ref = argmin(abs.(taus_T .- T(0.5)))
    Y_aug_ref = hcat(factors_mean[idx_ref], Y_key_mat)
    mani = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "method" => "vb", "r" => r, "p" => p, "taus" => Float64.(taus_T),
        "standardize" => standardize, "vb_maxiter" => vb_maxiter,
        "vb_tol" => Float64(vb_tol), "n_draws" => n_draws))

    qvar = QuantileVARModel{T}(A0_list, B_list, resid_list, se_list,
                               taus_T, ord, Y_aug_ref, p, aug_varnames, mani)

    return VBQuantileFAVAR{T}(
        qvar, factors_mean, loadings_mean, lambda_y_mean, intercept_mean,
        factor_cov, var_cov, A0_draws, B_draws, loadings_draws, lambda_y_draws,
        taus_T, X_mat, Y_key_mat, Y_key_indices, r, n_key, p, standardize,
        vb_converged, vb_iterations, pvn, aug_varnames, mani
    )
end

# =============================================================================
# Public Estimation API
# =============================================================================

"""
    estimate_qfavar(X_panel, Y_key, r, p, taus=[0.10, 0.50, 0.90];
                    method=:twostep, standardize=true, panel_varnames=nothing,
                    key_names=nothing, se=:none, vb_maxiter=100, vb_tol=1e-4,
                    n_draws=200, rng=Random.default_rng(), seed=nothing)
        -> Union{QuantileFAVAR, VBQuantileFAVAR}

Estimate a Quantile Factor-Augmented VAR (Korobilis & Schröder 2025) across quantile
levels `taus`:

```math
X_{t} = c(\\tau) + \\Lambda(\\tau)\\, F_t(\\tau) + \\Lambda_y(\\tau)\\, Y_{\\text{key}, t} + e_t(\\tau),
\\qquad
A_0(\\tau)\\, z_t = \\alpha(\\tau) + \\sum_{\\ell=1}^p A_\\ell(\\tau)\\, z_{t-\\ell} + u_t(\\tau)
```

where:
- ``X_t`` is the ``N \\times 1`` panel vector at time ``t``
- ``F_t(\\tau)`` is the ``r \\times 1`` latent quantile factor vector
- ``Y_{\\text{key}, t}`` is the ``n_{\\text{key}} \\times 1`` vector of observed key variables
- ``z_t = [F_t(\\tau)', Y_{\\text{key}, t}']'`` is the augmented ``(r + n_{\\text{key}}) \\times 1`` state vector
- ``\\Lambda(\\tau), \\Lambda_y(\\tau)`` are factor and direct key-variable loadings honoring `#528` unit-impact identity on `Y_key_indices`

# Arguments
- `X_panel::AbstractMatrix` — large panel matrix (``T_{\\text{obs}} \\times N``)
- `Y_key::Union{AbstractMatrix,AbstractVector{<:Integer}}` — key variable matrix (``T_{\\text{obs}} \\times n_{\\text{key}}``) or column indices into `X_panel`
- `r::Int` — number of latent quantile factors (``1 \\le r \\le \\min(T_{\\text{obs}}, N)``)
- `p::Int` — QVAR lag order (``p \\ge 1``)
- `taus::AbstractVector{<:Real} = [0.10, 0.50, 0.90]` — quantile levels strictly in ``(0, 1)``

# Keyword Arguments
- `method::Symbol = :twostep` — `:twostep` (QFM + BBE purging + QVAR, returning [`QuantileFAVAR`](@ref)) or `:vb` (Asymmetric Laplace Variational Bayes with Minnesota-Horseshoe shrinkage, returning [`VBQuantileFAVAR`](@ref))
- `standardize::Bool = true` — standardize panel series prior to factor extraction
- `panel_varnames::Union{Nothing,AbstractVector{<:AbstractString}} = nothing` — names for the ``N`` panel series
- `key_names::Union{Nothing,AbstractVector{<:AbstractString}} = nothing` — names for the ``n_{\\text{key}}`` key variables
- `se::Symbol = :none` — standard error mode passed to two-step QVAR (`:none`, `:asymptotic`, or `:bootstrap`)
- `vb_maxiter::Int = 100` — maximum CAVI iterations per quantile when `method == :vb`
- `vb_tol::Real = 1e-4` — CAVI convergence tolerance when `method == :vb`
- `n_draws::Int = 200` — number of variational posterior credible draws when `method == :vb`
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `seed::Union{Integer,Nothing} = nothing` — optional integer seed recorded in `manifest`

# Returns
A [`QuantileFAVAR`](@ref) when `method == :twostep` or [`VBQuantileFAVAR`](@ref) when `method == :vb`.

# Example
```julia
X = randn(200, 30)
qf = estimate_qfavar(X, [1], 2, 1, [0.10, 0.50, 0.90]; method=:twostep, seed=42)
ir = qfavar_panel_irf(qf, 10; tau_profile=0.10)
```

# References
- Korobilis, D., & Schröder, M. (2025). Monitoring macroeconomic tail risk with quantile
  factor-augmented VARs. *Journal of Econometrics*, forthcoming.
- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects of monetary policy:
  A Factor-Augmented Vector Autoregressive (FAVAR) approach. *Quarterly Journal of Economics*,
  120(1), 387-422.
"""
function estimate_qfavar(X_panel::AbstractMatrix{T}, Y_key::AbstractMatrix{T},
                         r::Int, p::Int,
                         taus::AbstractVector{<:Real}=T[T(0.10), T(0.50), T(0.90)];
                         method::Symbol=:twostep,
                         standardize::Bool=true,
                         panel_varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                         key_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                         se::Symbol=:none,
                         vb_maxiter::Int=100,
                         vb_tol::Real=1e-4,
                         n_draws::Int=200,
                         rng::AbstractRNG=Random.default_rng(),
                         seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X_panel, "X_panel")
    _validate_data(Y_key, "Y_key")
    T_obs, N = size(X_panel)
    T_key, n_key = size(Y_key)
    T_obs == T_key || throw(ArgumentError(
        "X_panel has $T_obs rows but Y_key has $T_key rows; they must match"))
    n_key >= 1 || throw(ArgumentError("Y_key must have at least 1 column"))
    validate_factor_inputs(T_obs, N, r)
    validate_var_inputs(T_obs, r + n_key, p)
    T_eff = T_obs - p
    n_aug = r + n_key
    T_eff > n_aug * (p + 1) || throw(ArgumentError(
        "Not enough effective observations (T_eff=$T_eff) for augmented QVAR($p) with $n_aug variables"))

    !isempty(taus) || throw(ArgumentError("taus must not be empty"))
    taus_T = Vector{T}(undef, length(taus))
    for (m, tm) in enumerate(taus)
        tm_T = T(tm)
        (isfinite(tm) && zero(T) < tm_T < one(T)) ||
            throw(ArgumentError("every tau must lie strictly in (0, 1), got $tm"))
        taus_T[m] = tm_T
    end

    validate_option(method, "method", (:twostep, :vb))
    validate_option(se, "se", (:none, :asymptotic, :bootstrap))
    vb_maxiter >= 1 || throw(ArgumentError("vb_maxiter must be >= 1, got $vb_maxiter"))
    (isfinite(vb_tol) && vb_tol > 0) || throw(ArgumentError("vb_tol must be > 0, got $vb_tol"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1, got $n_draws"))

    pvn = panel_varnames === nothing ? ["X$i" for i in 1:N] : String.(collect(panel_varnames))
    length(pvn) == N || throw(ArgumentError(
        "panel_varnames has $(length(pvn)) entries but X_panel has $N columns"))

    X_mat = Matrix{T}(X_panel)
    Y_key_mat = Matrix{T}(Y_key)
    Y_key_indices = _find_key_indices(X_mat, Y_key_mat)

    return _estimate_qfavar_core(X_mat, Y_key_mat, Y_key_indices, r, p, taus_T,
                                 method, standardize, pvn, key_names, se,
                                 vb_maxiter, vb_tol, n_draws, rng, seed)
end

function estimate_qfavar(X_panel::AbstractMatrix{T}, Y_key::AbstractVector{<:Integer},
                         r::Int, p::Int,
                         taus::AbstractVector{<:Real}=T[T(0.10), T(0.50), T(0.90)];
                         method::Symbol=:twostep,
                         standardize::Bool=true,
                         panel_varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                         key_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                         se::Symbol=:none,
                         vb_maxiter::Int=100,
                         vb_tol::Real=1e-4,
                         n_draws::Int=200,
                         rng::AbstractRNG=Random.default_rng(),
                         seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X_panel, "X_panel")
    T_obs, N = size(X_panel)
    !isempty(Y_key) || throw(ArgumentError("Y_key indices must not be empty"))
    idx_vec = Int.(collect(Y_key))
    for idx in idx_vec
        1 <= idx <= N || throw(ArgumentError(
            "Key variable index $idx is out of range [1, $N]"))
    end
    allunique(idx_vec) || throw(ArgumentError(
        "Y_key column indices must be unique, got $idx_vec"))

    n_key = length(idx_vec)
    validate_factor_inputs(T_obs, N, r)
    validate_var_inputs(T_obs, r + n_key, p)
    T_eff = T_obs - p
    n_aug = r + n_key
    T_eff > n_aug * (p + 1) || throw(ArgumentError(
        "Not enough effective observations (T_eff=$T_eff) for augmented QVAR($p) with $n_aug variables"))

    !isempty(taus) || throw(ArgumentError("taus must not be empty"))
    taus_T = Vector{T}(undef, length(taus))
    for (m, tm) in enumerate(taus)
        tm_T = T(tm)
        (isfinite(tm) && zero(T) < tm_T < one(T)) ||
            throw(ArgumentError("every tau must lie strictly in (0, 1), got $tm"))
        taus_T[m] = tm_T
    end

    validate_option(method, "method", (:twostep, :vb))
    validate_option(se, "se", (:none, :asymptotic, :bootstrap))
    vb_maxiter >= 1 || throw(ArgumentError("vb_maxiter must be >= 1, got $vb_maxiter"))
    (isfinite(vb_tol) && vb_tol > 0) || throw(ArgumentError("vb_tol must be > 0, got $vb_tol"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be >= 1, got $n_draws"))

    pvn = panel_varnames === nothing ? ["X$i" for i in 1:N] : String.(collect(panel_varnames))
    length(pvn) == N || throw(ArgumentError(
        "panel_varnames has $(length(pvn)) entries but X_panel has $N columns"))

    X_mat = Matrix{T}(X_panel)
    Y_key_mat = Matrix{T}(X_mat[:, idx_vec])
    return _estimate_qfavar_core(X_mat, Y_key_mat, idx_vec, r, p, taus_T,
                                 method, standardize, pvn, key_names, se,
                                 vb_maxiter, vb_tol, n_draws, rng, seed)
end

function estimate_qfavar(X_panel::AbstractMatrix, Y_key::AbstractMatrix,
                         r::Int, p::Int,
                         taus::AbstractVector{<:Real}=[0.10, 0.50, 0.90];
                         kwargs...)
    FP = float(promote_type(eltype(X_panel), eltype(Y_key)))
    estimate_qfavar(Matrix{FP}(X_panel), Matrix{FP}(Y_key), r, p, taus; kwargs...)
end

function estimate_qfavar(X_panel::AbstractMatrix, Y_key::AbstractVector{<:Integer},
                         r::Int, p::Int,
                         taus::AbstractVector{<:Real}=[0.10, 0.50, 0.90];
                         kwargs...)
    FP = float(eltype(X_panel))
    estimate_qfavar(Matrix{FP}(X_panel), Y_key, r, p, taus; kwargs...)
end

function _estimate_qfavar_core(X_mat::Matrix{T}, Y_key_mat::Matrix{T},
                               Y_key_indices::Vector{Int}, r::Int, p::Int,
                               taus_T::Vector{T}, method::Symbol, standardize::Bool,
                               pvn::Vector{String},
                               key_names::Union{Nothing,AbstractVector{<:AbstractString}},
                               se::Symbol, vb_maxiter::Int, vb_tol::Real,
                               n_draws::Int, rng::AbstractRNG,
                               seed::Union{Integer,Nothing}) where {T<:AbstractFloat}
    N = size(X_mat, 2)
    n_key = size(Y_key_mat, 2)
    kn = if key_names !== nothing
        kv = String.(collect(key_names))
        length(kv) == n_key || throw(ArgumentError(
            "key_names has $(length(kv)) entries but Y_key has $n_key columns"))
        kv
    elseif !isempty(Y_key_indices) && all(idx -> 1 <= idx <= N, Y_key_indices)
        String[pvn[idx] for idx in Y_key_indices]
    else
        String["Y$k" for k in 1:n_key]
    end
    aug_varnames = vcat(String["F$j" for j in 1:r], kn)
    rng_use = _resolve_repro_rng(rng, seed)

    if method === :twostep
        return _estimate_qfavar_twostep(
            X_mat, Y_key_mat, Y_key_indices, r, p, taus_T,
            standardize, pvn, aug_varnames, se, n_draws, rng_use, seed
        )
    else
        return _estimate_qfavar_vb(
            X_mat, Y_key_mat, Y_key_indices, r, p, taus_T,
            standardize, pvn, aug_varnames, vb_maxiter, vb_tol, n_draws, rng_use, seed
        )
    end
end

# =============================================================================
# Panel-Wide QIRF & Panel Quantile Forecast
# =============================================================================

"""Interpolate `(Λ_std(τ), Λ_{y, std}(τ))` across the model's sorted quantile grid."""
function _qfavar_interp_loadings(loadings_vec::Vector{Matrix{T}},
                                 lambda_y_vec::Vector{Matrix{T}},
                                 taus::Vector{T}, tau::T) where {T<:AbstractFloat}
    M = length(taus)
    M == 1 && return loadings_vec[1], lambda_y_vec[1]
    perm = sortperm(taus)
    ts = @view taus[perm]

    tol_match = max(T(1e-8), T(10) * eps(T))
    for idx in perm
        if abs(taus[idx] - tau) <= tol_match
            return loadings_vec[idx], lambda_y_vec[idx]
        end
    end

    if tau <= ts[1]
        return loadings_vec[perm[1]], lambda_y_vec[perm[1]]
    elseif tau >= ts[M]
        return loadings_vec[perm[M]], lambda_y_vec[perm[M]]
    end

    hi = searchsortedfirst(ts, tau)
    lo = hi - 1
    w = (tau - ts[lo]) / (ts[hi] - ts[lo])
    ilo, ihi = perm[lo], perm[hi]
    return (one(T) - w) .* loadings_vec[ilo] .+ w .* loadings_vec[ihi],
           (one(T) - w) .* lambda_y_vec[ilo] .+ w .* lambda_y_vec[ihi]
end

_qfavar_loadings_list(m::QuantileFAVAR) = (m.loadings_list, m.lambda_y_list, m.intercept_list, m.factors_list)
_qfavar_loadings_list(m::VBQuantileFAVAR) = (m.loadings_mean, m.lambda_y_mean, m.intercept_mean, m.factors_mean)

"""Map factor-space QIRF values `(H × (r+n_key) × n_shocks)` to raw panel units `(H × N × n_shocks)`."""
function _qfavar_map_irf_to_panel(aug_vals::Array{T,3},
                                  loadings_vec::Vector{Matrix{T}},
                                  lambda_y_vec::Vector{Matrix{T}},
                                  taus::Vector{T}, tau_h::Vector{T},
                                  s_X::Vector{T}, key_indices::Vector{Int},
                                  r::Int, n_key::Int, N::Int) where {T<:AbstractFloat}
    H = size(aug_vals, 1)
    n_shocks = size(aug_vals, 3)
    panel_vals = zeros(T, H, N, n_shocks)

    for h in 1:H
        Lam_std, Ly_std = _qfavar_interp_loadings(loadings_vec, lambda_y_vec, taus, tau_h[h])
        Lam_raw = Lam_std .* s_X
        Ly_raw = Ly_std .* s_X
        for s in 1:n_shocks
            f_irf = @view aug_vals[h, 1:r, s]
            y_irf = @view aug_vals[h, (r + 1):(r + n_key), s]
            @views panel_vals[h, :, s] .= Lam_raw * f_irf .+ Ly_raw * y_irf
        end
    end

    # Exact #528 unit-impact identity on key panel columns (eliminates s_X / s_X rounding)
    for (k, panel_idx) in enumerate(key_indices)
        (1 <= panel_idx <= N) || continue
        @inbounds for h in 1:H, s in 1:n_shocks
            panel_vals[h, panel_idx, s] = aug_vals[h, r + k, s]
        end
    end
    return panel_vals
end

"""
    qfavar_panel_irf(m, H; shock=nothing, shock_size=1.0, tau_profile=0.10,
                     baseline_tau=0.50, method=:analytical, ci_type=:none,
                     n_boot=199, conf_level=0.95, rng=Random.default_rng(),
                     seed=nothing) -> QuantileImpulseResponse

Compute panel-wide Quantile Impulse Response Functions across all ``N`` panel series
from a [`QuantileFAVAR`](@ref) or [`VBQuantileFAVAR`](@ref) model, rescaled to raw
panel units so that `Y_key_indices` columns satisfy the exact `#528` unit-impact identity
at impact ``h = 0``:

```math
\\psi_{h, \\cdot, j}(\\tau_h) = \\Lambda_{\\text{raw}}(\\tau_h) \\, \\theta_{h, F, j}(\\tau_h) +
\\Lambda_{y, \\text{raw}}(\\tau_h) \\, \\theta_{h, Y, j}(\\tau_h)
```

where:
- ``\\Lambda_{\\text{raw}}(\\tau_h)`` is the ``N \\times r`` loading matrix in raw panel units
- ``\\Lambda_{y, \\text{raw}}(\\tau_h)`` is the ``N \\times n_{\\text{key}}`` direct key-variable loading matrix in raw panel units
- ``\\theta_{h, F, j}(\\tau_h)`` and ``\\theta_{h, Y, j}(\\tau_h)`` are the factor and key-variable QIRFs from [`qirf`](@ref)

# Arguments
- `m::Union{QuantileFAVAR,VBQuantileFAVAR}` — estimated Quantile FAVAR model
- `H::Int` — IRF horizon length (``H \\ge 1``)

# Keyword Arguments
- `shock::Union{Int,Nothing} = nothing` — specific augmented shock index (`1:(r+n_key)`) or `nothing` for all shocks
- `shock_size::Real = 1.0` — structural shock magnitude
- `tau_profile::Union{Real,AbstractVector{<:Real}} = 0.10` — scalar quantile or horizon-specific quantile vector
- `baseline_tau::Real = 0.50` — baseline reference quantile
- `method::Symbol = :analytical` — QIRF propagation method (`:analytical` or `:simulation`)
- `ci_type::Symbol = :none` — interval type (`:none`, `:bootstrap`, or `:credible` for [`VBQuantileFAVAR`](@ref))
- `n_boot::Int = 199` — bootstrap replications when `ci_type == :bootstrap`
- `conf_level::Real = 0.95` — confidence / credible level strictly in ``(0, 1)``
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `seed::Union{Integer,Nothing} = nothing` — optional integer seed

# Returns
A [`QuantileImpulseResponse`](@ref) over all ``N`` panel series.
"""
function qfavar_panel_irf(m::Union{QuantileFAVAR{T},VBQuantileFAVAR{T}}, H::Int;
                          shock::Union{Int,Nothing}=nothing,
                          shock_size::Real=1.0,
                          tau_profile::Union{Real,AbstractVector{<:Real}}=0.10,
                          baseline_tau::Real=0.50,
                          method::Symbol=:analytical,
                          ci_type::Symbol=:none,
                          n_boot::Int=199,
                          conf_level::Real=0.95,
                          rng::AbstractRNG=Random.default_rng(),
                          seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("horizon H must be >= 1, got $H"))
    validate_option(method, "method", (:analytical, :simulation))
    validate_option(ci_type, "ci_type", (:none, :bootstrap, :credible))
    if ci_type === :credible && !(m isa VBQuantileFAVAR)
        throw(ArgumentError("ci_type=:credible is only supported for VBQuantileFAVAR; use :bootstrap or :none"))
    end
    conf_T = T(conf_level)
    (isfinite(conf_level) && zero(T) < conf_T < one(T)) ||
        throw(ArgumentError("conf_level must lie strictly in (0, 1), got $conf_level"))

    r = m.n_factors
    n_key = m.n_key
    N = size(m.X_panel, 2)
    rng_use = _resolve_repro_rng(rng, seed)

    _, s_X = m.standardize ?
        (vec(Statistics.mean(m.X_panel; dims=1)), max.(vec(Statistics.std(m.X_panel; dims=1)), max(T(1e-10), eps(T)))) :
        (zeros(T, N), ones(T, N))

    load_vec, ly_vec, _, _ = _qfavar_loadings_list(m)
    aug_qirf = qirf(m.qvar, H;
                    shock=shock, shock_size=shock_size,
                    tau_profile=tau_profile, baseline_tau=baseline_tau,
                    method=method, ci_type=:none, n_boot=n_boot,
                    conf_level=conf_level, rng=rng_use, seed=nothing)

    panel_vals = _qfavar_map_irf_to_panel(
        aug_qirf.values, load_vec, ly_vec, m.taus, aug_qirf.tau_profile,
        s_X, m.Y_key_indices, r, n_key, N
    )

    n_shocks = size(panel_vals, 3)
    ci_lo = zeros(T, H, N, n_shocks)
    ci_hi = zeros(T, H, N, n_shocks)

    if ci_type === :credible && m isa VBQuantileFAVAR
        n_dr = size(m.B_draws[1], 1)
        M = length(m.taus)
        panel_draws = Array{T,4}(undef, n_dr, H, N, n_shocks)
        for d in 1:n_dr
            A0_d = [Matrix{T}(m.A0_draws[mi][d, :, :]) for mi in 1:M]
            B_d = [Matrix{T}(m.B_draws[mi][d, :, :]) for mi in 1:M]
            L_d = [Matrix{T}(m.loadings_draws[mi][d, :, :]) for mi in 1:M]
            Ly_d = [Matrix{T}(m.lambda_y_draws[mi][d, :, :]) for mi in 1:M]
            qv_d = QuantileVARModel{T}(A0_d, B_d, m.qvar.resid_list, m.qvar.se_list,
                                       m.taus, m.qvar.ordering, m.qvar.Y, m.p, m.varnames, nothing)
            aug_d = if method === :analytical
                _qirf_analytical(qv_d, H, aug_qirf.tau_profile, shock, aug_qirf.shock_size)
            else
                _qirf_simulation(qv_d, H, aug_qirf.tau_profile, aug_qirf.baseline_tau,
                                 shock, aug_qirf.shock_size, 100, rng_use)
            end
            @views panel_draws[d, :, :, :] .= _qfavar_map_irf_to_panel(
                aug_d, L_d, Ly_d, m.taus, aug_qirf.tau_profile,
                s_X, m.Y_key_indices, r, n_key, N
            )
        end
        alpha = (one(T) - conf_T) / 2
        @inbounds for h in 1:H, v in 1:N, s in 1:n_shocks
            dv = @view panel_draws[:, h, v, s]
            ci_lo[h, v, s] = T(Statistics.quantile(dv, alpha))
            ci_hi[h, v, s] = T(Statistics.quantile(dv, one(T) - alpha))
        end
    elseif ci_type === :bootstrap
        n_boot >= 2 || throw(ArgumentError("n_boot must be >= 2 when ci_type == :bootstrap, got $n_boot"))
        Y_eff, X_lag = construct_var_matrices(m.qvar.Y, m.p)
        T_eff = size(Y_eff, 1)
        n_aug = r + n_key
        k_lag = 1 + n_aug * m.p
        ell = clamp(ceil(Int, cbrt(T_eff)), 1, T_eff)
        last_start = max(T_eff - ell + 1, 1)
        idx = Vector{Int}(undef, T_eff)
        Yb = Matrix{T}(undef, T_eff, n_aug)
        Xb = Matrix{T}(undef, T_eff, k_lag)
        boot_taus = method === :analytical ? unique(vcat(aug_qirf.tau_profile, aug_qirf.baseline_tau)) : copy(m.taus)
        M_b = length(boot_taus)
        panel_draws = Array{T,4}(undef, n_boot, H, N, n_shocks)
        kept = 0

        for _ in 1:n_boot
            pos_idx = 1
            while pos_idx <= T_eff
                s = rand(rng_use, 1:last_start)
                take = min(ell, T_eff - pos_idx + 1)
                @inbounds for j in 0:(take - 1)
                    idx[pos_idx + j] = s + j
                end
                pos_idx += take
            end
            @inbounds for col in 1:n_aug, row in 1:T_eff
                Yb[row, col] = Y_eff[idx[row], col]
            end
            @inbounds for col in 1:k_lag, row in 1:T_eff
                Xb[row, col] = X_lag[idx[row], col]
            end

            A0_b_list = Vector{Matrix{T}}(undef, M_b)
            B_b_list = Vector{Matrix{T}}(undef, M_b)
            R_b_list = Vector{Matrix{T}}(undef, M_b)
            se_b_list = Vector{Matrix{T}}(undef, M_b)
            ok = true
            for mi in 1:M_b
                A0_bm, B_bm, R_bm, se_bm = _qvar_fit_single_point(Yb, Xb, n_aug, m.p, boot_taus[mi], m.qvar.ordering)
                if !(all(isfinite, A0_bm) && all(isfinite, B_bm))
                    ok = false
                    break
                end
                A0_b_list[mi] = A0_bm
                B_b_list[mi] = B_bm
                R_b_list[mi] = R_bm
                se_b_list[mi] = se_bm
            end
            ok || continue

            qv_b = QuantileVARModel{T}(A0_b_list, B_b_list, R_b_list, se_b_list,
                                       boot_taus, m.qvar.ordering, m.qvar.Y, m.p, m.varnames, nothing)
            aug_b = if method === :analytical
                _qirf_analytical(qv_b, H, aug_qirf.tau_profile, shock, aug_qirf.shock_size)
            else
                _qirf_simulation(qv_b, H, aug_qirf.tau_profile, aug_qirf.baseline_tau,
                                 shock, aug_qirf.shock_size, 60, rng_use)
            end
            if all(isfinite, aug_b)
                kept += 1
                @views panel_draws[kept, :, :, :] .= _qfavar_map_irf_to_panel(
                    aug_b, load_vec, ly_vec, m.taus, aug_qirf.tau_profile,
                    s_X, m.Y_key_indices, r, n_key, N
                )
            end
        end
        if kept >= 2
            alpha = (one(T) - conf_T) / 2
            d_sub = @view panel_draws[1:kept, :, :, :]
            @inbounds for h in 1:H, v in 1:N, s in 1:n_shocks
                dv = @view d_sub[:, h, v, s]
                ci_lo[h, v, s] = T(Statistics.quantile(dv, alpha))
                ci_hi[h, v, s] = T(Statistics.quantile(dv, one(T) - alpha))
            end
        end
    end

    out_method = (m isa VBQuantileFAVAR && method === :analytical) ? :vb : method
    return QuantileImpulseResponse{T}(
        panel_vals, ci_lo, ci_hi, H, aug_qirf.tau_profile, aug_qirf.baseline_tau,
        aug_qirf.shock_size, out_method, ci_type, ci_type === :none ? zero(T) : conf_T,
        copy(m.panel_varnames), aug_qirf.shocks
    )
end

"""
    qfavar_panel_forecast(m, h; rearrange=true) -> NamedTuple

Compute multi-step panel quantile forecasts across all ``N`` panel series and all
estimated quantiles `m.taus` from a [`QuantileFAVAR`](@ref) or [`VBQuantileFAVAR`](@ref).

For each quantile level ``\\tau_m``, iterates the quantile-specific reduced-form VAR forward
``h`` steps from the final ``p`` observations of ``[\\tilde{F}_t(\\tau_m), Y_{\\text{key}, t}]``
and maps the predicted states to raw panel units via the quantile measurement equation
(honoring the exact `#528` identity on `Y_key_indices`).

When `rearrange=true` (default), applies Chernozhukov, Fernández-Val & Galichon (2010)
quantile rearrangement at each horizon and series so predicted quantiles are monotonically
non-decreasing across `m.taus`.

# Arguments
- `m::Union{QuantileFAVAR,VBQuantileFAVAR}` — estimated Quantile FAVAR model
- `h::Int` — forecast horizon (``h \\ge 1``)

# Keyword Arguments
- `rearrange::Bool = true` — enforce monotonic quantile ordering across `taus`

# Returns
A `NamedTuple` with fields:
- `quantiles::Array{T,3}` — ``h \\times N \\times M`` panel forecasts aligned with `m.taus`
- `forecast::Matrix{T}` — ``h \\times N`` central forecast (quantile closest to ``0.50``)
- `ci_lower::Matrix{T}` — ``h \\times N`` lowest-quantile forecast
- `ci_upper::Matrix{T}` — ``h \\times N`` highest-quantile forecast
- `factors::Array{T,3}` — ``h \\times r \\times M`` latent factor forecasts
- `taus::Vector{T}` — quantile levels
- `horizon::Int` — forecast horizon ``h``
- `varnames::Vector{String}` — panel variable names
- `rearranged::Bool` — whether quantile rearrangement was applied

# References
- Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and probability
  curves without crossing. *Econometrica*, 78(3), 1093-1125.
"""
function qfavar_panel_forecast(m::Union{QuantileFAVAR{T},VBQuantileFAVAR{T}}, h::Int;
                               rearrange::Bool=true) where {T<:AbstractFloat}
    h >= 1 || throw(ArgumentError("forecast horizon h must be >= 1, got $h"))
    r = m.n_factors
    n_key = m.n_key
    n_aug = r + n_key
    N = size(m.X_panel, 2)
    p = m.p
    M = length(m.taus)

    mu_X, s_X = m.standardize ?
        (vec(Statistics.mean(m.X_panel; dims=1)), max.(vec(Statistics.std(m.X_panel; dims=1)), max(T(1e-10), eps(T)))) :
        (zeros(T, N), ones(T, N))

    load_vec, ly_vec, int_vec, fac_vec = _qfavar_loadings_list(m)

    panel_q = zeros(T, h, N, M)
    factor_q = zeros(T, h, r, M)

    for mi in 1:M
        F_q = fac_vec[mi]
        hist = hcat(F_q[(end - p + 1):end, :], m.Y_key[(end - p + 1):end, :])

        A0_inv = Matrix{T}(robust_inv(m.qvar.A0_list[mi]; silent=true))
        B_s = m.qvar.B_list[mi]
        c_red = A0_inv * @view(B_s[1, :])
        A_struct = extract_ar_coefficients(B_s, n_aug, p)
        A_red = [A0_inv * A_struct[lag] for lag in 1:p]

        Lam_raw = load_vec[mi] .* s_X
        Ly_raw = ly_vec[mi] .* s_X
        c_raw = mu_X .+ s_X .* int_vec[mi]

        for step in 1:h
            y_hat = copy(c_red)
            for lag in 1:p
                y_hat .+= A_red[lag] * @view(hist[p - lag + 1, :])
            end
            @views factor_q[step, :, mi] .= y_hat[1:r]

            x_hat = c_raw .+ Lam_raw * @view(y_hat[1:r]) .+ Ly_raw * @view(y_hat[(r + 1):n_aug])
            for (k, panel_idx) in enumerate(m.Y_key_indices)
                (1 <= panel_idx <= N) || continue
                x_hat[panel_idx] = y_hat[r + k]
            end
            @views panel_q[step, :, mi] .= x_hat

            for row in 1:(p - 1)
                @views hist[row, :] .= hist[row + 1, :]
            end
            @views hist[p, :] .= y_hat
        end
    end

    if rearrange && M > 1
        perm = sortperm(m.taus)
        @inbounds for step in 1:h, i in 1:N
            sorted_vals = sort(@view(panel_q[step, i, :]))
            for (k, p_idx) in enumerate(perm)
                panel_q[step, i, p_idx] = sorted_vals[k]
            end
        end
    end

    idx_med = argmin(abs.(m.taus .- T(0.5)))
    idx_lo = argmin(m.taus)
    idx_hi = argmax(m.taus)

    return (
        quantiles = panel_q,
        forecast = Matrix{T}(panel_q[:, :, idx_med]),
        ci_lower = Matrix{T}(panel_q[:, :, idx_lo]),
        ci_upper = Matrix{T}(panel_q[:, :, idx_hi]),
        factors = factor_q,
        taus = copy(m.taus),
        horizon = h,
        varnames = copy(m.panel_varnames),
        rearranged = rearrange,
    )
end

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, m::QuantileFAVAR{T}) where {T}
    T_obs, N = size(m.X_panel)
    ts = join([_fmt(t; digits=2) for t in m.taus], ", ")
    spec = Any[
        "Method"           "Two-Step (QFM + QVAR)";
        "Factors (r)"      m.n_factors;
        "Key variables"    m.n_key;
        "VAR variables"    nvars(m);
        "Panel variables"  N;
        "Lags (p)"         m.p;
        "Observations"     T_obs;
        "Standardized"     m.standardize ? "Yes" : "No";
        "Quantiles (τ)"    ts
    ]
    _pretty_table(io, spec;
        title = "Quantile FAVAR($(m.p)) — $(m.n_factors) factors, $(m.n_key) key vars",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    M = length(m.taus)
    summary_mat = Matrix{Any}(undef, M, 4)
    for (mi, tau) in enumerate(m.taus)
        qfm = m.qfm_models[mi]
        summary_mat[mi, 1] = _fmt(tau; digits=2)
        summary_mat[mi, 2] = _fmt(qfm.objective; digits=4)
        summary_mat[mi, 3] = _fmt(Statistics.mean(qfm.pseudo_r2); digits=4)
        summary_mat[mi, 4] = qfm.converged ? "Yes" : "No"
    end
    _pretty_table(io, summary_mat;
        title = "Quantile Factor Summary",
        column_labels = ["τ", "Check Loss", "Mean R¹", "Converged"],
        alignment = [:r, :r, :r, :c],
    )
    _manifest_footer(io, m.manifest)
    return nothing
end

function Base.show(io::IO, m::VBQuantileFAVAR{T}) where {T}
    T_obs, N = size(m.X_panel)
    ts = join([_fmt(t; digits=2) for t in m.taus], ", ")
    n_dr = size(m.B_draws[1], 1)
    spec = Any[
        "Method"           "Variational Bayes (AL-Mixture)";
        "Factors (r)"      m.n_factors;
        "Key variables"    m.n_key;
        "VAR variables"    nvars(m);
        "Panel variables"  N;
        "Lags (p)"         m.p;
        "Observations"     T_obs;
        "Standardized"     m.standardize ? "Yes" : "No";
        "Credible draws"   n_dr;
        "Quantiles (τ)"    ts
    ]
    _pretty_table(io, spec;
        title = "VB Quantile FAVAR($(m.p)) — $(m.n_factors) factors, $(m.n_key) key vars",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    M = length(m.taus)
    conv_mat = Matrix{Any}(undef, M, 3)
    for (mi, tau) in enumerate(m.taus)
        conv_mat[mi, 1] = _fmt(tau; digits=2)
        conv_mat[mi, 2] = m.vb_converged[mi] ? "Yes" : "No"
        conv_mat[mi, 3] = m.vb_iterations[mi]
    end
    _pretty_table(io, conv_mat;
        title = "Variational Bayes Convergence",
        column_labels = ["τ", "Converged", "Iterations"],
        alignment = [:r, :c, :r],
    )
    _manifest_footer(io, m.manifest)
    return nothing
end

"""
    report(m::QuantileFAVAR; repro=false)

Print a comprehensive summary of the two-step [`QuantileFAVAR`](@ref) estimation,
factor fit diagnostics across quantiles, and underlying [`QuantileVARModel`](@ref).
Pass `repro=true` to include the reproducibility manifest footer.
"""
report(m::QuantileFAVAR; repro::Bool=false) = report(stdout, m; repro=repro)
function report(io::IO, m::QuantileFAVAR; repro::Bool=false)
    show(IOContext(io, :repro => repro), m)
    println(io)
    show(IOContext(io, :repro => false), m.qvar)
    return nothing
end

"""
    report(m::VBQuantileFAVAR; repro=false)

Print a comprehensive summary of the Variational Bayes [`VBQuantileFAVAR`](@ref)
estimation, CAVI convergence diagnostics, and posterior-mean [`QuantileVARModel`](@ref).
Pass `repro=true` to include the reproducibility manifest footer.
"""
report(m::VBQuantileFAVAR; repro::Bool=false) = report(stdout, m; repro=repro)
function report(io::IO, m::VBQuantileFAVAR; repro::Bool=false)
    show(IOContext(io, :repro => repro), m)
    println(io)
    show(IOContext(io, :repro => false), m.qvar)
    return nothing
end
