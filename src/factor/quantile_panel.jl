# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel quantile regression with interactive quantile factor effects (Ando & Bai 2020).

Combines observed panel covariates `W_{it}` with unobserved quantile-specific factor
structures `λ_i(τ)' f_t(τ)`:

    Q_{Y_{it}}(τ | W_{it}, f_t(τ), λ_i(τ)) = W_{it}' β(τ) + λ_i(τ)' f_t(τ)

supporting both homogeneous (`β(τ)`) and unit-heterogeneous (`β_i(τ)`) slope specifications,
automatic quantile factor count selection via `ic_criteria_qfm`, split-panel jackknife
bias correction, Powell Schur-complement asymptotic covariance, and cross-sectional
unit-bootstrap inference.

References:
- Ando, T., & Bai, J. (2020). Quantile co-movement in financial markets: A panel quantile
  model with unobserved heterogeneity. *Journal of the American Statistical Association*,
  115(529), 266–279.
- Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica*, 77(4),
  1229–1279.
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models. *Econometrica*,
  89(2), 875–910.
"""

# =============================================================================
# Model Type
# =============================================================================

"""
    QuantileFactorPanelModel{T<:AbstractFloat} <: StatsAPI.RegressionModel

Fitted panel quantile regression model with interactive quantile factor effects
(Ando & Bai 2020):

```math
Q_{Y_{it}}(\\tau \\mid W_{it}, f_t(\\tau), \\lambda_i(\\tau)) =
W_{it}' \\beta_{(i)}(\\tau) + \\lambda_i(\\tau)' f_t(\\tau)
```

where:
- ``Y_{it}`` is the dependent variable for cross-sectional unit ``i = 1, \\dots, N`` at time ``t = 1, \\dots, T``
- ``W_{it} \\in \\mathbb{R}^p`` is the vector of observed covariates
- ``\\beta_{(i)}(\\tau)`` is the quantile slope vector (common ``\\beta(\\tau)`` when `slope == :homogeneous` or unit-specific ``\\beta_i(\\tau)`` when `slope == :heterogeneous`)
- ``\\lambda_i(\\tau) \\in \\mathbb{R}^r`` are unit factor loadings at quantile ``\\tau``
- ``f_t(\\tau) \\in \\mathbb{R}^r`` are latent common quantile factors at quantile ``\\tau``

# Fields
- `beta::Vector{T}` — slope vector (`p`); cross-sectional mean when `slope == :heterogeneous`
- `beta_i::Matrix{T}` — unit-specific slope matrix (`N × p`) when `slope == :heterogeneous`, `0 × 0` otherwise
- `vcov_mat::Matrix{T}` — estimated asymptotic or bootstrap covariance matrix (`p × p`)
- `stderr::Vector{T}` — standard errors of `beta` (`p`)
- `factors::Matrix{T}` — normalized quantile factors (`T_obs × r`) satisfying `F'F / T_obs = I_r`
- `loadings::Matrix{T}` — normalized unit loadings (`N × r`) with diagonal orthogonal columns
- `tau::T` — estimated quantile level strictly in `(0, 1)`
- `r::Int` — number of latent quantile factors
- `slope::Symbol` — `:homogeneous` or `:heterogeneous`
- `residuals::Matrix{T}` — idiosyncratic check-loss residuals (`N × T_obs`)
- `fitted::Matrix{T}` — fitted conditional quantile surfaces (`N × T_obs`)
- `objective::T` — mean check-function loss `(N * T_obs)⁻¹ ∑ ρ_τ(Y_{it} - fitted_{it})`
- `converged::Bool` — whether the alternating optimization converged within `maxiter`
- `iterations::Int` — number of outer alternating iterations executed
- `n_units::Int` — number of cross-sectional units `N`
- `n_periods::Int` — number of balanced time periods `T_obs`
- `varnames::Vector{String}` — regressor names (`length p`)
- `manifest::Union{ReproManifest,Nothing}` — reproducibility metadata
"""
struct QuantileFactorPanelModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    beta_i::Matrix{T}
    vcov_mat::Matrix{T}
    stderr::Vector{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    tau::T
    r::Int
    slope::Symbol
    residuals::Matrix{T}
    fitted::Matrix{T}
    objective::T
    converged::Bool
    iterations::Int
    n_units::Int
    n_periods::Int
    varnames::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

QuantileFactorPanelModel{T}(beta, beta_i, vcov_mat, stderr, factors, loadings,
                            tau, r, slope, residuals, fitted, objective,
                            converged, iterations, n_units, n_periods, varnames;
                            manifest=nothing) where {T<:AbstractFloat} =
    QuantileFactorPanelModel{T}(beta, beta_i, vcov_mat, stderr, factors, loadings,
                                tau, r, slope, residuals, fitted, objective,
                                converged, iterations, n_units, n_periods, varnames, manifest)

# =============================================================================
# Internal Panel Extraction & Linear Algebra Helpers
# =============================================================================

"""Extract balanced `(Y_mat::Matrix{T}, W_arr::Array{T,3}, varnames)` from `PanelData{T}`."""
function _qfm_pan_extract(pd::PanelData{T}, y::Symbol,
                          x::AbstractVector{Symbol}) where {T<:AbstractFloat}
    isempty(x) && throw(ArgumentError("Regressor list x must not be empty"))

    y_name = String(y)
    y_col = findfirst(==(y_name), pd.varnames)
    y_col === nothing && throw(ArgumentError(
        "Dependent variable :$y not found in PanelData; available: $(pd.varnames)"))

    x_names = [String(sym) for sym in x]
    x_cols = Vector{Int}(undef, length(x_names))
    for (j, nm) in enumerate(x_names)
        idx = findfirst(==(nm), pd.varnames)
        idx === nothing && throw(ArgumentError(
            "Regressor :$(x[j]) not found in PanelData; available: $(pd.varnames)"))
        x_cols[j] = idx
    end

    N = pd.n_groups
    N >= 2 || throw(ArgumentError("Need at least 2 cross-sectional units, got N=$N"))
    p = length(x_cols)

    # Collect per-group maps from finite time_id -> row index
    unit_maps = [Dict{Int,Int}() for _ in 1:N]
    all_cols = vcat(y_col, x_cols)
    @inbounds for row in 1:size(pd.data, 1)
        g = pd.group_id[row]
        ok = true
        for c in all_cols
            if !isfinite(pd.data[row, c])
                ok = false
                break
            end
        end
        ok || continue
        tid = pd.time_id[row]
        unit_maps[g][tid] = row
    end

    isempty(unit_maps[1]) && throw(ArgumentError(
        "Unit 1 has no finite observations for variables $(vcat([y_name], x_names))"))

    common_times = sort!(collect(keys(unit_maps[1])))
    for g in 2:N
        m_g = unit_maps[g]
        filter!(t -> haskey(m_g, t), common_times)
    end

    T_obs = length(common_times)
    T_obs >= 2 || throw(ArgumentError(
        "Need at least 2 common complete time periods across all $N units, got T=$T_obs"))

    Y_mat = Matrix{T}(undef, N, T_obs)
    W_arr = Array{T,3}(undef, N, T_obs, p)
    @inbounds for (t_pos, tid) in enumerate(common_times)
        for i in 1:N
            row = unit_maps[i][tid]
            Y_mat[i, t_pos] = pd.data[row, y_col]
            for j in 1:p
                W_arr[i, t_pos, j] = pd.data[row, x_cols[j]]
            end
        end
    end

    return Y_mat, W_arr, x_names
end

"""Check-function loss normalized by `N * T_obs`."""
function _qfm_pan_loss(residuals::Matrix{T}, tau::T) where {T<:AbstractFloat}
    N, T_obs = size(residuals)
    total = zero(T)
    @inbounds for t in 1:T_obs, i in 1:N
        u = residuals[i, t]
        total += u * (tau - (u < zero(T) ? one(T) : zero(T)))
    end
    return total / T(N * T_obs)
end

"""Quantile-SVD initialization on `R_tn` (`T_obs × N`), returning normalized `(F0, Λ0)`."""
function _qfm_pan_qsvd_init(R_tn::Matrix{T}, r::Int, tau::T) where {T<:AbstractFloat}
    T_obs, N = size(R_tn)
    B = Matrix{T}(undef, T_obs, N)
    @inbounds for i in 1:N
        col = @view R_tn[:, i]
        qi = T(Statistics.quantile(col, tau))
        for t in 1:T_obs
            B[t, i] = (col[t] <= qi ? one(T) : zero(T)) - tau
        end
    end
    sv = svd(B)
    sqrt_T = sqrt(T(T_obs))
    F0 = sqrt_T .* Matrix{T}(sv.U[:, 1:r])
    Λ0 = (Matrix{T}(sv.V[:, 1:r]) .* sv.S[1:r]') ./ sqrt_T
    F_n, L_n, _ = _qfm_normalize(F0, Λ0)
    return F_n, L_n
end

"""Subtract `W * β` (`slope == :homogeneous` or `:heterogeneous`) into `R_nt` (`N × T_obs`)."""
function _qfm_pan_partial_resid!(R_nt::Matrix{T}, Y_mat::Matrix{T}, W_arr::Array{T,3},
                                 beta::Vector{T}, beta_i::Matrix{T},
                                 slope::Symbol) where {T<:AbstractFloat}
    N, T_obs, p = size(W_arr)
    copyto!(R_nt, Y_mat)
    if slope === :homogeneous
        @inbounds for j in 1:p
            bj = beta[j]
            for t in 1:T_obs, i in 1:N
                R_nt[i, t] -= W_arr[i, t, j] * bj
            end
        end
    else
        @inbounds for j in 1:p, t in 1:T_obs, i in 1:N
            R_nt[i, t] -= W_arr[i, t, j] * beta_i[i, j]
        end
    end
    return R_nt
end

"""Pre-compute `(W_flat, XtX_inv_Xt)` or per-unit matrices for fast LS-IFE warm starts."""
function _qfm_pan_ls_warmstart(Y_mat::Matrix{T}, W_arr::Array{T,3}, r::Int,
                               slope::Symbol; ls_iters::Int=20) where {T<:AbstractFloat}
    N, T_obs, p = size(W_arr)
    W_flat = reshape(W_arr, N * T_obs, p)
    y_vec = vec(Y_mat)

    # Pre-factor Gram matrices once for fast alternating least squares
    beta = zeros(T, p)
    beta_i = slope === :heterogeneous ? zeros(T, N, p) : Matrix{T}(undef, 0, 0)

    local gram_hom_inv::Matrix{T}
    local gram_unit_inv::Vector{Matrix{T}}
    local W_units::Vector{Matrix{T}}

    if slope === :homogeneous
        gram_hom_inv = Matrix{T}(robust_inv(Symmetric(W_flat' * W_flat)))
        # Two-way demeaned initial OLS to strip additive FEs before SVD iterations
        Y_dot = copy(Y_mat)
        y_imean = Statistics.mean(Y_mat; dims=2)
        y_tmean = Statistics.mean(Y_mat; dims=1)
        y_grand = Statistics.mean(Y_mat)
        Y_dot .= Y_mat .- y_imean .- y_tmean .+ y_grand

        W_dot = copy(W_arr)
        for j in 1:p
            Wj = @view W_arr[:, :, j]
            W_dot[:, :, j] .= Wj .- Statistics.mean(Wj; dims=2) .- Statistics.mean(Wj; dims=1) .+
                              Statistics.mean(Wj)
        end
        Wd_flat = reshape(W_dot, N * T_obs, p)
        G_dot_inv = Matrix{T}(robust_inv(Symmetric(Wd_flat' * Wd_flat)))
        beta .= G_dot_inv * (Wd_flat' * vec(Y_dot))
    else
        W_units = [Matrix{T}(W_arr[i, :, :]) for i in 1:N]
        gram_unit_inv = [Matrix{T}(robust_inv(Symmetric(Wi' * Wi))) for Wi in W_units]
        for i in 1:N
            beta_i[i, :] .= gram_unit_inv[i] * (W_units[i]' * @view(Y_mat[i, :]))
        end
        beta .= vec(Statistics.mean(beta_i; dims=1))
    end

    R_nt = Matrix{T}(undef, N, T_obs)
    Y_adj = Matrix{T}(undef, N, T_obs)
    F_ws = zeros(T, T_obs, r)
    L_ws = zeros(T, N, r)
    sqrt_T = sqrt(T(T_obs))

    for _ in 1:ls_iters
        _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
        sv = svd(R_nt)
        F_ws .= sqrt_T .* sv.V[:, 1:r]
        L_ws .= (sv.U[:, 1:r] .* sv.S[1:r]') ./ sqrt_T

        mul!(Y_adj, L_ws, transpose(F_ws))
        @. Y_adj = Y_mat - Y_adj

        if slope === :homogeneous
            beta_new = gram_hom_inv * (W_flat' * vec(Y_adj))
            step = maximum(abs.(beta_new .- beta))
            beta .= beta_new
            step < T(1e-5) && break
        else
            max_step = zero(T)
            @inbounds for i in 1:N
                bi_new = gram_unit_inv[i] * (W_units[i]' * @view(Y_adj[i, :]))
                for j in 1:p
                    d = abs(bi_new[j] - beta_i[i, j])
                    d > max_step && (max_step = d)
                    beta_i[i, j] = bi_new[j]
                end
            end
            beta .= vec(Statistics.mean(beta_i; dims=1))
            max_step < T(1e-5) && break
        end
    end

    F_n, L_n, _ = _qfm_normalize(F_ws, L_ws)
    return beta, beta_i, F_n, L_n
end

"""
Fixed-rank alternating check-loss solver for `(β, F, Λ)`.

Alternates exact interior-point quantile updates of `β(τ)` (holding `Λ F'` as offset)
and QFM IQR coordinate updates of `(F(τ), Λ(τ))` on partial residuals `Y - W β`.
"""
function _qfm_pan_solve_fixed(Y_mat::Matrix{T}, W_arr::Array{T,3}, r::Int,
                              tau::T, slope::Symbol, maxiter::Int, tol::T;
                              beta_init::Union{Nothing,Vector{T}}=nothing,
                              beta_i_init::Union{Nothing,Matrix{T}}=nothing,
                              F_init::Union{Nothing,Matrix{T}}=nothing) where {T<:AbstractFloat}
    N, T_obs, p = size(W_arr)
    W_flat = reshape(W_arr, N * T_obs, p)
    R_nt = Matrix{T}(undef, N, T_obs)
    R_tn = Matrix{T}(undef, T_obs, N)
    Y_adj = Matrix{T}(undef, N, T_obs)
    LF_nt = Matrix{T}(undef, N, T_obs)
    resid = Matrix{T}(undef, N, T_obs)

    # Pre-allocate unit designs and workspaces
    beta = Vector{T}(undef, p)
    beta_i = slope === :heterogeneous ? Matrix{T}(undef, N, p) : Matrix{T}(undef, 0, 0)
    W_units = slope === :heterogeneous ? [Matrix{T}(W_arr[i, :, :]) for i in 1:N] : Matrix{T}[]
    ws_hom = slope === :homogeneous ? _QRegWorkspace{T}(N * T_obs, p) : nothing
    ws_het = slope === :heterogeneous ? _QRegWorkspace{T}(T_obs, p) : nothing

    # Warm start
    local F_cur::Matrix{T}
    local L_cur::Matrix{T}
    if beta_init !== nothing && F_init !== nothing
        beta .= beta_init
        if slope === :heterogeneous
            if beta_i_init !== nothing && size(beta_i_init) == (N, p)
                beta_i .= beta_i_init
            else
                for i in 1:N
                    beta_i[i, :] .= beta_init
                end
            end
        end
        F_cur = copy(F_init)
        _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
        transpose!(R_tn, R_nt)
        L_cur = Matrix{T}(undef, N, r)
        L_rt = Matrix{T}(undef, r, N)
        _qreg_batch!(L_rt, F_cur, R_tn, tau; threaded=false, tol=T(1e-7), maxit=50)
        transpose!(L_cur, L_rt)
        F_cur, L_cur, _ = _qfm_normalize(F_cur, L_cur)
    else
        beta_ws, beta_i_ws, F_ls, _ = _qfm_pan_ls_warmstart(Y_mat, W_arr, r, slope)
        beta .= beta_ws
        if slope === :heterogeneous
            beta_i .= beta_i_ws
        end
        _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
        transpose!(R_tn, R_nt)

        # Pick lower check-loss start between LS-IFE factors and quantile-SVD factors
        F_qsvd, _ = _qfm_pan_qsvd_init(R_tn, r, tau)
        L_rt = Matrix{T}(undef, r, N)

        _qreg_batch!(L_rt, F_ls, R_tn, tau; threaded=false, tol=T(1e-7), maxit=50)
        L_ls = Matrix{T}(transpose(L_rt))
        mul!(LF_nt, L_ls, transpose(F_ls))
        @. resid = R_nt - LF_nt
        loss_ls = _qfm_pan_loss(resid, tau)

        _qreg_batch!(L_rt, F_qsvd, R_tn, tau; threaded=false, tol=T(1e-7), maxit=50)
        L_qsvd = Matrix{T}(transpose(L_rt))
        mul!(LF_nt, L_qsvd, transpose(F_qsvd))
        @. resid = R_nt - LF_nt
        loss_qsvd = _qfm_pan_loss(resid, tau)

        if loss_qsvd < loss_ls
            F_cur, L_cur, _ = _qfm_normalize(F_qsvd, L_qsvd)
        else
            F_cur, L_cur, _ = _qfm_normalize(F_ls, L_ls)
        end
    end

    # Alternating workspaces for (F, Λ) updates without repeated inner allocations
    ws_L = _QRegWorkspace{T}(T_obs, r)
    ws_F = _QRegWorkspace{T}(N, r)
    L_tmp = Matrix{T}(undef, N, r)
    L_rt_loop = Matrix{T}(undef, r, N)
    F_rt_loop = Matrix{T}(undef, r, T_obs)
    beta_prev = similar(beta)

    mul!(LF_nt, L_cur, transpose(F_cur))
    _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
    @. resid = R_nt - LF_nt
    obj_prev = _qfm_pan_loss(resid, tau)

    converged = false
    iters = 0
    tol_ipm = min(T(1e-8), tol / T(10))

    for it in 1:maxiter
        iters = it
        beta_prev .= beta

        # 1. Update β(τ) holding (Λ, F) fixed as offset
        mul!(LF_nt, L_cur, transpose(F_cur))
        @. Y_adj = Y_mat - LF_nt
        if slope === :homogeneous
            _qreg_ipm!(beta, W_flat, vec(Y_adj), tau, ws_hom; tol=tol_ipm, maxit=60)
        else
            for i in 1:N
                _qreg_ipm!(@view(beta_i[i, :]), W_units[i], @view(Y_adj[i, :]),
                           tau, ws_het; tol=tol_ipm, maxit=60)
            end
            beta .= vec(Statistics.mean(beta_i; dims=1))
        end

        # 2. Update (F(τ), Λ(τ)) on partial residuals Y - W β using QFM IQR sweeps
        _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
        transpose!(R_tn, R_nt)

        n_inner = it <= 3 ? 3 : 2
        for _ in 1:n_inner
            for i in 1:N
                _qreg_ipm!(@view(L_rt_loop[:, i]), F_cur, @view(R_tn[:, i]), tau, ws_L;
                           tol=tol_ipm, maxit=50)
            end
            transpose!(L_tmp, L_rt_loop)
            for t in 1:T_obs
                _qreg_ipm!(@view(F_rt_loop[:, t]), L_tmp, @view(R_nt[:, t]), tau, ws_F;
                           tol=tol_ipm, maxit=50)
            end
            transpose!(F_cur, F_rt_loop)
            L_cur .= L_tmp
        end
        F_cur, L_cur, _ = _qfm_normalize(F_cur, L_cur)

        # Evaluate full objective and convergence checks
        mul!(LF_nt, L_cur, transpose(F_cur))
        @. resid = R_nt - LF_nt
        obj_cur = _qfm_pan_loss(resid, tau)
        rel_obj = abs(obj_prev - obj_cur) / max(obj_prev, one(T))
        param_diff = maximum(abs.(beta .- beta_prev))

        obj_prev = obj_cur
        if it >= 2 && rel_obj < tol && param_diff < tol
            converged = true
            break
        end
    end

    # Final polish of (F, Λ) & fitted values
    _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
    transpose!(R_tn, R_nt)
    mul!(LF_nt, L_cur, transpose(F_cur))
    @. resid = R_nt - LF_nt
    fitted = Y_mat .- resid
    obj = _qfm_pan_loss(resid, tau)

    return beta, beta_i, F_cur, L_cur, resid, fitted, obj, converged, iters
end

# =============================================================================
# Bias Correction & Covariance Estimation
# =============================================================================

"""
Split-panel jackknife bias correction (Dhaene & Jochmans 2015; Ando & Bai 2020)
removing `O(1/N) + O(1/T)` incidental-parameter bias for homogeneous slopes.
"""
function _qfm_pan_jackknife(Y_mat::Matrix{T}, W_arr::Array{T,3}, r::Int, tau::T,
                            beta_full::Vector{T}, F_full::Matrix{T},
                            maxiter::Int, tol::T) where {T<:AbstractFloat}
    N, T_obs, _ = size(W_arr)
    N_half = N ÷ 2
    T_half = T_obs ÷ 2
    sub_maxiter = max(25, maxiter ÷ 2)
    tol_sub = max(tol, T(1e-4))

    # Time splits: 1:T_half and (T_half+1):T_obs
    b_T1, _, _, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
        Y_mat[:, 1:T_half], W_arr[:, 1:T_half, :], r, tau, :homogeneous, sub_maxiter, tol_sub;
        beta_init=beta_full, F_init=F_full[1:T_half, :])
    b_T2, _, _, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
        Y_mat[:, (T_half + 1):T_obs], W_arr[:, (T_half + 1):T_obs, :], r, tau, :homogeneous,
        sub_maxiter, tol_sub;
        beta_init=beta_full, F_init=F_full[(T_half + 1):T_obs, :])

    # Cross-section splits: 1:N_half and (N_half+1):N
    b_N1, _, _, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
        Y_mat[1:N_half, :], W_arr[1:N_half, :, :], r, tau, :homogeneous, sub_maxiter, tol_sub;
        beta_init=beta_full, F_init=F_full)
    b_N2, _, _, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
        Y_mat[(N_half + 1):N, :], W_arr[(N_half + 1):N, :, :], r, tau, :homogeneous,
        sub_maxiter, tol_sub;
        beta_init=beta_full, F_init=F_full)

    beta_bc = T(3) .* beta_full .- T(0.5) .* (b_T1 .+ b_T2) .- T(0.5) .* (b_N1 .+ b_N2)
    return all(isfinite, beta_bc) ? beta_bc : copy(beta_full)
end

"""
Weighted Schur-complement projection of matrix `M` (`N × T_obs`) orthogonal to
`{δλ_i' F' + Λ δf_t'}` under Powell density weights `D` (`N × T_obs`).
"""
function _qfm_pan_project_ife!(M::Matrix{T}, D::Matrix{T}, Λ::Matrix{T},
                               F::Matrix{T}; sweeps::Int=8) where {T<:AbstractFloat}
    N, T_obs = size(M)
    r = size(F, 2)
    G = Matrix{T}(undef, r, r)
    rhs = Vector{T}(undef, r)
    coef_r = Vector{T}(undef, r)
    floor_g = T(1e-10)

    for _ in 1:sweeps
        # Project out span(F) across t for each unit i under weights D[i, :]
        for i in 1:N
            fill!(G, zero(T))
            fill!(rhs, zero(T))
            @inbounds for t in 1:T_obs
                w = D[i, t]
                m = M[i, t]
                wm = w * m
                for a in 1:r
                    fa = F[t, a]
                    rhs[a] += fa * wm
                    for b in a:r
                        G[a, b] += w * fa * F[t, b]
                    end
                end
            end
            @inbounds for a in 1:r, b in 1:(a - 1)
                G[a, b] = G[b, a]
            end
            @inbounds for a in 1:r
                G[a, a] += floor_g
            end
            mul!(coef_r, Matrix{T}(robust_inv(Symmetric(G))), rhs)
            @inbounds for t in 1:T_obs
                proj = zero(T)
                for a in 1:r
                    proj += F[t, a] * coef_r[a]
                end
                M[i, t] -= proj
            end
        end

        # Project out span(Λ) across i for each period t under weights D[:, t]
        for t in 1:T_obs
            fill!(G, zero(T))
            fill!(rhs, zero(T))
            @inbounds for i in 1:N
                w = D[i, t]
                m = M[i, t]
                wm = w * m
                for a in 1:r
                    la = Λ[i, a]
                    rhs[a] += la * wm
                    for b in a:r
                        G[a, b] += w * la * Λ[i, b]
                    end
                end
            end
            @inbounds for a in 1:r, b in 1:(a - 1)
                G[a, b] = G[b, a]
            end
            @inbounds for a in 1:r
                G[a, a] += floor_g
            end
            mul!(coef_r, Matrix{T}(robust_inv(Symmetric(G))), rhs)
            @inbounds for i in 1:N
                proj = zero(T)
                for a in 1:r
                    proj += Λ[i, a] * coef_r[a]
                end
                M[i, t] -= proj
            end
        end
    end
    return M
end

"""
Asymptotic covariance of `β̂(τ)` (Ando & Bai 2020 Theorems 1–2):
- `:homogeneous`: Powell Gaussian-kernel density-weighted Schur-complement sandwich
  `H_β⁻¹ Ω_β H_β⁻¹` after profiling out `(Λ(τ), F(τ))`.
- `:heterogeneous`: Pesaran–Smith / Swamy Mean-Group covariance
  `(N(N−1))⁻¹ ∑ (β̂_i − β̄)(β̂_i − β̄)'`.
"""
function _qfm_pan_vcov_asymp(W_arr::Array{T,3}, resid::Matrix{T},
                             Λ::Matrix{T}, F::Matrix{T},
                             beta_i::Matrix{T}, tau::T, slope::Symbol) where {T<:AbstractFloat}
    N, T_obs, p = size(W_arr)
    if slope === :heterogeneous
        mu = vec(Statistics.mean(beta_i; dims=1))
        V = zeros(T, p, p)
        @inbounds for i in 1:N
            for a in 1:p, b in 1:p
                V[a, b] += (beta_i[i, a] - mu[a]) * (beta_i[i, b] - mu[b])
            end
        end
        V ./= T(N * (N - 1))
        return Matrix{T}(Symmetric(V))
    end

    # Powell Gaussian kernel weights D[i, t] = ϕ(u_{it} / bw) / bw
    u_vec = vec(resid)
    h_bw = _hall_sheather_bandwidth(N * T_obs, tau, T(0.05))
    iqr_scale = (T(Statistics.quantile(u_vec, 0.75)) - T(Statistics.quantile(u_vec, 0.25))) /
                T(1.3489795003921634)
    scale_u = iqr_scale > T(1e-8) ? iqr_scale : Statistics.std(u_vec)
    bw = max(h_bw * scale_u, T(1e-6))
    inv_norm = one(T) / (bw * sqrt(T(2π)))

    D = Matrix{T}(undef, N, T_obs)
    @inbounds for idx in eachindex(resid)
        z = resid[idx] / bw
        D[idx] = exp(-T(0.5) * z * z) * inv_norm
    end

    # Profile out (Λ, F) from each covariate under Powell density weights D
    W_tilde = Matrix{T}(undef, N * T_obs, p)
    Mj = Matrix{T}(undef, N, T_obs)
    for j in 1:p
        copyto!(Mj, @view(W_arr[:, :, j]))
        _qfm_pan_project_ife!(Mj, D, Λ, F)
        W_tilde[:, j] .= vec(Mj)
    end

    d_vec = vec(D)
    H_beta = (W_tilde .* d_vec)' * W_tilde
    H_inv = Matrix{T}(robust_inv(Symmetric(H_beta)))
    Omega = tau * (one(T) - tau) .* (W_tilde' * W_tilde)
    V = H_inv * Omega * H_inv
    return Matrix{T}(Symmetric(V))
end

"""Cross-sectional unit bootstrap covariance resampling units `1:N` with replacement."""
function _qfm_pan_vcov_boot(Y_mat::Matrix{T}, W_arr::Array{T,3}, r::Int, tau::T,
                            slope::Symbol, beta_ref::Vector{T}, F_ref::Matrix{T},
                            n_boot::Int, maxiter::Int, tol::T,
                            threaded::Bool, rng::AbstractRNG) where {T<:AbstractFloat}
    N, T_obs, p = size(W_arr)
    boot_seeds = rand(rng, UInt64, n_boot)
    draws = Matrix{T}(undef, p, n_boot)
    sub_maxiter = max(25, maxiter ÷ 2)
    tol_sub = max(tol, T(1e-4))

    function _run_unit_boot!(b::Int, Y_b::Matrix{T}, W_b::Array{T,3}, idx::Vector{Int})
        local_rng = Random.Xoshiro(boot_seeds[b])
        @inbounds for i in 1:N
            idx[i] = rand(local_rng, 1:N)
            for t in 1:T_obs
                Y_b[i, t] = Y_mat[idx[i], t]
                for j in 1:p
                    W_b[i, t, j] = W_arr[idx[i], t, j]
                end
            end
        end
        b_hat, _, _, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
            Y_b, W_b, r, tau, slope, sub_maxiter, tol_sub;
            beta_init=beta_ref, F_init=F_ref)
        @inbounds for j in 1:p
            draws[j, b] = b_hat[j]
        end
        return nothing
    end

    if threaded && Threads.nthreads() > 1 && n_boot > 1
        nw = min(n_boot, Threads.nthreads())
        Y_pool = [Matrix{T}(undef, N, T_obs) for _ in 1:nw]
        W_pool = [Array{T,3}(undef, N, T_obs, p) for _ in 1:nw]
        idx_pool = [Vector{Int}(undef, N) for _ in 1:nw]
        base, rem = divrem(n_boot, nw)
        Threads.@threads for c in 1:nw
            s = (c - 1) * base + min(c - 1, rem) + 1
            len = base + (c <= rem ? 1 : 0)
            for b in s:(s + len - 1)
                _run_unit_boot!(b, Y_pool[c], W_pool[c], idx_pool[c])
            end
        end
    else
        Y_b = Matrix{T}(undef, N, T_obs)
        W_b = Array{T,3}(undef, N, T_obs, p)
        idx = Vector{Int}(undef, N)
        for b in 1:n_boot
            _run_unit_boot!(b, Y_b, W_b, idx)
        end
    end

    n_boot < 2 && return zeros(T, p, p)
    mu = vec(Statistics.mean(draws; dims=2))
    V = zeros(T, p, p)
    @inbounds for b in 1:n_boot, i in 1:p, j in 1:p
        V[i, j] += (draws[i, b] - mu[i]) * (draws[j, b] - mu[j])
    end
    V ./= T(n_boot - 1)
    return Matrix{T}(Symmetric(V))
end

# =============================================================================
# Core Estimation Driver & Public Entrypoints
# =============================================================================

function _qfm_pan_estimate(Y_mat::Matrix{T}, W_arr::Array{T,3},
                           r::Union{Int,Symbol}, tau::Real;
                           slope::Symbol=:homogeneous,
                           se::Symbol=:asymptotic,
                           n_boot::Int=199,
                           r_max::Int=6,
                           maxiter::Int=100,
                           tol::Real=1e-5,
                           varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                           bias_correct::Bool=true,
                           threaded::Bool=true,
                           rng::AbstractRNG=Random.default_rng(),
                           seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(Y_mat, "Y_mat")
    isempty(W_arr) && throw(ArgumentError("W_arr must not be empty"))
    N, T_obs = size(Y_mat)
    size(W_arr, 1) == N && size(W_arr, 2) == T_obs || throw(DimensionMismatch(
        "W_arr must have first two dimensions ($N, $T_obs), got ($(size(W_arr, 1)), $(size(W_arr, 2)))"))
    p = size(W_arr, 3)
    p >= 1 || throw(ArgumentError("Need at least 1 regressor, got p=$p"))
    _validate_data(reshape(W_arr, N * T_obs, p), "W_arr")
    N >= 2 || throw(ArgumentError("Need at least 2 cross-sectional units, got N=$N"))
    T_obs >= 2 || throw(ArgumentError("Need at least 2 time periods, got T_obs=$T_obs"))
    p >= 1 || throw(ArgumentError("Need at least 1 regressor, got p=$p"))

    tau_T = T(tau)
    (isfinite(tau) && zero(T) < tau_T < one(T)) ||
        throw(ArgumentError("tau must lie strictly in (0, 1), got $tau"))

    validate_option(slope, "slope", (:homogeneous, :heterogeneous))
    validate_option(se, "se", (:asymptotic, :bootstrap))
    n_boot >= 1 || throw(ArgumentError("n_boot must be >= 1, got $n_boot"))
    r_max >= 1 || throw(ArgumentError("r_max must be >= 1, got $r_max"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("tol must be > 0, got $tol"))

    vnames = varnames === nothing ? ["x$j" for j in 1:p] : String.(collect(varnames))
    length(vnames) == p || throw(ArgumentError(
        "varnames has $(length(vnames)) entries but W_arr has $p regressors"))

    rng_use = _resolve_repro_rng(rng, seed)
    tol_T = T(tol)
    max_feasible = min(N, T_obs) - 1
    max_feasible >= 1 || throw(ArgumentError(
        "Panel ($N × $T_obs) too small to identify at least 1 interactive factor"))

    local r_sel::Int
    local beta::Vector{T}
    local beta_i::Matrix{T}
    local F_hat::Matrix{T}
    local L_hat::Matrix{T}
    local resid::Matrix{T}
    local fitted::Matrix{T}
    local obj::T
    local converged::Bool
    local iters::Int

    if r === :auto
        r_cap = min(r_max, max_feasible)
        b_init, bi_init, F_init, _, _, _, _, _, _ = _qfm_pan_solve_fixed(
            Y_mat, W_arr, r_cap, tau_T, slope, maxiter, tol_T)
        R_nt = Matrix{T}(undef, N, T_obs)
        _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, b_init, bi_init, slope)
        R_tn = Matrix{T}(transpose(R_nt))
        ic = ic_criteria_qfm(R_tn, r_cap, tau_T;
                             warn_boundary=false, init=:quantile_svd,
                             se=:none, threaded=threaded, seed=seed)
        r_sel = min(r_cap, ic.r_rank >= 1 ? ic.r_rank : max(1, ic.r_IC2))
        beta, beta_i, F_hat, L_hat, resid, fitted, obj, converged, iters = _qfm_pan_solve_fixed(
            Y_mat, W_arr, r_sel, tau_T, slope, maxiter, tol_T;
            beta_init=b_init, beta_i_init=bi_init, F_init=F_init[:, 1:r_sel])
    elseif r isa Int
        1 <= r <= max_feasible || throw(ArgumentError(
            "Factor count r must satisfy 1 <= r <= $(max_feasible), got r=$r"))
        r_sel = r
        beta, beta_i, F_hat, L_hat, resid, fitted, obj, converged, iters = _qfm_pan_solve_fixed(
            Y_mat, W_arr, r_sel, tau_T, slope, maxiter, tol_T)
    else
        throw(ArgumentError("r must be a positive Int or :auto, got :$r"))
    end

    # Ando & Bai (2020) split-panel jackknife bias correction for homogeneous slopes
    if slope === :homogeneous && bias_correct && min(N ÷ 2, T_obs ÷ 2) >= max(r_sel + p + 2, 6)
        beta_bc = _qfm_pan_jackknife(Y_mat, W_arr, r_sel, tau_T, beta, F_hat, maxiter, tol_T)
        if maximum(abs.(beta_bc .- beta)) < T(0.5) * max(one(T), maximum(abs, beta))
            beta .= beta_bc
            R_nt = Matrix{T}(undef, N, T_obs)
            _qfm_pan_partial_resid!(R_nt, Y_mat, W_arr, beta, beta_i, slope)
            R_tn = Matrix{T}(transpose(R_nt))
            L_rt = Matrix{T}(undef, r_sel, N)
            _qreg_batch!(L_rt, F_hat, R_tn, tau_T; threaded=false, tol=T(1e-7), maxit=50)
            L_new = Matrix{T}(transpose(L_rt))
            F_hat, L_hat, _ = _qfm_normalize(F_hat, L_new)
            mul!(resid, L_hat, transpose(F_hat))
            resid .= R_nt .- resid
            fitted .= Y_mat .- resid
            obj = _qfm_pan_loss(resid, tau_T)
        end
    end

    vcov_mat = if se === :asymptotic
        _qfm_pan_vcov_asymp(W_arr, resid, L_hat, F_hat, beta_i, tau_T, slope)
    else
        _qfm_pan_vcov_boot(Y_mat, W_arr, r_sel, tau_T, slope, beta, F_hat,
                           n_boot, maxiter, tol_T, threaded, rng_use)
    end
    stderr = sqrt.(max.(diag(vcov_mat), zero(T)))

    model = QuantileFactorPanelModel{T}(
        beta, beta_i, vcov_mat, stderr, F_hat, L_hat,
        tau_T, r_sel, slope, resid, fitted, obj,
        converged, iters, N, T_obs, vnames, nothing,
    )
    return _with_manifest(model, capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "tau" => Float64(tau_T), "r" => r_sel, "slope" => String(slope),
            "se" => String(se), "n_units" => N, "n_periods" => T_obs,
        )))
end

"""
    estimate_xtqreg_factors(pd::PanelData, y::Symbol, x::AbstractVector{Symbol},
                            r::Union{Int,Symbol}=1, tau::Real=0.5;
                            slope=:homogeneous, se=:asymptotic, n_boot=199,
                            r_max=6, maxiter=100, tol=1e-5, varnames=nothing,
                            bias_correct=true, threaded=true,
                            rng=Random.default_rng(), seed=nothing) -> QuantileFactorPanelModel

    estimate_xtqreg_factors(Y_mat::AbstractMatrix{<:Real}, W_arr::AbstractArray{<:Real,3},
                            r::Union{Int,Symbol}=1, tau::Real=0.5;
                            slope=:homogeneous, se=:asymptotic, n_boot=199,
                            r_max=6, maxiter=100, tol=1e-5, varnames=nothing,
                            bias_correct=true, threaded=true,
                            rng=Random.default_rng(), seed=nothing) -> QuantileFactorPanelModel

Estimate panel quantile regression with interactive quantile factor effects (Ando & Bai 2020):

```math
Q_{Y_{it}}(\\tau \\mid W_{it}, f_t(\\tau), \\lambda_i(\\tau)) =
W_{it}' \\beta_{(i)}(\\tau) + \\lambda_i(\\tau)' f_t(\\tau)
```

where:
- ``Y_{it}`` is the outcome for unit ``i = 1, \\dots, N`` in period ``t = 1, \\dots, T``
- ``W_{it} \\in \\mathbb{R}^p`` is the vector of observed panel covariates
- ``\\beta_{(i)}(\\tau)`` is the quantile slope coefficient vector (common across units when `slope=:homogeneous`, or unit-specific when `slope=:heterogeneous`)
- ``\\lambda_i(\\tau)' f_t(\\tau)`` captures unobserved interactive factor heterogeneity at quantile ``\\tau``

The estimator alternates interior-point check-loss updates of ``\\beta(\\tau)`` (via `_qreg_ipm!`
with interactive factor offset) and quantile factor IQR updates of ``(\\Lambda(\\tau), F(\\tau))``
on partial residuals ``Y_{it} - W_{it}' \\beta(\\tau)``. When `r === :auto`, the rank ``r(\\tau)``
is selected on partial residuals via [`ic_criteria_qfm`](@ref) and the model is re-polished.

# Arguments
- `pd::PanelData` — balanced (or balanceable) panel container from [`xtset`](@ref)
- `y::Symbol` — name of the dependent variable in `pd`
- `x::AbstractVector{Symbol}` — names of the explanatory variables in `pd`
- `Y_mat::AbstractMatrix{<:Real}` — raw `(N × T_obs)` dependent variable matrix
- `W_arr::AbstractArray{<:Real,3}` — raw `(N × T_obs × p)` covariate array
- `r::Union{Int,Symbol} = 1` — number of latent quantile factors, or `:auto` for selection via [`ic_criteria_qfm`](@ref)
- `tau::Real = 0.5` — quantile level strictly in `(0, 1)`

# Keyword Arguments
- `slope::Symbol = :homogeneous` — `:homogeneous` (common `β(τ)` with split-panel jackknife bias correction) or `:heterogeneous` (unit-specific `β_i(τ)` and mean-group summary)
- `se::Symbol = :asymptotic` — `:asymptotic` (Powell Schur-complement sandwich for homogeneous slopes; Pesaran–Smith Swamy variance for heterogeneous slopes) or `:bootstrap` (cross-sectional unit bootstrap)
- `n_boot::Int = 199` — number of bootstrap replications when `se = :bootstrap`
- `r_max::Int = 6` — maximum factor count considered when `r = :auto`
- `maxiter::Int = 100` — maximum outer alternating optimization iterations
- `tol::Real = 1e-5` — relative objective and parameter convergence tolerance
- `varnames::Union{Nothing,AbstractVector{<:AbstractString}} = nothing` — optional regressor labels
- `bias_correct::Bool = true` — apply split-panel jackknife bias correction for `:homogeneous` slopes
- `threaded::Bool = true` — parallelize bootstrap replications across threads with pre-seeded per-slot RNGs
- `rng::AbstractRNG = Random.default_rng()` — RNG for bootstrap inference
- `seed::Union{Integer,Nothing} = nothing` — optional seed recorded in `manifest`

# Returns
A [`QuantileFactorPanelModel`](@ref) subtyping `StatsAPI.RegressionModel`.

# Examples
```julia
N, T_obs = 60, 80
f = randn(T_obs, 1)
lam = randn(N, 1)
W = 0.8 .* (lam * f') .+ randn(N, T_obs)
Y = 1.5 .* W .+ lam * f' .+ 0.5 .* randn(N, T_obs)
m = estimate_xtqreg_factors(Y, reshape(W, N, T_obs, 1), 1, 0.5)
report(m)
```

# References
- Ando, T., & Bai, J. (2020). Quantile co-movement in financial markets: A panel quantile
  model with unobserved heterogeneity. *Journal of the American Statistical Association*,
  115(529), 266–279.
- Chen, L., Dolado, J. J., & Gonzalo, J. (2021). Quantile factor models. *Econometrica*,
  89(2), 875–910.
"""
function estimate_xtqreg_factors(pd::PanelData{T}, y::Symbol, x::AbstractVector{Symbol},
                                 r::Union{Int,Symbol}=1, tau::Real=0.5;
                                 varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                                 kwargs...) where {T<:AbstractFloat}
    Y_mat, W_arr, x_names = _qfm_pan_extract(pd, y, x)
    vnames = varnames === nothing ? x_names : String.(collect(varnames))
    return _qfm_pan_estimate(Y_mat, W_arr, r, tau; varnames=vnames, kwargs...)
end

function estimate_xtqreg_factors(Y_mat::AbstractMatrix{<:Real},
                                 W_arr::AbstractArray{<:Real,3},
                                 r::Union{Int,Symbol}=1, tau::Real=0.5;
                                 kwargs...)
    FP = float(promote_type(eltype(Y_mat), eltype(W_arr)))
    return _qfm_pan_estimate(Matrix{FP}(Y_mat), Array{FP,3}(W_arr), r, tau; kwargs...)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::QuantileFactorPanelModel) = m.beta
StatsAPI.coefnames(m::QuantileFactorPanelModel) = m.varnames
StatsAPI.vcov(m::QuantileFactorPanelModel) = m.vcov_mat
StatsAPI.stderror(m::QuantileFactorPanelModel) = m.stderr
StatsAPI.residuals(m::QuantileFactorPanelModel) = m.residuals
StatsAPI.fitted(m::QuantileFactorPanelModel) = m.fitted
StatsAPI.nobs(m::QuantileFactorPanelModel) = m.n_units * m.n_periods
StatsAPI.islinear(::QuantileFactorPanelModel) = false

function StatsAPI.dof(m::QuantileFactorPanelModel)
    p = length(m.beta)
    beta_df = m.slope === :heterogeneous ? m.n_units * p : p
    return beta_df + m.r * (m.n_units + m.n_periods - m.r)
end

StatsAPI.dof_residual(m::QuantileFactorPanelModel) = max(StatsAPI.nobs(m) - StatsAPI.dof(m), 1)

function StatsAPI.confint(m::QuantileFactorPanelModel{T}; level::Real=0.95) where {T}
    (zero(T) < T(level) < one(T)) || throw(ArgumentError("level must be in (0, 1), got $level"))
    df_r = StatsAPI.dof_residual(m)
    crit = T(Distributions.quantile(Distributions.TDist(df_r), one(T) - (one(T) - T(level)) / 2))
    lo = m.beta .- crit .* m.stderr
    hi = m.beta .+ crit .* m.stderr
    return hcat(lo, hi)
end

StatsAPI.predict(m::QuantileFactorPanelModel) = m.fitted

function StatsAPI.predict(m::QuantileFactorPanelModel{T}, W_new::AbstractArray{<:Real,3}) where {T}
    N, T_obs, p = m.n_units, m.n_periods, length(m.beta)
    size(W_new) == (N, T_obs, p) || throw(DimensionMismatch(
        "W_new must have shape ($N, $T_obs, $p) matching training panel for interactive-effect predictions, got $(size(W_new))"))
    out = m.loadings * transpose(m.factors)
    W_T = Array{T,3}(W_new)
    if m.slope === :homogeneous
        @inbounds for j in 1:p
            bj = m.beta[j]
            for t in 1:T_obs, i in 1:N
                out[i, t] += W_T[i, t, j] * bj
            end
        end
    else
        @inbounds for j in 1:p, t in 1:T_obs, i in 1:N
            out[i, t] += W_T[i, t, j] * m.beta_i[i, j]
        end
    end
    return out
end

function StatsAPI.predict(m::QuantileFactorPanelModel{T}, W_new::AbstractMatrix{<:Real}) where {T}
    p = length(m.beta)
    size(W_new, 2) == p || throw(DimensionMismatch(
        "W_new must have $p columns matching model regressors, got $(size(W_new, 2))"))
    return Matrix{T}(W_new) * m.beta
end

# =============================================================================
# Tables.jl Source Interface (DataFrame(m) & CSV export)
# =============================================================================

function _qfm_pan_coef_nt(m::QuantileFactorPanelModel; level::Real=0.95)
    est = Float64.(m.beta)
    s = Float64.(m.stderr)
    stat = est ./ s
    dof_r = StatsAPI.dof_residual(m)
    a = 1 - level
    td = Distributions.TDist(dof_r)
    zc = Distributions.quantile(td, 1 - a / 2)
    pval = Float64[2 * (1 - Distributions.cdf(td, abs(z))) for z in stat]
    return (
        term=String.(m.varnames),
        estimate=est,
        std_error=s,
        stat=stat,
        p_value=pval,
        ci_lower=est .- zc .* s,
        ci_upper=est .+ zc .* s,
    )
end

Tables.istable(::Type{<:QuantileFactorPanelModel}) = true
Tables.columnaccess(::Type{<:QuantileFactorPanelModel}) = true
Tables.columns(m::QuantileFactorPanelModel) = _qfm_pan_coef_nt(m)
Tables.schema(m::QuantileFactorPanelModel) = Tables.schema(_qfm_pan_coef_nt(m))

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, m::QuantileFactorPanelModel{T}) where {T}
    spec = Any[
        "Quantile (τ)"     _fmt(m.tau; digits=3);
        "Factors (r)"      m.r;
        "Slope"            m.slope === :homogeneous ? "Homogeneous" : "Heterogeneous (Mean Group)";
        "Units (N)"        m.n_units;
        "Periods (T)"      m.n_periods;
        "Observations"     StatsAPI.nobs(m);
        "Check Loss"       _fmt(m.objective; digits=4);
        "Converged"        "$(m.converged ? "Yes" : "No") ($(m.iterations) iters)"
    ]
    _pretty_table(io, spec;
        title = "Panel Quantile Regression with Interactive Factors (τ=$(_fmt(m.tau; digits=3)), r=$(m.r))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    m.converged || @warn "Panel quantile factor optimization did not converge within $(m.iterations) iterations." maxlog = 1
    tbl_title = m.slope === :heterogeneous ? "Mean Group Slope Estimates" : "Slope Estimates"
    _coef_table(io, tbl_title, m.varnames, m.beta, m.stderr;
                dist=:t, dof_r=StatsAPI.dof_residual(m))
    _sig_legend(io)
    _manifest_footer(io, m.manifest)
    return nothing
end

"""
    report(m::QuantileFactorPanelModel; repro=false)

Print a Stata-style specification and coefficient summary table for a panel quantile
regression with interactive quantile factor effects ([`QuantileFactorPanelModel`](@ref)).
Pass `repro=true` to include the reproducibility manifest footer.
"""
report(m::QuantileFactorPanelModel; repro::Bool=false) = report(stdout, m; repro=repro)
report(io::IO, m::QuantileFactorPanelModel; repro::Bool=false) =
    (show(IOContext(io, :repro => repro), m); nothing)
