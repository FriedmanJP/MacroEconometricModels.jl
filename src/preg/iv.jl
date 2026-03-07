# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel instrumental-variables estimation: FE-IV, RE-IV (EC2SLS),
FD-IV, and Hausman-Taylor (1981).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Panel First-Stage F-statistic
# =============================================================================

"""
    _panel_first_stage_f(X_endog, Z) -> T

Minimum first-stage F across endogenous variables. For each endogenous column
x_j, regress x_j on Z and compute the partial F for joint significance.
"""
function _panel_first_stage_f(X_endog::Matrix{T}, Z::Matrix{T}) where {T<:AbstractFloat}
    n, m = size(Z)
    k_endog = size(X_endog, 2)
    k_endog == 0 && return T(Inf)

    ZtZinv = robust_inv(Z' * Z)
    f_min = T(Inf)

    for j in 1:k_endog
        x_j = X_endog[:, j]
        gamma = ZtZinv * (Z' * x_j)
        resid_j = x_j .- Z * gamma
        ssr_j = dot(resid_j, resid_j)

        x_bar = mean(x_j)
        tss_j = sum((xi - x_bar)^2 for xi in x_j)
        tss_j = max(tss_j, T(1e-300))

        df2 = n - m
        df2 > 0 || continue

        f_j = ((tss_j - ssr_j) / T(m)) / (ssr_j / T(df2))
        f_min = min(f_min, f_j)
    end

    f_min
end

# =============================================================================
# Panel Sargan-Hansen Test
# =============================================================================

"""
    _panel_sargan_test(resid, Z, k_regressors) -> (stat, pval) or (nothing, nothing)

Sargan-Hansen J-test for overidentifying restrictions in panel IV.
"""
function _panel_sargan_test(resid::Vector{T}, Z::Matrix{T},
                            k_regressors::Int) where {T<:AbstractFloat}
    n, m = size(Z)
    dof_sargan = m - k_regressors
    dof_sargan <= 0 && return (nothing, nothing)

    ZtZinv = robust_inv(Z' * Z)
    P_Z_e = Z * (ZtZinv * (Z' * resid))
    sigma2 = dot(resid, resid) / T(n)
    sigma2 = max(sigma2, T(1e-300))

    j_stat = dot(resid, P_Z_e) / sigma2
    j_pval = T(1 - cdf(Chisq(dof_sargan), max(j_stat, zero(T))))

    (j_stat, j_pval)
end

# =============================================================================
# 2SLS Core
# =============================================================================

"""
    _tsls(y, X, Z) -> (beta, XhXinv, resid, fitted, X_hat)

Core 2SLS: project X onto Z-space, then regress y on projected X.
Returns coefficients, bread matrix, residuals from original X, fitted, and X_hat.
"""
function _tsls(y::Vector{T}, X::Matrix{T}, Z::Matrix{T}) where {T<:AbstractFloat}
    ZtZinv = robust_inv(Z' * Z)
    X_hat = Z * (ZtZinv * (Z' * X))

    XhX = X_hat' * X
    XhXinv = robust_inv(XhX)
    beta = XhXinv * (X_hat' * y)

    fitted = X * beta
    resid = y .- fitted

    (beta, XhXinv, resid, fitted, X_hat)
end

# =============================================================================
# FE-IV Estimation
# =============================================================================

function _estimate_fe_iv(pd::PanelData{T}, y::Vector{T}, X_exog::Matrix{T},
                         X_endog::Matrix{T}, Z_excl::Matrix{T},
                         groups::Vector{Int}, time_ids::Vector{Int},
                         unique_groups::Vector{Int},
                         N::Int, n::Int, exog_names::Vector{Symbol},
                         endog_names::Vector{Symbol},
                         instrument_names::Vector{Symbol},
                         cov_type::Symbol) where {T}
    # Combine exogenous + endogenous into full X, and exogenous + excluded instruments into Z
    X_full = hcat(X_exog, X_endog)
    Z_full = hcat(X_exog, Z_excl)
    k = size(X_full, 2)

    # Within-demean everything
    y_dm, _ = _within_demean(y, groups, unique_groups)
    X_dm, _ = _within_demean_matrix(X_full, groups, unique_groups)
    Z_dm, _ = _within_demean_matrix(Z_full, groups, unique_groups)

    # Also demean just endogenous part for first-stage F
    X_endog_dm, _ = _within_demean_matrix(X_endog, groups, unique_groups)

    # 2SLS on demeaned data (no intercept)
    beta, XhXinv, resid_dm, _, X_hat = _tsls(y_dm, X_dm, Z_dm)

    # First-stage F on demeaned endogenous
    first_stage_f = _panel_first_stage_f(X_endog_dm, Z_dm)

    # Sargan test
    sargan_s, sargan_p = _panel_sargan_test(resid_dm, Z_dm, k)

    # Fitted and residuals on original scale
    # Recover group effects: alpha_i = y_bar_i - x_bar_i' * beta
    _, y_gm = _within_demean(y, groups, unique_groups)
    _, X_gm = _within_demean_matrix(X_full, groups, unique_groups)

    fitted_full = zeros(T, n)
    resid_full = zeros(T, n)
    for i in 1:n
        g = groups[i]
        g_idx = searchsortedfirst(unique_groups, g)
        alpha_i = y_gm[g] - dot(X_gm[g], beta)
        fitted_full[i] = dot(@view(X_full[i, :]), beta) + alpha_i
        resid_full[i] = y[i] - fitted_full[i]
    end

    # R-squared variants
    r2_within, r2_between, r2_overall = _panel_r2(y, fitted_full, groups, unique_groups)

    # Variance components
    dof_fe = max(n - N - k, 1)
    ssr_dm = dot(resid_dm, resid_dm)
    sigma_e2 = ssr_dm / T(dof_fe)
    sigma_e = sqrt(sigma_e2)

    # sigma_u from group effects
    group_effects = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        group_effects[j] = y_gm[g] - dot(X_gm[g], beta)
    end
    sigma_u2 = var(group_effects; corrected=true)
    sigma_u = sqrt(max(sigma_u2, zero(T)))
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # Covariance matrix — use projected X_hat for bread
    vcov_mat = _panel_vcov(X_hat, resid_dm, XhXinv, groups, time_ids, cov_type)

    # Variable names
    vn = vcat([String(v) for v in exog_names], [String(v) for v in endog_names])
    en = [String(v) for v in endog_names]
    in_ = [String(v) for v in instrument_names]

    PanelIVModel{T}(
        beta, vcov_mat, resid_full, fitted_full, y, X_full, Z_full,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        first_stage_f, sargan_s, sargan_p,
        vn, en, in_,
        :fe_iv, cov_type, n, N, pd
    )
end

# =============================================================================
# FD-IV Estimation
# =============================================================================

function _estimate_fd_iv(pd::PanelData{T}, y::Vector{T}, X_exog::Matrix{T},
                         X_endog::Matrix{T}, Z_excl::Matrix{T},
                         groups::Vector{Int}, time_ids::Vector{Int},
                         unique_groups::Vector{Int},
                         N::Int, n::Int, exog_names::Vector{Symbol},
                         endog_names::Vector{Symbol},
                         instrument_names::Vector{Symbol},
                         cov_type::Symbol) where {T}
    X_full = hcat(X_exog, X_endog)
    Z_full = hcat(X_exog, Z_excl)
    k_exog = size(X_exog, 2)
    k_endog = size(X_endog, 2)
    k = size(X_full, 2)
    m_excl = size(Z_excl, 2)

    # First-difference all variables within each group
    dy_list = Vector{T}()
    dX_list = Vector{Vector{T}}()
    dZ_list = Vector{Vector{T}}()
    dX_endog_list = Vector{Vector{T}}()
    dgroups = Int[]
    dtimes = Int[]

    for g in unique_groups
        idx = findall(==(g), groups)
        t_g = time_ids[idx]
        perm = sortperm(t_g)
        idx_sorted = idx[perm]
        t_sorted = t_g[perm]

        for j in 2:length(idx_sorted)
            if t_sorted[j] == t_sorted[j-1] + 1
                push!(dy_list, y[idx_sorted[j]] - y[idx_sorted[j-1]])
                dx = Vector{T}(undef, k)
                for p in 1:k
                    dx[p] = X_full[idx_sorted[j], p] - X_full[idx_sorted[j-1], p]
                end
                push!(dX_list, dx)

                dz = Vector{T}(undef, size(Z_full, 2))
                for p in 1:size(Z_full, 2)
                    dz[p] = Z_full[idx_sorted[j], p] - Z_full[idx_sorted[j-1], p]
                end
                push!(dZ_list, dz)

                dx_e = Vector{T}(undef, k_endog)
                for p in 1:k_endog
                    dx_e[p] = X_endog[idx_sorted[j], p] - X_endog[idx_sorted[j-1], p]
                end
                push!(dX_endog_list, dx_e)

                push!(dgroups, g)
                push!(dtimes, t_sorted[j])
            end
        end
    end

    n_fd = length(dy_list)
    n_fd > k + 1 || throw(ArgumentError("Too few observations after first differencing (n_fd=$n_fd, k=$k)"))

    dy = Vector{T}(dy_list)
    dX = zeros(T, n_fd, k)
    dZ = zeros(T, n_fd, size(Z_full, 2))
    dX_endog = zeros(T, n_fd, k_endog)
    for i in 1:n_fd
        dX[i, :] .= dX_list[i]
        dZ[i, :] .= dZ_list[i]
        dX_endog[i, :] .= dX_endog_list[i]
    end

    # Add intercept to differenced data
    dX_c = hcat(ones(T, n_fd), dX)
    dZ_c = hcat(ones(T, n_fd), dZ)
    k_full = k + 1

    # 2SLS
    beta_full, XhXinv, resid_fd, fitted_fd, X_hat = _tsls(dy, dX_c, dZ_c)
    beta = beta_full[2:end]

    # First-stage F on differenced endogenous
    first_stage_f = _panel_first_stage_f(dX_endog, dZ_c)

    # Sargan test
    sargan_s, sargan_p = _panel_sargan_test(resid_fd, dZ_c, k_full)

    # Vcov — extract slope portion
    vcov_full = _panel_vcov(X_hat, resid_fd, XhXinv, dgroups, dtimes, cov_type)
    vcov_mat = vcov_full[2:end, 2:end]

    # R-squared
    tss = dot(dy .- mean(dy), dy .- mean(dy))
    tss = max(tss, T(1e-300))
    ssr = dot(resid_fd, resid_fd)
    r2_within = one(T) - ssr / tss
    r2_between = zero(T)
    r2_overall = r2_within

    # Variance
    sigma_e2 = ssr / T(max(n_fd - k_full, 1))
    sigma_e = sqrt(sigma_e2)

    N_fd = length(unique(dgroups))
    vn = vcat([String(v) for v in exog_names], [String(v) for v in endog_names])
    en = [String(v) for v in endog_names]
    in_ = [String(v) for v in instrument_names]

    PanelIVModel{T}(
        beta, vcov_mat, resid_fd, fitted_fd, dy, dX, dZ,
        r2_within, r2_between, r2_overall,
        zero(T), sigma_e, zero(T),
        first_stage_f, sargan_s, sargan_p,
        vn, en, in_,
        :fd_iv, cov_type, n_fd, N_fd, pd
    )
end

# =============================================================================
# RE-IV / EC2SLS Estimation
# =============================================================================

function _estimate_re_iv(pd::PanelData{T}, y::Vector{T}, X_exog::Matrix{T},
                         X_endog::Matrix{T}, Z_excl::Matrix{T},
                         groups::Vector{Int}, time_ids::Vector{Int},
                         unique_groups::Vector{Int},
                         N::Int, n::Int, exog_names::Vector{Symbol},
                         endog_names::Vector{Symbol},
                         instrument_names::Vector{Symbol},
                         cov_type::Symbol) where {T}
    X_full = hcat(X_exog, X_endog)
    Z_full = hcat(X_exog, Z_excl)
    k = size(X_full, 2)

    # Step 1: Run FE-IV internally to get sigma_e^2
    y_dm, y_gm = _within_demean(y, groups, unique_groups)
    X_dm, X_gm = _within_demean_matrix(X_full, groups, unique_groups)
    Z_dm, _ = _within_demean_matrix(Z_full, groups, unique_groups)

    beta_fe, _, resid_fe, _, _ = _tsls(y_dm, X_dm, Z_dm)
    ssr_fe = dot(resid_fe, resid_fe)
    dof_fe = max(n - N - k, 1)
    sigma_e2 = ssr_fe / T(dof_fe)

    # Step 2: Between regression for sigma_between^2
    _, sigma2_between, _, _, _, _, _ = _between_regression(y, X_full, groups, unique_groups)

    # Step 3: Swamy-Arora variance components
    T_i = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        T_i[j] = T(count(==(g), groups))
    end
    T_bar_harmonic = T(N) / sum(one(T) ./ T_i)
    sigma_u2 = max(sigma2_between - sigma_e2 / T_bar_harmonic, zero(T))

    # Step 4: Compute theta per group
    theta_vec = zeros(T, N)
    for j in 1:N
        denom = T_i[j] * sigma_u2 + sigma_e2
        if denom > zero(T)
            theta_vec[j] = one(T) - sqrt(sigma_e2 / denom)
        end
    end

    # Step 5: Quasi-demean y, X, Z
    y_qd = copy(y)
    X_qd = copy(X_full)
    Z_qd = copy(Z_full)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        ym = y_gm[g]
        xm = X_gm[g]
        # Z group means
        z_idx_g = findall(==(g), groups)
        zm = vec(mean(@view(Z_full[z_idx_g, :]); dims=1))
        for i in idx
            y_qd[i] = y[i] - th * ym
            for p in 1:k
                X_qd[i, p] = X_full[i, p] - th * xm[p]
            end
            for p in 1:size(Z_full, 2)
                Z_qd[i, p] = Z_full[i, p] - th * zm[p]
            end
        end
    end

    # EC2SLS instruments: [Z_qd, Z_bar_i] (within + between)
    # Compute Z_bar expanded to all obs
    Z_bar = zeros(T, n, size(Z_full, 2))
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        zm = vec(mean(@view(Z_full[idx, :]); dims=1))
        for i in idx
            Z_bar[i, :] .= zm
        end
    end
    Z_ec2sls = hcat(Z_qd, Z_bar)

    # Add intercept
    X_qd_c = hcat(ones(T, n), X_qd)
    Z_ec2sls_c = hcat(ones(T, n), Z_ec2sls)
    k_full = k + 1

    # 2SLS
    beta_full, XhXinv, resid_qd, _, X_hat = _tsls(y_qd, X_qd_c, Z_ec2sls_c)
    beta = beta_full[2:end]

    # Original-scale residuals
    fitted_full = X_full * beta .+ beta_full[1]
    resid_full = y .- fitted_full

    # First-stage F on quasi-demeaned endogenous
    X_endog_qd = copy(X_endog)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        endog_gm = vec(mean(@view(X_endog[idx, :]); dims=1))
        for i in idx
            for p in 1:size(X_endog, 2)
                X_endog_qd[i, p] = X_endog[i, p] - th * endog_gm[p]
            end
        end
    end
    first_stage_f = _panel_first_stage_f(X_endog_qd, Z_ec2sls_c)

    # Sargan test
    sargan_s, sargan_p = _panel_sargan_test(resid_qd, Z_ec2sls_c, k_full)

    # R-squared
    r2_within, r2_between, r2_overall = _panel_r2(y, fitted_full, groups, unique_groups)

    # Variance components
    sigma_e = sqrt(sigma_e2)
    sigma_u = sqrt(sigma_u2)
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # Vcov — drop intercept
    vcov_full = _panel_vcov(X_hat, resid_qd, XhXinv, groups, time_ids, cov_type)
    vcov_mat = vcov_full[2:end, 2:end]

    vn = vcat([String(v) for v in exog_names], [String(v) for v in endog_names])
    en = [String(v) for v in endog_names]
    in_ = [String(v) for v in instrument_names]

    PanelIVModel{T}(
        beta, vcov_mat, resid_full, fitted_full, y, X_full, Z_full,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        first_stage_f, sargan_s, sargan_p,
        vn, en, in_,
        :re_iv, cov_type, n, N, pd
    )
end

# =============================================================================
# Hausman-Taylor Estimation
# =============================================================================

function _estimate_hausman_taylor(pd::PanelData{T}, y::Vector{T},
                                   X_tv_exog::Matrix{T}, X_tv_endog::Matrix{T},
                                   X_ti_exog::Matrix{T}, X_ti_endog::Matrix{T},
                                   groups::Vector{Int}, time_ids::Vector{Int},
                                   unique_groups::Vector{Int},
                                   N::Int, n::Int,
                                   tv_exog_names::Vector{Symbol},
                                   tv_endog_names::Vector{Symbol},
                                   ti_exog_names::Vector{Symbol},
                                   ti_endog_names::Vector{Symbol},
                                   cov_type::Symbol) where {T}
    k_tv_exog = size(X_tv_exog, 2)
    k_tv_endog = size(X_tv_endog, 2)
    k_ti_exog = size(X_ti_exog, 2)
    k_ti_endog = size(X_ti_endog, 2)

    # Full X: [tv_exog, tv_endog, ti_exog, ti_endog]
    X_tv = hcat(X_tv_exog, X_tv_endog)
    X_ti = hcat(X_ti_exog, X_ti_endog)
    X_full = hcat(X_tv, X_ti)
    k_tv = k_tv_exog + k_tv_endog
    k_ti = k_ti_exog + k_ti_endog
    k = k_tv + k_ti

    # ---- Step 1: Within estimation for sigma_e^2 ----
    y_dm, y_gm = _within_demean(y, groups, unique_groups)
    X_tv_dm, _ = _within_demean_matrix(X_tv, groups, unique_groups)

    XtX_w = X_tv_dm' * X_tv_dm
    XtXinv_w = robust_inv(XtX_w)
    delta_w = XtXinv_w * (X_tv_dm' * y_dm)
    resid_w = y_dm .- X_tv_dm * delta_w
    ssr_w = dot(resid_w, resid_w)
    dof_w = max(n - N - k_tv, 1)
    sigma_e2 = ssr_w / T(dof_w)

    # ---- Step 2: Estimate sigma_u^2 ----
    # From within residuals, compute sigma_u^2 via between variation
    # Use full residuals from within step: d_i = y_bar_i - x_tv_bar_i' * delta_w
    T_i = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        T_i[j] = T(count(==(g), groups))
    end
    T_bar_harmonic = T(N) / sum(one(T) ./ T_i)

    # Group mean residuals from the within estimator
    d_bar = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        d_bar[j] = mean(@view y[idx]) - dot(vec(mean(@view(X_tv[idx, :]); dims=1)), delta_w)
    end

    # d_bar_i = intercept + X_ti_i' * gamma + alpha_i + e_bar_i
    # Var(d_bar_i) = sigma_u^2 + sigma_e^2 / T_i
    # Use IV on group means: d_bar = const + X_ti * gamma, instruments = [ti_exog, tv_exog_bar]
    X_ti_bar = zeros(T, N, k_ti)
    X_tv_exog_bar = zeros(T, N, k_tv_exog)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        X_ti_bar[j, :] .= X_ti[idx[1], :]  # time-invariant, same for all obs in group
        X_tv_exog_bar[j, :] .= vec(mean(@view(X_tv_exog[idx, :]); dims=1))
    end

    X_b = hcat(ones(T, N), X_ti_bar)
    Z_b = hcat(ones(T, N), X_ti_bar[:, 1:k_ti_exog], X_tv_exog_bar)

    ZtZinv_b = robust_inv(Z_b' * Z_b)
    X_hat_b = Z_b * (ZtZinv_b * (Z_b' * X_b))
    XhX_b = X_hat_b' * X_b
    XhXinv_b = robust_inv(XhX_b)
    gamma_b = XhXinv_b * (X_hat_b' * d_bar)
    resid_b = d_bar .- X_b * gamma_b

    sigma2_d = dot(resid_b, resid_b) / T(max(N - k_ti - 1, 1))
    sigma_u2 = max(sigma2_d - sigma_e2 / T_bar_harmonic, zero(T))

    # ---- Step 3: GLS theta and quasi-demean ----
    theta_vec = zeros(T, N)
    for j in 1:N
        denom = T_i[j] * sigma_u2 + sigma_e2
        if denom > zero(T)
            theta_vec[j] = one(T) - sqrt(sigma_e2 / denom)
        end
    end

    # Compute group means for quasi-demeaning
    X_full_gm = Dict{Int,Vector{T}}()
    for g in unique_groups
        idx = findall(==(g), groups)
        X_full_gm[g] = vec(mean(@view(X_full[idx, :]); dims=1))
    end

    y_qd = copy(y)
    X_full_qd = copy(X_full)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        ym = y_gm[g]
        xm = X_full_gm[g]
        for i in idx
            y_qd[i] = y[i] - th * ym
            for p in 1:k
                X_full_qd[i, p] = X_full[i, p] - th * xm[p]
            end
        end
    end

    # ---- HT instruments ----
    # A = [within-deviations of ALL time-varying vars (exog + endog)]
    # B = [ti_exog (constant within group)]
    # C = [group means of tv_exog]
    # Full instruments: [A, B, C]
    X_tv_all_dm, _ = _within_demean_matrix(X_tv, groups, unique_groups)

    tv_exog_bar_expanded = zeros(T, n, k_tv_exog)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        xm = vec(mean(@view(X_tv_exog[idx, :]); dims=1))
        for i in idx
            tv_exog_bar_expanded[i, :] .= xm
        end
    end

    Z_ht = hcat(X_tv_all_dm, X_ti_exog, tv_exog_bar_expanded)

    # Quasi-demean Z
    Z_ht_gm = Dict{Int,Vector{T}}()
    for g in unique_groups
        idx = findall(==(g), groups)
        Z_ht_gm[g] = vec(mean(@view(Z_ht[idx, :]); dims=1))
    end

    Z_ht_qd = copy(Z_ht)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        zm = Z_ht_gm[g]
        for i in idx
            for p in 1:size(Z_ht, 2)
                Z_ht_qd[i, p] = Z_ht[i, p] - th * zm[p]
            end
        end
    end

    # Add intercept
    X_qd_c = hcat(ones(T, n), X_full_qd)
    Z_qd_c = hcat(ones(T, n), Z_ht_qd)
    k_full = k + 1

    # 2SLS
    beta_full, XhXinv, resid_qd, _, X_hat = _tsls(y_qd, X_qd_c, Z_qd_c)
    beta = beta_full[2:end]

    # Original-scale residuals
    fitted_full = X_full * beta .+ beta_full[1]
    resid_full = y .- fitted_full

    # First-stage F: endogenous = tv_endog + ti_endog
    X_endog_all = hcat(X_tv_endog, X_ti_endog)
    X_endog_qd = copy(X_endog_all)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        em = vec(mean(@view(X_endog_all[idx, :]); dims=1))
        for i in idx
            for p in 1:size(X_endog_all, 2)
                X_endog_qd[i, p] = X_endog_all[i, p] - th * em[p]
            end
        end
    end
    first_stage_f = _panel_first_stage_f(X_endog_qd, Z_qd_c)

    # Sargan test
    sargan_s, sargan_p = _panel_sargan_test(resid_qd, Z_qd_c, k_full)

    # R-squared
    r2_within, r2_between, r2_overall = _panel_r2(y, fitted_full, groups, unique_groups)

    # Variance components
    sigma_e = sqrt(sigma_e2)
    sigma_u = sqrt(max(sigma_u2, zero(T)))
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # Vcov — drop intercept
    vcov_full = _panel_vcov(X_hat, resid_qd, XhXinv, groups, time_ids, cov_type)
    vcov_mat = vcov_full[2:end, 2:end]

    vn = vcat([String(v) for v in tv_exog_names],
              [String(v) for v in tv_endog_names],
              [String(v) for v in ti_exog_names],
              [String(v) for v in ti_endog_names])
    en = vcat([String(v) for v in tv_endog_names],
              [String(v) for v in ti_endog_names])
    in_ = vcat([String(v) for v in tv_exog_names],
               [String(v) for v in ti_exog_names])

    PanelIVModel{T}(
        beta, vcov_mat, resid_full, fitted_full, y, X_full, Z_ht,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        first_stage_f, sargan_s, sargan_p,
        vn, en, in_,
        :hausman_taylor, cov_type, n, N, pd
    )
end

# =============================================================================
# Main API: estimate_xtiv
# =============================================================================

"""
    estimate_xtiv(pd::PanelData{T}, depvar::Symbol, exog::Vector{Symbol}, endog::Vector{Symbol};
                  instruments::Vector{Symbol}, model::Symbol=:fe, cov_type::Symbol=:cluster,
                  time_invariant_exog::Vector{Symbol}=Symbol[],
                  time_invariant_endog::Vector{Symbol}=Symbol[]) -> PanelIVModel{T}

Estimate a panel instrumental-variables regression model.

# Arguments
- `pd::PanelData{T}` — panel data container (created via `xtset`)
- `depvar::Symbol` — dependent variable name
- `exog::Vector{Symbol}` — exogenous regressors (time-varying for Hausman-Taylor)
- `endog::Vector{Symbol}` — endogenous regressors (time-varying for Hausman-Taylor)

# Keyword Arguments
- `instruments::Vector{Symbol}` — excluded instruments
- `model::Symbol` — `:fe` (FE-IV), `:re` (EC2SLS), `:fd` (FD-IV), `:hausman_taylor`
- `cov_type::Symbol` — `:ols`, `:cluster` (default)
- `time_invariant_exog::Vector{Symbol}` — time-invariant exogenous (Hausman-Taylor only)
- `time_invariant_endog::Vector{Symbol}` — time-invariant endogenous (Hausman-Taylor only)

# Returns
`PanelIVModel{T}` with estimated coefficients, first-stage F, and Sargan test.

# Methods
- **FE-IV** (`model=:fe`): Within-transform then 2SLS. No intercept.
- **RE-IV / EC2SLS** (`model=:re`): Quasi-demean then 2SLS with [Z̃, Z̄ᵢ] instruments.
- **FD-IV** (`model=:fd`): First-difference then 2SLS with intercept.
- **Hausman-Taylor** (`model=:hausman_taylor`): GLS-IV using within-deviations of
  time-varying exogenous as instruments for time-invariant endogenous.

# Examples
```julia
using DataFrames
df = DataFrame(id=repeat(1:50, inner=20), t=repeat(1:20, 50),
               x=randn(1000), z=randn(1000))
df.x_endog = 0.5 .* df.z .+ randn(1000)
df.y = repeat(randn(50), inner=20) .+ 1.5 .* df.x .+ 2.0 .* df.x_endog .+ randn(1000)
pd = xtset(df, :id, :t)
m = estimate_xtiv(pd, :y, [:x], [:x_endog]; instruments=[:z])
```

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer, ch. 7.
- Hausman, J. A. & Taylor, W. E. (1981). *Econometrica* 49(6), 1377-1398.
- Baltagi, B. H. (1981). *Econometrica* 49(4), 1049-1054. (EC2SLS)
"""
function estimate_xtiv(pd::PanelData{T}, depvar::Symbol, exog::Vector{Symbol},
                       endog::Vector{Symbol};
                       instruments::Vector{Symbol}=Symbol[],
                       model::Symbol=:fe,
                       cov_type::Symbol=:cluster,
                       time_invariant_exog::Vector{Symbol}=Symbol[],
                       time_invariant_endog::Vector{Symbol}=Symbol[]) where {T<:AbstractFloat}

    model in (:fe, :re, :fd, :hausman_taylor) ||
        throw(ArgumentError("model must be :fe, :re, :fd, or :hausman_taylor; got :$model"))
    cov_type in (:ols, :cluster, :twoway, :driscoll_kraay) ||
        throw(ArgumentError("cov_type must be :ols, :cluster, :twoway, or :driscoll_kraay; got :$cov_type"))

    # ---- Extract data columns ----
    function _get_col(v::Symbol)
        idx = findfirst(==(String(v)), pd.varnames)
        idx === nothing && throw(ArgumentError("Variable :$v not found in panel data. Available: $(pd.varnames)"))
        pd.data[:, idx]
    end

    y = _get_col(depvar)
    n = length(y)

    groups = pd.group_id
    time_ids = pd.time_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)

    if model == :hausman_taylor
        # Hausman-Taylor: exog/endog are time-varying, plus time-invariant kwargs
        X_tv_exog = length(exog) > 0 ? hcat([_get_col(v) for v in exog]...) : zeros(T, n, 0)
        X_tv_endog = length(endog) > 0 ? hcat([_get_col(v) for v in endog]...) : zeros(T, n, 0)
        X_ti_exog = length(time_invariant_exog) > 0 ? hcat([_get_col(v) for v in time_invariant_exog]...) : zeros(T, n, 0)
        X_ti_endog = length(time_invariant_endog) > 0 ? hcat([_get_col(v) for v in time_invariant_endog]...) : zeros(T, n, 0)

        return _estimate_hausman_taylor(pd, y, X_tv_exog, X_tv_endog,
                                        X_ti_exog, X_ti_endog,
                                        groups, time_ids, unique_groups,
                                        N, n, exog, endog,
                                        time_invariant_exog, time_invariant_endog,
                                        cov_type)
    end

    # Standard IV models: need instruments
    isempty(instruments) && throw(ArgumentError("instruments must be non-empty for IV estimation"))
    isempty(endog) && throw(ArgumentError("endog must be non-empty for IV estimation"))

    X_exog = length(exog) > 0 ? hcat([_get_col(v) for v in exog]...) : zeros(T, n, 0)
    X_endog = hcat([_get_col(v) for v in endog]...)
    Z_excl = hcat([_get_col(v) for v in instruments]...)

    # Order condition
    size(Z_excl, 2) + size(X_exog, 2) >= size(X_exog, 2) + size(X_endog, 2) ||
        throw(ArgumentError("Order condition violated: need at least as many instruments as endogenous variables"))

    if model == :fe
        return _estimate_fe_iv(pd, y, X_exog, X_endog, Z_excl,
                               groups, time_ids, unique_groups, N, n,
                               exog, endog, instruments, cov_type)
    elseif model == :fd
        return _estimate_fd_iv(pd, y, X_exog, X_endog, Z_excl,
                               groups, time_ids, unique_groups, N, n,
                               exog, endog, instruments, cov_type)
    elseif model == :re
        return _estimate_re_iv(pd, y, X_exog, X_endog, Z_excl,
                               groups, time_ids, unique_groups, N, n,
                               exog, endog, instruments, cov_type)
    end
end

