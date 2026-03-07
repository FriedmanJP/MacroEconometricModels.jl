# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel estimation: Fixed Effects (within), Random Effects (GLS),
First Differences, Between, and Correlated Random Effects (Mundlak).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Within-Transformation Helpers
# =============================================================================

"""
    _within_demean(v, groups, unique_groups) -> (demeaned, group_means)

Demean vector `v` by subtracting group means. Returns the demeaned vector
and a dictionary of group means.
"""
function _within_demean(v::AbstractVector{T}, groups::AbstractVector{Int},
                        unique_groups::AbstractVector{Int}) where {T}
    n = length(v)
    demeaned = similar(v)
    group_means = Dict{Int,T}()

    for g in unique_groups
        idx = findall(==(g), groups)
        gm = mean(@view v[idx])
        group_means[g] = gm
        for i in idx
            demeaned[i] = v[i] - gm
        end
    end

    demeaned, group_means
end

"""
    _within_demean_matrix(X, groups, unique_groups) -> (demeaned, group_means_matrix)

Demean each column of matrix `X` by group means.
"""
function _within_demean_matrix(X::AbstractMatrix{T}, groups::AbstractVector{Int},
                               unique_groups::AbstractVector{Int}) where {T}
    n, k = size(X)
    demeaned = similar(X)
    # group_means_matrix[g] = k-vector of means for group g
    group_means = Dict{Int,Vector{T}}()

    for g in unique_groups
        idx = findall(==(g), groups)
        gm = vec(mean(@view(X[idx, :]); dims=1))
        group_means[g] = gm
        for i in idx
            for j in 1:k
                demeaned[i, j] = X[i, j] - gm[j]
            end
        end
    end

    demeaned, group_means
end

"""
    _twoway_demean!(y_dm, X_dm, y, X, groups, time_ids, unique_groups, unique_times)

Apply two-way within-transformation: subtract entity means, time means,
and add back the grand mean.

y_dm_{it} = y_{it} - y_bar_i - y_bar_t + y_bar
"""
function _twoway_demean!(y_dm::Vector{T}, X_dm::Matrix{T},
                         y::Vector{T}, X::Matrix{T},
                         groups::Vector{Int}, time_ids::Vector{Int},
                         unique_groups::Vector{Int},
                         unique_times::Vector{Int}) where {T}
    n, k = size(X)
    grand_mean_y = mean(y)
    grand_mean_X = vec(mean(X; dims=1))

    # Entity means
    entity_mean_y = Dict{Int,T}()
    entity_mean_X = Dict{Int,Vector{T}}()
    for g in unique_groups
        idx = findall(==(g), groups)
        entity_mean_y[g] = mean(@view y[idx])
        entity_mean_X[g] = vec(mean(@view(X[idx, :]); dims=1))
    end

    # Time means
    time_mean_y = Dict{Int,T}()
    time_mean_X = Dict{Int,Vector{T}}()
    for t in unique_times
        idx = findall(==(t), time_ids)
        time_mean_y[t] = mean(@view y[idx])
        time_mean_X[t] = vec(mean(@view(X[idx, :]); dims=1))
    end

    # Apply two-way demeaning
    for i in 1:n
        g = groups[i]
        t = time_ids[i]
        y_dm[i] = y[i] - entity_mean_y[g] - time_mean_y[t] + grand_mean_y
        for j in 1:k
            X_dm[i, j] = X[i, j] - entity_mean_X[g][j] - time_mean_X[t][j] + grand_mean_X[j]
        end
    end
end

# =============================================================================
# Between Regression Helper
# =============================================================================

"""
    _between_regression(y, X, groups, unique_groups) -> (beta, sigma2, y_bar, X_bar)

Run OLS on group means: ȳᵢ on X̄ᵢ with intercept. Returns coefficients
(including intercept as first element), residual variance, and group-mean vectors.
"""
function _between_regression(y::Vector{T}, X::Matrix{T},
                              groups::Vector{Int},
                              unique_groups::Vector{Int}) where {T}
    N = length(unique_groups)
    k = size(X, 2)

    y_bar = zeros(T, N)
    X_bar = zeros(T, N, k)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        y_bar[j] = mean(@view y[idx])
        X_bar[j, :] .= vec(mean(@view(X[idx, :]); dims=1))
    end

    # Add intercept
    X_bar_c = hcat(ones(T, N), X_bar)
    XtX = X_bar_c' * X_bar_c
    XtXinv = robust_inv(XtX)
    beta_between = XtXinv * (X_bar_c' * y_bar)
    resid_between = y_bar .- X_bar_c * beta_between
    sigma2_between = dot(resid_between, resid_between) / T(max(N - k - 1, 1))

    beta_between, sigma2_between, y_bar, X_bar, X_bar_c, XtXinv, resid_between
end

# =============================================================================
# R-squared Variants Helper
# =============================================================================

"""
    _panel_r2(y, fitted, groups, unique_groups) -> (r2_within, r2_between, r2_overall)

Compute within, between, and overall R-squared from full residuals.
"""
function _panel_r2(y::Vector{T}, fitted::Vector{T},
                   groups::Vector{Int}, unique_groups::Vector{Int}) where {T}
    N = length(unique_groups)

    # Within R²: from demeaned data
    y_dm, _ = _within_demean(y, groups, unique_groups)
    f_dm, _ = _within_demean(fitted, groups, unique_groups)
    ssr_w = dot(y_dm .- f_dm, y_dm .- f_dm)
    tss_w = dot(y_dm, y_dm)
    tss_w = max(tss_w, T(1e-300))
    r2_within = one(T) - ssr_w / tss_w

    # Between R²: corr(ȳᵢ, ŷ̄ᵢ)²
    y_bar_g = zeros(T, N)
    yhat_bar_g = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        y_bar_g[j] = mean(@view y[idx])
        yhat_bar_g[j] = mean(@view fitted[idx])
    end
    r2_between = if std(y_bar_g) > T(1e-10) && std(yhat_bar_g) > T(1e-10)
        cor(y_bar_g, yhat_bar_g)^2
    else
        zero(T)
    end

    # Overall R²: corr(yᵢₜ, ŷᵢₜ)²
    r2_overall = if std(y) > T(1e-10) && std(fitted) > T(1e-10)
        cor(y, fitted)^2
    else
        zero(T)
    end

    r2_within, r2_between, r2_overall
end

# =============================================================================
# Fixed Effects (Within) Estimation
# =============================================================================

"""
    estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                   model=:fe, twoway=false, cov_type=:cluster, bandwidth=nothing) -> PanelRegModel{T}

Estimate a linear panel regression model.

# Arguments
- `pd::PanelData{T}` — panel data container (created via `xtset`)
- `depvar::Symbol` — dependent variable name
- `indepvars::Vector{Symbol}` — independent variable names

# Keyword Arguments
- `model::Symbol` — `:fe`, `:re`, `:fd`, `:between`, or `:cre` (default: `:fe`)
- `twoway::Bool` — include time fixed effects (FE only, default: `false`)
- `cov_type::Symbol` — covariance type: `:ols`, `:cluster` (default), `:twoway`, `:driscoll_kraay`
- `bandwidth::Union{Nothing,Int}` — Driscoll-Kraay bandwidth (default: auto)

# Returns
`PanelRegModel{T}` with estimated coefficients, variance components, and R-squared variants.

# Examples
```julia
using DataFrames
df = DataFrame(id=repeat(1:50, inner=20), t=repeat(1:20, 50),
               x1=randn(1000), x2=randn(1000))
df.y = repeat(randn(50), inner=20) .+ 1.5 .* df.x1 .- 0.8 .* df.x2 .+ 0.5 .* randn(1000)
pd = xtset(df, :id, :t)
m_fe = estimate_xtreg(pd, :y, [:x1, :x2])
m_re = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)
m_fd = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fd)
m_be = estimate_xtreg(pd, :y, [:x1, :x2]; model=:between)
m_cre = estimate_xtreg(pd, :y, [:x1, :x2]; model=:cre)
```

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
- Swamy, P. A. V. B. & Arora, S. S. (1972). *Econometrica* 40(2), 311-323.
- Mundlak, Y. (1978). *Econometrica* 46(1), 69-85.
"""
function estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                        model::Symbol=:fe, twoway::Bool=false,
                        cov_type::Symbol=:cluster,
                        bandwidth::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    model in (:fe, :re, :fd, :between, :cre) ||
        throw(ArgumentError("model must be :fe, :re, :fd, :between, or :cre; got :$model"))
    cov_type in (:ols, :cluster, :twoway, :driscoll_kraay) ||
        throw(ArgumentError("cov_type must be :ols, :cluster, :twoway, or :driscoll_kraay; got :$cov_type"))

    # ---- Extract data columns ----
    y_idx = findfirst(==(String(depvar)), pd.varnames)
    y_idx === nothing && throw(ArgumentError("Variable :$depvar not found in panel data. Available: $(pd.varnames)"))

    x_idxs = Int[]
    for v in indepvars
        idx = findfirst(==(String(v)), pd.varnames)
        idx === nothing && throw(ArgumentError("Variable :$v not found in panel data. Available: $(pd.varnames)"))
        push!(x_idxs, idx)
    end

    y = pd.data[:, y_idx]
    X = pd.data[:, x_idxs]
    n = length(y)
    k = length(indepvars)

    groups = pd.group_id
    time_ids = pd.time_id
    unique_groups = sort(unique(groups))
    unique_times = sort(unique(time_ids))
    N = length(unique_groups)
    n_times = length(unique_times)

    # Dispatch to specific estimator
    if model == :fe
        return _estimate_fe(pd, y, X, groups, time_ids, unique_groups, unique_times,
                            N, n_times, n, k, indepvars, twoway, cov_type, bandwidth)
    elseif model == :re
        return _estimate_re(pd, y, X, groups, time_ids, unique_groups, unique_times,
                            N, n_times, n, k, indepvars, cov_type, bandwidth)
    elseif model == :fd
        return _estimate_fd(pd, y, X, groups, time_ids, unique_groups,
                            N, n, k, indepvars, cov_type, bandwidth)
    elseif model == :between
        return _estimate_between(pd, y, X, groups, time_ids, unique_groups,
                                  N, n, k, indepvars, cov_type, bandwidth)
    elseif model == :cre
        return _estimate_cre(pd, y, X, groups, time_ids, unique_groups, unique_times,
                             N, n_times, n, k, indepvars, cov_type, bandwidth)
    end
end

# =============================================================================
# Fixed Effects (Within) Estimation
# =============================================================================

function _estimate_fe(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                      groups::Vector{Int}, time_ids::Vector{Int},
                      unique_groups::Vector{Int}, unique_times::Vector{Int},
                      N::Int, n_times::Int, n::Int, k::Int,
                      indepvars::Vector{Symbol}, twoway::Bool,
                      cov_type::Symbol, bandwidth) where {T}

    n > k + N || throw(ArgumentError("Need more observations than parameters (n=$n, k=$k, N=$N)"))

    # ---- Within-transformation ----
    if twoway
        y_dm = similar(y)
        X_dm = similar(X)
        _twoway_demean!(y_dm, X_dm, y, X, groups, time_ids,
                        unique_groups, unique_times)
    else
        y_dm, y_group_means = _within_demean(y, groups, unique_groups)
        X_dm, X_group_means = _within_demean_matrix(X, groups, unique_groups)
    end

    # ---- OLS on demeaned data (no intercept) ----
    XtX = X_dm' * X_dm
    XtXinv = robust_inv(XtX)
    beta = XtXinv * (X_dm' * y_dm)

    # ---- Residuals (from demeaned regression) ----
    resid_dm = y_dm .- X_dm * beta

    # ---- Recover entity fixed effects ----
    # alpha_i = y_bar_i - x_bar_i' * beta
    group_effects = zeros(T, N)
    if !twoway
        for (j, g) in enumerate(unique_groups)
            group_effects[j] = y_group_means[g] - dot(X_group_means[g], beta)
        end
    else
        # For two-way FE, recover from original data
        for (j, g) in enumerate(unique_groups)
            idx = findall(==(g), groups)
            ym = mean(@view y[idx])
            xm = vec(mean(@view(X[idx, :]); dims=1))
            group_effects[j] = ym - dot(xm, beta)
        end
    end

    # ---- Full residuals (original scale) ----
    fitted_full = similar(y)
    resid_full = similar(y)
    for i in 1:n
        g = groups[i]
        g_idx = searchsortedfirst(unique_groups, g)
        fitted_full[i] = dot(@view(X[i, :]), beta) + group_effects[g_idx]
        resid_full[i] = y[i] - fitted_full[i]
    end

    # ---- R-squared variants ----
    # Within R^2: 1 - SSR_within / TSS_within
    ssr_within = dot(resid_dm, resid_dm)
    tss_within = dot(y_dm, y_dm)
    tss_within = max(tss_within, T(1e-300))
    r2_within = one(T) - ssr_within / tss_within

    # Between R^2: corr(y_bar_i, x_bar_i' beta + alpha_i)^2
    y_bar_g = zeros(T, N)
    yhat_bar_g = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        y_bar_g[j] = mean(@view y[idx])
        yhat_bar_g[j] = mean(@view fitted_full[idx])
    end
    r2_between = if std(y_bar_g) > T(1e-10) && std(yhat_bar_g) > T(1e-10)
        cor(y_bar_g, yhat_bar_g)^2
    else
        zero(T)
    end

    # Overall R^2: corr(y_it, yhat_it)^2
    r2_overall = if std(y) > T(1e-10) && std(fitted_full) > T(1e-10)
        cor(y, fitted_full)^2
    else
        zero(T)
    end

    # ---- Variance components ----
    # sigma_e^2 = SSR / (NT - N - K)  [or NT - N - K - n_times + 1 for twoway]
    dof_fe = twoway ? n - N - k - n_times + 1 : n - N - k
    dof_fe = max(dof_fe, 1)
    sigma_e2 = ssr_within / T(dof_fe)
    sigma_e = sqrt(sigma_e2)

    # sigma_u^2 from between variation of group effects
    mean_alpha = mean(group_effects)
    sigma_u2 = var(group_effects; corrected=true)
    sigma_u = sqrt(max(sigma_u2, zero(T)))

    # rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # ---- F-test for joint significance ----
    vcov_mat = _panel_vcov(X_dm, resid_dm, XtXinv, groups, time_ids, cov_type;
                           bandwidth=bandwidth)

    f_stat = try
        T(dot(beta, robust_inv(vcov_mat) * beta) / k)
    catch
        zero(T)
    end
    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df2 = N - 1  # cluster-corrected df
        df2 > 0 ? T(1 - cdf(FDist(k, df2), f_stat)) : one(T)
    else
        one(T)
    end

    # ---- Log-likelihood, AIC, BIC ----
    sigma2_ml = ssr_within / T(n)
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(max(sigma2_ml, T(1e-300))) - T(n) / 2
    aic_val = -2 * loglik + 2 * T(k)
    bic_val = -2 * loglik + log(T(n)) * T(k)

    # ---- Average periods per group ----
    n_periods_avg = T(n) / T(N)

    # ---- Variable names ----
    vn = [String(v) for v in indepvars]

    PanelRegModel{T}(
        beta, vcov_mat, resid_full, fitted_full, y, X,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        nothing,  # theta (RE only)
        f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :fe, twoway, cov_type,
        n, N, n_periods_avg,
        group_effects, pd
    )
end

# =============================================================================
# Random Effects (GLS) Estimation — Swamy-Arora
# =============================================================================

function _estimate_re(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                      groups::Vector{Int}, time_ids::Vector{Int},
                      unique_groups::Vector{Int}, unique_times::Vector{Int},
                      N::Int, n_times::Int, n::Int, k::Int,
                      indepvars::Vector{Symbol},
                      cov_type::Symbol, bandwidth) where {T}

    # Step 1: Run FE internally to get sigma_e^2
    y_dm, y_group_means = _within_demean(y, groups, unique_groups)
    X_dm, X_group_means = _within_demean_matrix(X, groups, unique_groups)

    XtX_fe = X_dm' * X_dm
    XtXinv_fe = robust_inv(XtX_fe)
    beta_fe = XtXinv_fe * (X_dm' * y_dm)
    resid_fe = y_dm .- X_dm * beta_fe
    ssr_fe = dot(resid_fe, resid_fe)
    dof_fe = max(n - N - k, 1)
    sigma_e2 = ssr_fe / T(dof_fe)

    # Step 2: Run Between internally to get sigma_between^2
    _, sigma2_between, y_bar, X_bar, _, _, _ =
        _between_regression(y, X, groups, unique_groups)

    # Step 3: Swamy-Arora variance components
    # Compute T_i for each group and harmonic mean
    T_i = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        T_i[j] = T(count(==(g), groups))
    end
    T_bar_harmonic = T(N) / sum(one(T) ./ T_i)

    # sigma_u^2 = max(sigma2_between - sigma_e2 / T_bar, 0)
    sigma_u2 = max(sigma2_between - sigma_e2 / T_bar_harmonic, zero(T))

    # Step 4: Compute theta (quasi-demeaning parameter)
    # For unbalanced panels: theta_i per group
    # theta_i = 1 - sqrt(sigma_e2 / (T_i * sigma_u2 + sigma_e2))
    theta_vec = zeros(T, N)  # per group
    for j in 1:N
        denom = T_i[j] * sigma_u2 + sigma_e2
        if denom > zero(T)
            theta_vec[j] = one(T) - sqrt(sigma_e2 / denom)
        end
    end

    # Use average theta for display
    theta_avg = mean(theta_vec)

    # Step 5: Quasi-demean data
    # ỹᵢₜ = yᵢₜ - θᵢȳᵢ, X̃ᵢₜ = Xᵢₜ - θᵢX̄ᵢ
    y_qd = copy(y)
    X_qd = copy(X)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        ym = y_group_means[g]
        xm = X_group_means[g]
        for i in idx
            y_qd[i] = y[i] - th * ym
            for p in 1:k
                X_qd[i, p] = X[i, p] - th * xm[p]
            end
        end
    end

    # Step 6: OLS on quasi-demeaned data WITH intercept
    X_qd_c = hcat(ones(T, n), X_qd)
    k_full = k + 1  # includes intercept
    XtX = X_qd_c' * X_qd_c
    XtXinv = robust_inv(XtX)
    beta_full = XtXinv * (X_qd_c' * y_qd)

    # Separate intercept and slopes
    intercept = beta_full[1]
    beta = beta_full[2:end]

    # ---- Fitted values and residuals (on original data) ----
    fitted_full = X * beta .+ intercept
    resid_full = y .- fitted_full

    # Residuals for covariance estimation (quasi-demeaned)
    resid_qd = y_qd .- X_qd_c * beta_full

    # ---- R-squared variants ----
    r2_within, r2_between, r2_overall = _panel_r2(y, fitted_full, groups, unique_groups)

    # ---- Variance components for display ----
    sigma_e = sqrt(sigma_e2)
    sigma_u = sqrt(sigma_u2)
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # ---- Covariance matrix ----
    # Use the quasi-demeaned X (with intercept) for vcov
    vcov_full = _panel_vcov(X_qd_c, resid_qd, XtXinv, groups, time_ids, cov_type;
                            bandwidth=bandwidth)
    # Extract slope portion (drop intercept row/col)
    vcov_mat = vcov_full[2:end, 2:end]

    # ---- F-test for joint significance (slopes only) ----
    f_stat = try
        T(dot(beta, robust_inv(vcov_mat) * beta) / k)
    catch
        zero(T)
    end
    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df2 = N - 1
        df2 > 0 ? T(1 - cdf(FDist(k, df2), f_stat)) : one(T)
    else
        one(T)
    end

    # ---- Log-likelihood, AIC, BIC ----
    ssr = dot(resid_full, resid_full)
    sigma2_ml = ssr / T(n)
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(max(sigma2_ml, T(1e-300))) - T(n) / 2
    aic_val = -2 * loglik + 2 * T(k_full)
    bic_val = -2 * loglik + log(T(n)) * T(k_full)

    n_periods_avg = T(n) / T(N)
    vn = [String(v) for v in indepvars]

    PanelRegModel{T}(
        beta, vcov_mat, resid_full, fitted_full, y, X,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        theta_avg,
        f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :re, false, cov_type,
        n, N, n_periods_avg,
        nothing, pd  # no group_effects for RE
    )
end

# =============================================================================
# First Differences Estimation
# =============================================================================

function _estimate_fd(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                      groups::Vector{Int}, time_ids::Vector{Int},
                      unique_groups::Vector{Int},
                      N::Int, n::Int, k::Int,
                      indepvars::Vector{Symbol},
                      cov_type::Symbol, bandwidth) where {T}

    # Compute first differences within each group
    # Only difference consecutive time periods (handle gaps)
    dy_list = Vector{T}()
    dX_list = Vector{Vector{T}}()
    dgroups = Int[]
    dtimes = Int[]

    for g in unique_groups
        idx = findall(==(g), groups)
        # Sort by time within group
        t_g = time_ids[idx]
        perm = sortperm(t_g)
        idx_sorted = idx[perm]
        t_sorted = t_g[perm]

        for j in 2:length(idx_sorted)
            # Only difference if consecutive time periods
            if t_sorted[j] == t_sorted[j-1] + 1
                push!(dy_list, y[idx_sorted[j]] - y[idx_sorted[j-1]])
                dx = Vector{T}(undef, k)
                for p in 1:k
                    dx[p] = X[idx_sorted[j], p] - X[idx_sorted[j-1], p]
                end
                push!(dX_list, dx)
                push!(dgroups, g)
                push!(dtimes, t_sorted[j])
            end
        end
    end

    n_fd = length(dy_list)
    n_fd > k + 1 || throw(ArgumentError("Too few observations after first differencing (n_fd=$n_fd, k=$k)"))

    dy = Vector{T}(dy_list)
    dX = zeros(T, n_fd, k)
    for i in 1:n_fd
        dX[i, :] .= dX_list[i]
    end

    # OLS on (Δy, ΔX) WITH intercept (captures trend)
    dX_c = hcat(ones(T, n_fd), dX)
    k_full = k + 1
    XtX = dX_c' * dX_c
    XtXinv = robust_inv(XtX)
    beta_full = XtXinv * (dX_c' * dy)

    intercept = beta_full[1]
    beta = beta_full[2:end]

    resid_fd = dy .- dX_c * beta_full

    # Fitted and residuals on original scale (using first-differenced data)
    fitted_fd = dX_c * beta_full

    # ---- R-squared ----
    ssr = dot(resid_fd, resid_fd)
    tss = dot(dy .- mean(dy), dy .- mean(dy))
    tss = max(tss, T(1e-300))
    r2_within = one(T) - ssr / tss

    # Between and overall R² don't apply cleanly to FD; use correlations
    N_fd = length(unique(dgroups))
    unique_dgroups = sort(unique(dgroups))

    y_bar_g = zeros(T, N_fd)
    yhat_bar_g = zeros(T, N_fd)
    for (j, g) in enumerate(unique_dgroups)
        idx = findall(==(g), dgroups)
        y_bar_g[j] = mean(@view dy[idx])
        yhat_bar_g[j] = mean(@view fitted_fd[idx])
    end
    r2_between = if std(y_bar_g) > T(1e-10) && std(yhat_bar_g) > T(1e-10)
        cor(y_bar_g, yhat_bar_g)^2
    else
        zero(T)
    end
    r2_overall = if std(dy) > T(1e-10) && std(fitted_fd) > T(1e-10)
        cor(dy, fitted_fd)^2
    else
        zero(T)
    end

    # ---- Variance components ----
    sigma_e2 = ssr / T(max(n_fd - k_full, 1))
    sigma_e = sqrt(sigma_e2)
    sigma_u = zero(T)
    rho = zero(T)

    # ---- Covariance ----
    vcov_full = _panel_vcov(dX_c, resid_fd, XtXinv, dgroups, dtimes, cov_type;
                            bandwidth=bandwidth)
    vcov_mat = vcov_full[2:end, 2:end]

    # ---- F-test ----
    f_stat = try
        T(dot(beta, robust_inv(vcov_mat) * beta) / k)
    catch
        zero(T)
    end
    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df2 = N_fd - 1
        df2 > 0 ? T(1 - cdf(FDist(k, df2), f_stat)) : one(T)
    else
        one(T)
    end

    # ---- Log-likelihood, AIC, BIC ----
    sigma2_ml = ssr / T(n_fd)
    loglik = -T(n_fd) / 2 * log(T(2) * T(pi)) - T(n_fd) / 2 * log(max(sigma2_ml, T(1e-300))) - T(n_fd) / 2
    aic_val = -2 * loglik + 2 * T(k_full)
    bic_val = -2 * loglik + log(T(n_fd)) * T(k_full)

    n_periods_avg = T(n_fd) / T(N_fd)
    vn = [String(v) for v in indepvars]

    PanelRegModel{T}(
        beta, vcov_mat, resid_fd, fitted_fd, dy, dX,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        nothing,
        f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :fd, false, cov_type,
        n_fd, N_fd, n_periods_avg,
        nothing, pd
    )
end

# =============================================================================
# Between Estimator
# =============================================================================

function _estimate_between(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                            groups::Vector{Int}, time_ids::Vector{Int},
                            unique_groups::Vector{Int},
                            N::Int, n::Int, k::Int,
                            indepvars::Vector{Symbol},
                            cov_type::Symbol, bandwidth) where {T}

    beta_full, sigma2_between, y_bar, X_bar, X_bar_c, XtXinv, resid_between =
        _between_regression(y, X, groups, unique_groups)

    intercept = beta_full[1]
    beta = beta_full[2:end]

    # Fitted values and residuals (group-level)
    fitted_between = X_bar_c * beta_full
    resid_be = y_bar .- fitted_between

    # ---- R-squared ----
    tss = dot(y_bar .- mean(y_bar), y_bar .- mean(y_bar))
    tss = max(tss, T(1e-300))
    ssr = dot(resid_be, resid_be)
    r2_between = one(T) - ssr / tss

    # Within and overall: use correlations on group means
    r2_within = zero(T)  # not meaningful for between
    r2_overall = r2_between  # between = overall at group level

    # ---- Variance components ----
    k_full = k + 1
    sigma_e = sqrt(sigma2_between)
    sigma_u = zero(T)
    rho = zero(T)

    # ---- Covariance: OLS on group means ----
    # For between estimator, use OLS vcov (no clustering at group level — each obs is a group)
    sigma2_ols = ssr / T(max(N - k_full, 1))
    vcov_full = sigma2_ols .* XtXinv
    vcov_mat = vcov_full[2:end, 2:end]

    # ---- F-test ----
    f_stat = try
        T(dot(beta, robust_inv(vcov_mat) * beta) / k)
    catch
        zero(T)
    end
    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df2 = max(N - k_full, 1)
        T(1 - cdf(FDist(k, df2), f_stat))
    else
        one(T)
    end

    # ---- Log-likelihood, AIC, BIC ----
    sigma2_ml = ssr / T(N)
    loglik = -T(N) / 2 * log(T(2) * T(pi)) - T(N) / 2 * log(max(sigma2_ml, T(1e-300))) - T(N) / 2
    aic_val = -2 * loglik + 2 * T(k_full)
    bic_val = -2 * loglik + log(T(N)) * T(k_full)

    n_periods_avg = T(n) / T(N)
    vn = [String(v) for v in indepvars]

    PanelRegModel{T}(
        beta, vcov_mat, resid_be, fitted_between, y_bar, X_bar,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        nothing,
        f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn, :between, false, cov_type,
        N, N, n_periods_avg,  # n_obs = N for between
        nothing, pd
    )
end

# =============================================================================
# Correlated Random Effects (Mundlak) Estimation
# =============================================================================

function _estimate_cre(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                       groups::Vector{Int}, time_ids::Vector{Int},
                       unique_groups::Vector{Int}, unique_times::Vector{Int},
                       N::Int, n_times::Int, n::Int, k::Int,
                       indepvars::Vector{Symbol},
                       cov_type::Symbol, bandwidth) where {T}

    # Step 1: Compute group means X̄ᵢ for all regressors
    X_group_means = Dict{Int,Vector{T}}()
    y_group_means = Dict{Int,T}()
    for g in unique_groups
        idx = findall(==(g), groups)
        X_group_means[g] = vec(mean(@view(X[idx, :]); dims=1))
        y_group_means[g] = mean(@view y[idx])
    end

    # Step 2: Augment X with group means (repeated for each obs)
    X_aug = zeros(T, n, 2k)
    for i in 1:n
        g = groups[i]
        for p in 1:k
            X_aug[i, p] = X[i, p]
            X_aug[i, k+p] = X_group_means[g][p]
        end
    end

    # Step 3: Run RE estimation on augmented model
    # We need to do the full RE procedure with the augmented X

    # FE on augmented model to get sigma_e^2
    # Note: group means are collinear in within-transformation (they are constant within group)
    # So FE uses only the original X columns — sigma_e^2 comes from FE on original X
    y_dm, _ = _within_demean(y, groups, unique_groups)
    X_dm, _ = _within_demean_matrix(X, groups, unique_groups)

    XtX_fe = X_dm' * X_dm
    XtXinv_fe = robust_inv(XtX_fe)
    beta_fe = XtXinv_fe * (X_dm' * y_dm)
    resid_fe = y_dm .- X_dm * beta_fe
    ssr_fe = dot(resid_fe, resid_fe)
    dof_fe = max(n - N - k, 1)
    sigma_e2 = ssr_fe / T(dof_fe)

    # Between regression (on original X, not augmented — for variance component)
    _, sigma2_between, _, _, _, _, _ = _between_regression(y, X, groups, unique_groups)

    # Swamy-Arora variance components
    T_i = zeros(T, N)
    for (j, g) in enumerate(unique_groups)
        T_i[j] = T(count(==(g), groups))
    end
    T_bar_harmonic = T(N) / sum(one(T) ./ T_i)
    sigma_u2 = max(sigma2_between - sigma_e2 / T_bar_harmonic, zero(T))

    # Theta per group
    theta_vec = zeros(T, N)
    for j in 1:N
        denom = T_i[j] * sigma_u2 + sigma_e2
        if denom > zero(T)
            theta_vec[j] = one(T) - sqrt(sigma_e2 / denom)
        end
    end
    theta_avg = mean(theta_vec)

    # Quasi-demean the augmented data
    y_qd = copy(y)
    X_aug_qd = copy(X_aug)
    for (j, g) in enumerate(unique_groups)
        idx = findall(==(g), groups)
        th = theta_vec[j]
        ym = y_group_means[g]
        # Group means for augmented X: original vars + mean vars
        xm_orig = X_group_means[g]
        # The mean vars are constant within group, so their group mean = themselves
        xm_mean = X_group_means[g]
        for i in idx
            y_qd[i] = y[i] - th * ym
            for p in 1:k
                X_aug_qd[i, p] = X_aug[i, p] - th * xm_orig[p]
                X_aug_qd[i, k+p] = X_aug[i, k+p] - th * xm_mean[p]
            end
        end
    end

    # OLS on quasi-demeaned augmented data WITH intercept
    X_aug_qd_c = hcat(ones(T, n), X_aug_qd)
    k_aug = 2k + 1  # intercept + k slopes + k means
    XtX = X_aug_qd_c' * X_aug_qd_c
    XtXinv = robust_inv(XtX)
    beta_full = XtXinv * (X_aug_qd_c' * y_qd)

    # Separate intercept + original slopes + mean slopes
    intercept = beta_full[1]
    beta_all = beta_full[2:end]  # 2k coefficients

    # Fitted values on original data
    fitted_full = X_aug * beta_all .+ intercept
    resid_full = y .- fitted_full

    # Residuals for covariance (quasi-demeaned)
    resid_qd = y_qd .- X_aug_qd_c * beta_full

    # ---- R-squared variants ----
    r2_within, r2_between, r2_overall = _panel_r2(y, fitted_full, groups, unique_groups)

    # ---- Variance components ----
    sigma_e = sqrt(sigma_e2)
    sigma_u = sqrt(sigma_u2)
    total_var = sigma_u2 + sigma_e2
    rho = total_var > zero(T) ? sigma_u2 / total_var : zero(T)

    # ---- Covariance ----
    vcov_full = _panel_vcov(X_aug_qd_c, resid_qd, XtXinv, groups, time_ids, cov_type;
                            bandwidth=bandwidth)
    vcov_mat = vcov_full[2:end, 2:end]  # drop intercept

    # ---- F-test (on all slopes) ----
    f_stat = try
        T(dot(beta_all, robust_inv(vcov_mat) * beta_all) / (2k))
    catch
        zero(T)
    end
    f_pval = if f_stat > zero(T) && isfinite(f_stat)
        df2 = N - 1
        df2 > 0 ? T(1 - cdf(FDist(2k, df2), f_stat)) : one(T)
    else
        one(T)
    end

    # ---- Log-likelihood, AIC, BIC ----
    ssr = dot(resid_full, resid_full)
    sigma2_ml = ssr / T(n)
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(max(sigma2_ml, T(1e-300))) - T(n) / 2
    aic_val = -2 * loglik + 2 * T(k_aug)
    bic_val = -2 * loglik + log(T(n)) * T(k_aug)

    n_periods_avg = T(n) / T(N)

    # Variable names: original + mean variables
    vn = [String(v) for v in indepvars]
    vn_mean = [String(v) * "_mean" for v in indepvars]
    vn_all = vcat(vn, vn_mean)

    PanelRegModel{T}(
        beta_all, vcov_mat, resid_full, fitted_full, y, X_aug,
        r2_within, r2_between, r2_overall,
        sigma_u, sigma_e, rho,
        theta_avg,
        f_stat, f_pval,
        loglik, aic_val, bic_val,
        vn_all, :cre, false, cov_type,
        n, N, n_periods_avg,
        nothing, pd
    )
end
