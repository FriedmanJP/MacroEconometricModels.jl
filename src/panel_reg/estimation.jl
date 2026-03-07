# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Fixed Effects (within) estimation for panel data.

Implements the within-transformation approach with entity and optionally
time fixed effects, entity-cluster standard errors, and R-squared variants.
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
# Fixed Effects (Within) Estimation
# =============================================================================

"""
    estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                   model=:fe, twoway=false, cov_type=:cluster, bandwidth=nothing) -> PanelRegModel{T}

Estimate a linear panel regression model.

Currently supports Fixed Effects (`:fe`) via within-transformation.

# Arguments
- `pd::PanelData{T}` — panel data container (created via `xtset`)
- `depvar::Symbol` — dependent variable name
- `indepvars::Vector{Symbol}` — independent variable names

# Keyword Arguments
- `model::Symbol` — `:fe` for Fixed Effects (default)
- `twoway::Bool` — include time fixed effects (default: `false`)
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
m = estimate_xtreg(pd, :y, [:x1, :x2])
```

# References
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
function estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                        model::Symbol=:fe, twoway::Bool=false,
                        cov_type::Symbol=:cluster,
                        bandwidth::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    model == :fe || throw(ArgumentError("Only model=:fe is currently supported; got :$model"))
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
    # F = (beta' V^{-1} beta) / k  using cluster-robust V
    # Use the chosen covariance for the F-test
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

