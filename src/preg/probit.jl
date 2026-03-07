# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel probit regression: pooled, random effects (Gauss-Hermite quadrature),
and correlated random effects (Mundlak).
No FE probit (incidental parameters problem -- no conditioning trick for probit).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# estimate_xtprobit — Main Entry Point
# =============================================================================

"""
    estimate_xtprobit(pd, depvar, indepvars; model=:pooled, cov_type=:cluster,
                      maxiter=200, tol=1e-8, n_quadrature=12) -> PanelProbitModel{T}

Estimate a panel probit regression model.

# Arguments
- `pd::PanelData{T}` -- panel data container (created via `xtset`)
- `depvar::Symbol` -- binary dependent variable (0/1)
- `indepvars::Vector{Symbol}` -- independent variable names

# Keyword Arguments
- `model::Symbol` -- `:pooled`, `:re`, or `:cre` (default: `:pooled`).
  No `:fe` — the incidental parameters problem has no conditioning trick for probit.
- `cov_type::Symbol` -- covariance type: `:ols`, `:cluster` (default)
- `maxiter::Int` -- maximum iterations (default 200)
- `tol` -- convergence tolerance (default 1e-8)
- `n_quadrature::Int` -- Gauss-Hermite quadrature points for RE/CRE (default 12)

# Returns
`PanelProbitModel{T}` with estimated coefficients and diagnostics.

# Examples
```julia
using DataFrames, Distributions
df = DataFrame(id=repeat(1:50, inner=10), t=repeat(1:10, 50),
               x1=randn(500), x2=randn(500))
p = cdf.(Normal(), 0.8 .* df.x1 .- 0.5 .* df.x2)
df.y = Float64.(rand(500) .< p)
pd = xtset(df, :id, :t)
m = estimate_xtprobit(pd, :y, [:x1, :x2])
m_re = estimate_xtprobit(pd, :y, [:x1, :x2]; model=:re)
```

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
function estimate_xtprobit(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                           model::Symbol=:pooled, cov_type::Symbol=:cluster,
                           maxiter::Int=200, tol::Real=1e-8,
                           n_quadrature::Int=12) where {T<:AbstractFloat}
    model in (:pooled, :re, :cre) ||
        throw(ArgumentError("model must be :pooled, :re, or :cre; got :$model (no FE probit — incidental parameters problem)"))

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

    all(yi -> yi == zero(T) || yi == one(T), y) ||
        throw(ArgumentError("Dependent variable must be binary (0/1)"))

    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)

    if model == :pooled
        return _xtprobit_pooled(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol))
    elseif model == :re
        return _xtprobit_re(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol), n_quadrature)
    elseif model == :cre
        return _xtprobit_cre(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol), n_quadrature)
    end
end

# =============================================================================
# Pooled Probit with Cluster SEs
# =============================================================================

function _xtprobit_pooled(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                          groups::Vector{Int}, unique_groups::Vector{Int},
                          N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                          cov_type::Symbol, maxiter::Int, tol::T) where {T}
    X_c = hcat(ones(T, n), X)
    k_full = k + 1
    vn = vcat(["_cons"], [String(v) for v in indepvars])

    beta, mu, w, loglik_val, converged, iterations = _irls_probit(y, X_c; maxiter=maxiter, tol=tol)

    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    W = Diagonal(w)
    XtWX = X_c' * W * X_c
    info_inv = robust_inv(XtWX)

    if cov_type == :cluster
        score_resid = y .- mu
        vcov_mat = _panel_cluster_vcov(X_c, score_resid, info_inv, groups)
    else
        vcov_mat = info_inv
    end

    pseudo_r2 = one(T) - loglik_val / loglik_null
    aic_val = -2 * loglik_val + 2 * T(k_full)
    bic_val = -2 * loglik_val + log(T(n)) * T(k_full)

    PanelProbitModel{T}(
        beta, vcov_mat, y, X_c, mu,
        loglik_val, loglik_null, pseudo_r2, aic_val, bic_val,
        nothing, nothing,
        vn, :pooled, cov_type, converged, iterations, n, N, pd
    )
end

# =============================================================================
# RE Probit — Gauss-Hermite Quadrature
# =============================================================================

function _re_probit_loglik(theta::Vector{T}, y::Vector{T}, X_c::Matrix{T},
                           groups::Vector{Int}, unique_groups::Vector{Int},
                           group_obs::Dict{Int,Vector{Int}},
                           nodes::Vector{Float64}, weights::Vector{Float64}) where {T}
    k = size(X_c, 2)
    beta = theta[1:k]
    log_sigma_u = theta[k + 1]
    sigma_u = exp(log_sigma_u)

    d = Normal(zero(T), one(T))
    n_quad = length(nodes)
    loglik = zero(T)
    grad = zeros(T, k + 1)

    for g in unique_groups
        idx = group_obs[g]
        X_g = X_c[idx, :]
        y_g = y[idx]
        T_g = length(idx)
        eta_g = X_g * beta

        Li = zero(T)
        dLi_dbeta = zeros(T, k)
        dLi_dsig = zero(T)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))

            log_prod = zero(T)
            for j in 1:T_g
                mu_j = clamp(T(cdf(d, eta_g[j] + alpha)), T(1e-10), one(T) - T(1e-10))
                log_prod += y_g[j] * log(mu_j) + (one(T) - y_g[j]) * log(one(T) - mu_j)
            end

            contrib = wq * exp(log_prod)
            Li += contrib

            if contrib > T(1e-300)
                score_j = zeros(T, T_g)
                for j in 1:T_g
                    mu_j = clamp(T(cdf(d, eta_g[j] + alpha)), T(1e-10), one(T) - T(1e-10))
                    phi_j = max(T(pdf(d, eta_g[j] + alpha)), T(1e-10))
                    denom = mu_j * (one(T) - mu_j)
                    denom = max(denom, T(1e-10))
                    score_j[j] = (y_g[j] - mu_j) * phi_j / denom
                end

                dLi_dbeta .+= contrib .* (X_g' * score_j)
                dalpha_dlogsig = sqrt(T(2)) * T(nodes[q]) * sigma_u
                dLi_dsig += contrib * sum(score_j) * dalpha_dlogsig
            end
        end

        Li = max(Li, T(1e-300))
        loglik += log(Li)
        grad[1:k] .+= dLi_dbeta ./ Li
        grad[k + 1] += dLi_dsig / Li
    end

    return loglik, grad
end

function _xtprobit_re(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                      groups::Vector{Int}, unique_groups::Vector{Int},
                      N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                      cov_type::Symbol, maxiter::Int, tol::T,
                      n_quadrature::Int) where {T}
    X_c = hcat(ones(T, n), X)
    k_full = k + 1
    vn = vcat(["_cons"], [String(v) for v in indepvars])

    group_obs = Dict{Int,Vector{Int}}()
    for g in unique_groups
        group_obs[g] = findall(==(g), groups)
    end

    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature)

    beta_init, _, _, _, _, _ = _irls_probit(y, X_c; maxiter=50, tol=T(1e-6))
    theta = vcat(beta_init, zero(T))

    loglik_old = T(-Inf)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        loglik, grad = _re_probit_loglik(theta, y, X_c, groups, unique_groups,
                                          group_obs, nodes, weights)

        if abs(loglik - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik
            break
        end
        loglik_old = loglik

        step_size = T(0.5)
        for _ in 1:20
            theta_new = theta .+ step_size .* grad
            ll_new, _ = _re_probit_loglik(theta_new, y, X_c, groups, unique_groups,
                                           group_obs, nodes, weights)
            if ll_new > loglik
                theta = theta_new
                break
            end
            step_size *= T(0.5)
        end
    end

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    loglik_final = loglik_old

    # Numerical Hessian
    n_params = k_full + 1
    hess = zeros(T, n_params, n_params)
    eps_h = T(1e-4)
    for i in 1:n_params
        theta_p = copy(theta)
        theta_m = copy(theta)
        theta_p[i] += eps_h
        theta_m[i] -= eps_h
        _, g_p = _re_probit_loglik(theta_p, y, X_c, groups, unique_groups,
                                    group_obs, nodes, weights)
        _, g_m = _re_probit_loglik(theta_m, y, X_c, groups, unique_groups,
                                    group_obs, nodes, weights)
        hess[:, i] .= (g_p .- g_m) ./ (2 * eps_h)
    end
    hess = (hess .+ hess') ./ 2

    vcov_full = try
        robust_inv(-hess)
    catch
        zeros(T, n_params, n_params)
    end
    vcov_mat = vcov_full[1:k_full, 1:k_full]

    # Fitted probabilities
    d = Normal(zero(T), one(T))
    fitted = zeros(T, n)
    for g in unique_groups
        idx = group_obs[g]
        X_g = X_c[idx, :]
        eta_g = X_g * beta
        for (j, i) in enumerate(idx)
            p = zero(T)
            for q in 1:n_quadrature
                alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
                wq = T(weights[q]) / sqrt(T(pi))
                p += wq * T(cdf(d, eta_g[j] + alpha))
            end
            fitted[i] = p
        end
    end

    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    pseudo_r2 = one(T) - loglik_final / loglik_null
    aic_val = -2 * loglik_final + 2 * T(n_params)
    bic_val = -2 * loglik_final + log(T(n)) * T(n_params)

    # rho = sigma_u^2 / (sigma_u^2 + 1) for probit (variance of logistic = pi^2/3, normal = 1)
    rho = sigma_u^2 / (sigma_u^2 + one(T))

    PanelProbitModel{T}(
        beta, vcov_mat, y, X_c, fitted,
        loglik_final, loglik_null, pseudo_r2, aic_val, bic_val,
        sigma_u, rho,
        vn, :re, cov_type, converged, iterations, n, N, pd
    )
end

# =============================================================================
# CRE Probit — Mundlak Augmentation + RE
# =============================================================================

function _xtprobit_cre(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                       groups::Vector{Int}, unique_groups::Vector{Int},
                       N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                       cov_type::Symbol, maxiter::Int, tol::T,
                       n_quadrature::Int) where {T}
    # Compute group means
    X_means = zeros(T, n, k)
    for g in unique_groups
        idx = findall(==(g), groups)
        gm = vec(mean(@view(X[idx, :]); dims=1))
        for i in idx
            X_means[i, :] .= gm
        end
    end

    X_aug = hcat(X, X_means)
    vn_orig = [String(v) for v in indepvars]
    vn_mean = [String(v) * "_mean" for v in indepvars]

    X_c = hcat(ones(T, n), X_aug)
    k_full = 2k + 1
    vn = vcat(["_cons"], vn_orig, vn_mean)

    group_obs = Dict{Int,Vector{Int}}()
    for g in unique_groups
        group_obs[g] = findall(==(g), groups)
    end

    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature)

    beta_init, _, _, _, _, _ = _irls_probit(y, X_c; maxiter=50, tol=T(1e-6))
    theta = vcat(beta_init, zero(T))

    loglik_old = T(-Inf)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        loglik, grad = _re_probit_loglik(theta, y, X_c, groups, unique_groups,
                                          group_obs, nodes, weights)

        if abs(loglik - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik
            break
        end
        loglik_old = loglik

        step_size = T(0.5)
        for _ in 1:20
            theta_new = theta .+ step_size .* grad
            ll_new, _ = _re_probit_loglik(theta_new, y, X_c, groups, unique_groups,
                                           group_obs, nodes, weights)
            if ll_new > loglik
                theta = theta_new
                break
            end
            step_size *= T(0.5)
        end
    end

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    loglik_final = loglik_old

    n_params = k_full + 1
    hess = zeros(T, n_params, n_params)
    eps_h = T(1e-4)
    for i in 1:n_params
        theta_p = copy(theta)
        theta_m = copy(theta)
        theta_p[i] += eps_h
        theta_m[i] -= eps_h
        _, g_p = _re_probit_loglik(theta_p, y, X_c, groups, unique_groups,
                                    group_obs, nodes, weights)
        _, g_m = _re_probit_loglik(theta_m, y, X_c, groups, unique_groups,
                                    group_obs, nodes, weights)
        hess[:, i] .= (g_p .- g_m) ./ (2 * eps_h)
    end
    hess = (hess .+ hess') ./ 2

    vcov_full = try
        robust_inv(-hess)
    catch
        zeros(T, n_params, n_params)
    end
    vcov_mat = vcov_full[1:k_full, 1:k_full]

    # Fitted probabilities
    d = Normal(zero(T), one(T))
    fitted = zeros(T, n)
    n_quad = length(nodes)
    for g in unique_groups
        idx = group_obs[g]
        X_g = X_c[idx, :]
        eta_g = X_g * beta
        for (j, i) in enumerate(idx)
            p = zero(T)
            for q in 1:n_quad
                alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
                wq = T(weights[q]) / sqrt(T(pi))
                p += wq * T(cdf(d, eta_g[j] + alpha))
            end
            fitted[i] = p
        end
    end

    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    pseudo_r2 = one(T) - loglik_final / loglik_null
    aic_val = -2 * loglik_final + 2 * T(n_params)
    bic_val = -2 * loglik_final + log(T(n)) * T(n_params)

    rho = sigma_u^2 / (sigma_u^2 + one(T))

    PanelProbitModel{T}(
        beta, vcov_mat, y, X_c, fitted,
        loglik_final, loglik_null, pseudo_r2, aic_val, bic_val,
        sigma_u, rho,
        vn, :cre, cov_type, converged, iterations, n, N, pd
    )
end
