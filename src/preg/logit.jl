# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel logistic regression: pooled, FE conditional (Chamberlain 1980),
random effects (Gauss-Hermite quadrature), and correlated random effects (Mundlak).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# estimate_xtlogit — Main Entry Point
# =============================================================================

"""
    estimate_xtlogit(pd, depvar, indepvars; model=:pooled, cov_type=:cluster,
                     maxiter=200, tol=1e-8, n_quadrature=12) -> PanelLogitModel{T}

Estimate a panel logistic regression model.

# Arguments
- `pd::PanelData{T}` -- panel data container (created via `xtset`)
- `depvar::Symbol` -- binary dependent variable (0/1)
- `indepvars::Vector{Symbol}` -- independent variable names

# Keyword Arguments
- `model::Symbol` -- `:pooled`, `:fe`, `:re`, or `:cre` (default: `:pooled`)
- `cov_type::Symbol` -- covariance type: `:ols`, `:cluster` (default)
- `maxiter::Int` -- maximum iterations (default 200)
- `tol` -- convergence tolerance (default 1e-8)
- `n_quadrature::Int` -- Gauss-Hermite quadrature points for RE/CRE (default 12)

# Returns
`PanelLogitModel{T}` with estimated coefficients and diagnostics.

# Examples
```julia
using DataFrames
df = DataFrame(id=repeat(1:50, inner=10), t=repeat(1:10, 50),
               x1=randn(500), x2=randn(500))
df.y = Float64.(rand(500) .< 1 ./ (1 .+ exp.(-1.0 .* df.x1 .+ 0.5 .* df.x2)))
pd = xtset(df, :id, :t)
m = estimate_xtlogit(pd, :y, [:x1, :x2])
m_fe = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:fe)
m_re = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:re)
```

# References
- Chamberlain, G. (1980). *Review of Economic Studies* 47(1), 225-238.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
function estimate_xtlogit(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
                          model::Symbol=:pooled, cov_type::Symbol=:cluster,
                          maxiter::Int=200, tol::Real=1e-8,
                          n_quadrature::Int=12) where {T<:AbstractFloat}
    model in (:pooled, :fe, :re, :cre) ||
        throw(ArgumentError("model must be :pooled, :fe, :re, or :cre; got :$model"))

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

    # Validate binary
    all(yi -> yi == zero(T) || yi == one(T), y) ||
        throw(ArgumentError("Dependent variable must be binary (0/1)"))

    groups = pd.group_id
    unique_groups = sort(unique(groups))
    N = length(unique_groups)

    if model == :pooled
        return _xtlogit_pooled(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol))
    elseif model == :fe
        return _xtlogit_fe(pd, y, X, groups, unique_groups, N, n, k, indepvars, maxiter, T(tol))
    elseif model == :re
        return _xtlogit_re(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol), n_quadrature)
    elseif model == :cre
        return _xtlogit_cre(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type, maxiter, T(tol), n_quadrature)
    end
end

# =============================================================================
# Pooled Logit with Cluster SEs
# =============================================================================

function _xtlogit_pooled(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                         groups::Vector{Int}, unique_groups::Vector{Int},
                         N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                         cov_type::Symbol, maxiter::Int, tol::T) where {T}
    # Add intercept
    X_c = hcat(ones(T, n), X)
    k_full = k + 1
    vn = vcat(["_cons"], [String(v) for v in indepvars])

    # IRLS logit
    beta, mu, w, loglik_val, converged, iterations = _irls_logit(y, X_c; maxiter=maxiter, tol=tol)

    # Null log-likelihood
    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    # Covariance: information matrix + cluster adjustment
    W = Diagonal(w)
    XtWX = X_c' * W * X_c
    info_inv = robust_inv(XtWX)

    if cov_type == :cluster
        # Sandwich with entity-clustered meat
        score_resid = y .- mu
        vcov_mat = _panel_cluster_vcov(X_c, score_resid, info_inv, groups)
    else
        vcov_mat = info_inv
    end

    pseudo_r2 = one(T) - loglik_val / loglik_null
    aic_val = -2 * loglik_val + 2 * T(k_full)
    bic_val = -2 * loglik_val + log(T(n)) * T(k_full)

    PanelLogitModel{T}(
        beta, vcov_mat, y, X_c, mu,
        loglik_val, loglik_null, pseudo_r2, aic_val, bic_val,
        nothing, nothing,  # sigma_u, rho
        vn, :pooled, cov_type, converged, iterations, n, N, pd
    )
end

# =============================================================================
# FE Conditional Logit — Chamberlain (1980)
# =============================================================================

"""
    _clogit_dp_logsum(X_g, beta, s) -> (log_denom, grad, hess)

Dynamic programming to compute log(sum of exp(d' X_g beta)) over all binary vectors d
of length T_g with sum(d) = s, plus gradient and Hessian contributions.

Uses the recursion:
    f(t, j) = f(t-1, j) + f(t-1, j-1) * exp(x_t' beta)
where f(0,0) = 1, f(0, j>0) = 0.

We also track gradient terms via the chain rule.
"""
function _clogit_dp_logsum(X_g::AbstractMatrix{T}, beta::Vector{T}, s::Int) where {T}
    T_g = size(X_g, 1)
    k = length(beta)

    # DP table for the partition function
    # f[j+1] = sum over subsets of {1,...,t} with exactly j elements
    # We store log-scale values using log-sum-exp for stability
    # But for gradient/Hessian we need the actual sums, so use probability approach

    # Compute eta_t = x_t' beta for each t
    eta = X_g * beta
    exp_eta = exp.(eta)

    # Forward pass: f[t][j] = sum over subsets of {1,...,t} with j ones of prod exp(eta_i * d_i)
    # f[0][0] = 1, f[0][j>0] = 0
    # f[t][j] = f[t-1][j] + f[t-1][j-1] * exp(eta_t)

    # We need f[T_g][s] = denominator
    # Use 0-indexed j: j = 0, ..., s
    f_prev = zeros(T, s + 1)
    f_prev[1] = one(T)  # f[0][0] = 1

    for t in 1:T_g
        f_curr = zeros(T, s + 1)
        f_curr[1] = f_prev[1]  # j=0: don't include obs t
        for j in 1:min(s, t)
            f_curr[j + 1] = f_prev[j + 1] + f_prev[j] * exp_eta[t]
        end
        f_prev = f_curr
    end

    denom = f_prev[s + 1]

    # For gradient: d log(denom) / d beta = (1/denom) * d(denom)/d(beta)
    # d(denom)/d(beta_p) = sum over valid d of [sum_t d_t x_{t,p}] * exp(d' eta)
    #                    = sum_t x_{t,p} * g_t
    # where g_t = sum over d with sum=s and d_t=1 of exp(d' eta)
    # g_t = exp(eta_t) * f_{-t}[s-1]
    # f_{-t}[s-1] = sum over subsets of {1,...,T}\{t} with s-1 ones

    # To get g_t we can use forward-backward DP
    # Forward: fwd[t][j] = sum over subsets of {1,...,t} with j ones
    # Backward: bwd[t][j] = sum over subsets of {t,...,T} with j ones
    # Then f_{-t}[s-1] = sum_{j=0}^{s-1} fwd[t-1][j] * bwd[t+1][s-1-j]

    # Forward tables
    fwd = Vector{Vector{T}}(undef, T_g + 1)
    fwd[1] = zeros(T, s + 1)
    fwd[1][1] = one(T)
    for t in 1:T_g
        fwd[t + 1] = zeros(T, s + 1)
        fwd[t + 1][1] = fwd[t][1]
        for j in 1:min(s, t)
            fwd[t + 1][j + 1] = fwd[t][j + 1] + fwd[t][j] * exp_eta[t]
        end
    end

    # Backward tables
    bwd = Vector{Vector{T}}(undef, T_g + 2)
    bwd[T_g + 2] = zeros(T, s + 1)
    bwd[T_g + 2][1] = one(T)
    for t in T_g:-1:1
        bwd[t] = zeros(T, s + 1)  # bwd[t] = backward from t
        # Don't include t: bwd[t][j] = bwd[t+1][j] (skip) + bwd[t+1][j-1]*exp_eta[t] (include)
        # Wait, backward should be: sum over subsets of {t,...,T_g} with j ones
        # bwd[T_g+1][0] = 1
        # bwd[t][j] = bwd[t+1][j] + bwd[t+1][j-1] * exp_eta[t]
    end
    # Redo backward properly
    bwd[T_g + 1] = zeros(T, s + 1)
    bwd[T_g + 1][1] = one(T)
    for t in T_g:-1:1
        bwd[t] = zeros(T, s + 1)
        bwd[t][1] = bwd[t + 1][1]
        remaining = T_g - t + 1
        for j in 1:min(s, remaining)
            bwd[t][j + 1] = bwd[t + 1][j + 1] + bwd[t + 1][j] * exp_eta[t]
        end
    end

    # Compute g_t = P(d_t = 1 | sum = s) * denom = exp(eta_t) * f_{-t}(s-1)
    # f_{-t}(s-1) = sum_{j=0}^{s-1} fwd[t][j+1] * bwd[t+1][s-1-j+1]
    # (fwd[t] is forward up to t-1, i.e., fwd[t] = fwd over {1,...,t-1})
    g = zeros(T, T_g)
    for t in 1:T_g
        val = zero(T)
        for j in 0:min(s - 1, t - 1)
            bwd_idx = s - 1 - j  # need bwd[t+1][bwd_idx]
            remaining_after_t = T_g - t
            if bwd_idx <= remaining_after_t && bwd_idx >= 0
                val += fwd[t][j + 1] * bwd[t + 1][bwd_idx + 1]
            end
        end
        g[t] = exp_eta[t] * val
    end

    # prob_t = g_t / denom = P(d_t = 1 | sum_d = s)
    prob = denom > zero(T) ? g ./ denom : zeros(T, T_g)

    # Gradient of log-likelihood for this group:
    # ll_g = sum_t y_t * eta_t - log(denom)
    # d ll_g / d beta = X_g' * (y_g - prob)  (but we only need the denom gradient part)
    # Actually the full conditional log-lik for group:
    # ll_g = y_g' * X_g * beta - log(denom)
    # grad_g = X_g' * y_g - X_g' * prob = X_g' * (y_g - prob)

    # Hessian: -X_g' * diag(prob .* (1 .- prob)) * X_g
    # (This is the expected Hessian from the conditional distribution)

    log_denom = log(max(denom, T(1e-300)))

    return log_denom, prob
end

function _xtlogit_fe(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                     groups::Vector{Int}, unique_groups::Vector{Int},
                     N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                     maxiter::Int, tol::T) where {T}
    # No intercept for conditional logit (conditioned out)
    vn = [String(v) for v in indepvars]

    # Identify groups to keep (must have variation in y)
    keep_groups = Int[]
    group_obs = Dict{Int,Vector{Int}}()
    for g in unique_groups
        idx = findall(==(g), groups)
        y_g = y[idx]
        s_g = round(Int, sum(y_g))
        T_g = length(idx)
        if s_g > 0 && s_g < T_g
            push!(keep_groups, g)
            group_obs[g] = idx
        end
    end

    length(keep_groups) >= 2 ||
        throw(ArgumentError("Fewer than 2 groups with variation in y; cannot estimate FE logit"))

    # Newton-Raphson on conditional log-likelihood
    beta = zeros(T, k)
    loglik_old = T(-Inf)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        loglik = zero(T)
        grad = zeros(T, k)
        hess = zeros(T, k, k)

        for g in keep_groups
            idx = group_obs[g]
            X_g = X[idx, :]
            y_g = y[idx]
            s_g = round(Int, sum(y_g))

            log_denom, prob = _clogit_dp_logsum(X_g, beta, s_g)

            # Conditional log-likelihood: y_g' * X_g * beta - log(denom)
            loglik += dot(y_g, X_g * beta) - log_denom

            # Gradient: X_g' * (y_g - prob)
            grad .+= X_g' * (y_g .- prob)

            # Hessian: -X_g' * diag(prob .* (1 .- prob)) * X_g
            w_g = prob .* (one(T) .- prob)
            hess .-= X_g' * Diagonal(w_g) * X_g
        end

        # Convergence check
        if abs(loglik - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik
            break
        end
        loglik_old = loglik

        # Newton step: beta_new = beta - H^{-1} g
        hess_reg = hess
        # Regularize if needed
        if any(isnan, hess_reg) || any(isinf, hess_reg)
            break
        end
        step = try
            hess_reg \ grad
        catch
            break
        end
        beta .-= step
    end

    # Covariance: -H^{-1}
    hess_final = zeros(T, k, k)
    loglik_final = zero(T)
    fitted_all = zeros(T, n)

    for g in keep_groups
        idx = group_obs[g]
        X_g = X[idx, :]
        y_g = y[idx]
        s_g = round(Int, sum(y_g))

        log_denom, prob = _clogit_dp_logsum(X_g, beta, s_g)
        loglik_final += dot(y_g, X_g * beta) - log_denom

        w_g = prob .* (one(T) .- prob)
        hess_final .-= X_g' * Diagonal(w_g) * X_g

        # Store conditional probabilities as fitted values
        for (j, i) in enumerate(idx)
            fitted_all[i] = prob[j]
        end
    end

    vcov_mat = try
        robust_inv(-hess_final)
    catch
        zeros(T, k, k)
    end

    # Null log-likelihood for pseudo R2 (conditional, no covariates)
    loglik_null = zero(T)
    for g in keep_groups
        idx = group_obs[g]
        T_g = length(idx)
        s_g = round(Int, sum(y[idx]))
        # log(C(T_g, s_g))
        loglik_null -= log(T(binomial(BigInt(T_g), BigInt(s_g))))
    end

    pseudo_r2 = one(T) - loglik_final / loglik_null
    n_used = sum(length(group_obs[g]) for g in keep_groups)
    aic_val = -2 * loglik_final + 2 * T(k)
    bic_val = -2 * loglik_final + log(T(n_used)) * T(k)

    # For FE, fitted values are conditional probabilities for kept groups
    # For dropped groups, fitted is 0 or 1 trivially

    PanelLogitModel{T}(
        beta, vcov_mat, y, X, fitted_all,
        loglik_final, loglik_null, pseudo_r2, aic_val, bic_val,
        nothing, nothing,  # sigma_u, rho (not applicable for conditional)
        vn, :fe, :ols, converged, iterations, n_used, length(keep_groups), pd
    )
end

# =============================================================================
# RE Logit — Gauss-Hermite Quadrature
# =============================================================================

"""
    _re_logit_loglik(beta, log_sigma_u, y, X, groups, unique_groups, group_obs, nodes, weights)

Compute log-likelihood, gradient, and Hessian for RE logit model via GH quadrature.
"""
function _re_logit_loglik(theta::Vector{T}, y::Vector{T}, X_c::Matrix{T},
                          groups::Vector{Int}, unique_groups::Vector{Int},
                          group_obs::Dict{Int,Vector{Int}},
                          nodes::Vector{Float64}, weights::Vector{Float64}) where {T}
    k = size(X_c, 2)
    beta = theta[1:k]
    log_sigma_u = theta[k + 1]
    sigma_u = exp(log_sigma_u)

    n_quad = length(nodes)
    loglik = zero(T)
    grad = zeros(T, k + 1)
    hess = zeros(T, k + 1, k + 1)

    for g in unique_groups
        idx = group_obs[g]
        X_g = X_c[idx, :]
        y_g = y[idx]
        T_g = length(idx)
        eta_g = X_g * beta  # linear predictor without RE

        # Gauss-Hermite: integrate over alpha_i ~ N(0, sigma_u^2)
        # nodes for exp(-x^2), so alpha = sqrt(2) * sigma_u * node
        # weight correction: 1/sqrt(pi)
        Li = zero(T)
        dLi_dbeta = zeros(T, k)
        dLi_dsig = zero(T)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))

            # Product of likelihoods for this group at this quadrature point
            log_prod = zero(T)
            for j in 1:T_g
                mu_j = one(T) / (one(T) + exp(-(eta_g[j] + alpha)))
                mu_j = clamp(mu_j, T(1e-10), one(T) - T(1e-10))
                log_prod += y_g[j] * log(mu_j) + (one(T) - y_g[j]) * log(one(T) - mu_j)
            end

            contrib = wq * exp(log_prod)
            Li += contrib

            # Gradient contributions (for numerical stability, compute ratios)
            # d(prod)/d(beta_p) = prod * sum_j (y_j - mu_j) * x_{j,p}
            # d(prod)/d(sigma_u) = prod * sum_j (y_j - mu_j) * d(alpha)/d(sigma_u)
            # where d(alpha)/d(sigma_u) = sqrt(2) * node_q * d(sigma_u)/d(log_sigma_u) = sqrt(2) * node_q * sigma_u
            # but we parameterize by log_sigma_u, so d(alpha)/d(log_sigma_u) = sqrt(2) * node_q * sigma_u

            if contrib > T(1e-300)
                score_j = zeros(T, T_g)
                for j in 1:T_g
                    mu_j = one(T) / (one(T) + exp(-(eta_g[j] + alpha)))
                    mu_j = clamp(mu_j, T(1e-10), one(T) - T(1e-10))
                    score_j[j] = y_g[j] - mu_j
                end

                dLi_dbeta .+= contrib .* (X_g' * score_j)
                dalpha_dlogsig = sqrt(T(2)) * T(nodes[q]) * sigma_u
                dLi_dsig += contrib * dot(score_j, fill(dalpha_dlogsig, T_g))
            end
        end

        Li = max(Li, T(1e-300))
        loglik += log(Li)

        # Gradient: d(log Li)/d(theta) = (1/Li) * dLi/d(theta)
        grad[1:k] .+= dLi_dbeta ./ Li
        grad[k + 1] += dLi_dsig / Li
    end

    return loglik, grad
end

function _xtlogit_re(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                     groups::Vector{Int}, unique_groups::Vector{Int},
                     N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                     cov_type::Symbol, maxiter::Int, tol::T,
                     n_quadrature::Int) where {T}
    # Add intercept
    X_c = hcat(ones(T, n), X)
    k_full = k + 1
    vn = vcat(["_cons"], [String(v) for v in indepvars])

    # Pre-compute group obs indices
    group_obs = Dict{Int,Vector{Int}}()
    for g in unique_groups
        group_obs[g] = findall(==(g), groups)
    end

    # GH nodes/weights
    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature)

    # Initialize: pooled logit coefficients + log(sigma_u) = log(1)
    beta_init, _, _, _, _, _ = _irls_logit(y, X_c; maxiter=50, tol=T(1e-6))
    theta = vcat(beta_init, zero(T))  # [beta; log_sigma_u]

    loglik_old = T(-Inf)
    converged = false
    iterations = 0

    # BFGS-like optimization via gradient ascent with step size control
    for iter in 1:maxiter
        iterations = iter

        loglik, grad = _re_logit_loglik(theta, y, X_c, groups, unique_groups,
                                         group_obs, nodes, weights)

        if abs(loglik - loglik_old) < tol * (abs(loglik_old) + one(T))
            converged = true
            loglik_old = loglik
            break
        end
        loglik_old = loglik

        # Simple gradient ascent with adaptive step size
        step_size = T(0.5)
        for _ in 1:20
            theta_new = theta .+ step_size .* grad
            ll_new, _ = _re_logit_loglik(theta_new, y, X_c, groups, unique_groups,
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

    # Numerical Hessian for covariance
    loglik_final, _ = _re_logit_loglik(theta, y, X_c, groups, unique_groups,
                                        group_obs, nodes, weights)

    n_params = k_full + 1
    hess = zeros(T, n_params, n_params)
    eps_h = T(1e-4)
    for i in 1:n_params
        theta_p = copy(theta)
        theta_m = copy(theta)
        theta_p[i] += eps_h
        theta_m[i] -= eps_h
        _, g_p = _re_logit_loglik(theta_p, y, X_c, groups, unique_groups,
                                   group_obs, nodes, weights)
        _, g_m = _re_logit_loglik(theta_m, y, X_c, groups, unique_groups,
                                   group_obs, nodes, weights)
        hess[:, i] .= (g_p .- g_m) ./ (2 * eps_h)
    end
    hess = (hess .+ hess') ./ 2

    # Covariance of beta only (not log_sigma_u)
    vcov_full = try
        robust_inv(-hess)
    catch
        zeros(T, n_params, n_params)
    end
    vcov_mat = vcov_full[1:k_full, 1:k_full]

    # Fitted probabilities (marginal, integrating out alpha)
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
                p += wq / (one(T) + exp(-(eta_g[j] + alpha)))
            end
            fitted[i] = p
        end
    end

    # Null log-likelihood
    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    pseudo_r2 = one(T) - loglik_final / loglik_null
    aic_val = -2 * loglik_final + 2 * T(n_params)
    bic_val = -2 * loglik_final + log(T(n)) * T(n_params)

    # rho = sigma_u^2 / (sigma_u^2 + pi^2/3)
    rho = sigma_u^2 / (sigma_u^2 + T(pi)^2 / 3)

    PanelLogitModel{T}(
        beta, vcov_mat, y, X_c, fitted,
        loglik_final, loglik_null, pseudo_r2, aic_val, bic_val,
        sigma_u, rho,
        vn, :re, cov_type, converged, iterations, n, N, pd
    )
end

# =============================================================================
# CRE Logit — Mundlak Augmentation + RE
# =============================================================================

function _xtlogit_cre(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
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

    # Augment X with group means
    X_aug = hcat(X, X_means)

    # Variable names
    vn_orig = [String(v) for v in indepvars]
    vn_mean = [String(v) * "_mean" for v in indepvars]

    # Create augmented PanelData-like call to RE
    # We can reuse _xtlogit_re with augmented X
    k_aug = 2k
    indepvars_aug = vcat(indepvars, [Symbol(String(v) * "_mean") for v in indepvars])

    # Add intercept
    X_c = hcat(ones(T, n), X_aug)
    k_full = k_aug + 1
    vn = vcat(["_cons"], vn_orig, vn_mean)

    # Pre-compute group obs indices
    group_obs = Dict{Int,Vector{Int}}()
    for g in unique_groups
        group_obs[g] = findall(==(g), groups)
    end

    # GH nodes/weights
    nodes, weights = _gauss_hermite_nodes_weights(n_quadrature)

    # Initialize
    beta_init, _, _, _, _, _ = _irls_logit(y, X_c; maxiter=50, tol=T(1e-6))
    theta = vcat(beta_init, zero(T))

    loglik_old = T(-Inf)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        loglik, grad = _re_logit_loglik(theta, y, X_c, groups, unique_groups,
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
            ll_new, _ = _re_logit_loglik(theta_new, y, X_c, groups, unique_groups,
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
        _, g_p = _re_logit_loglik(theta_p, y, X_c, groups, unique_groups,
                                   group_obs, nodes, weights)
        _, g_m = _re_logit_loglik(theta_m, y, X_c, groups, unique_groups,
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
                p += wq / (one(T) + exp(-(eta_g[j] + alpha)))
            end
            fitted[i] = p
        end
    end

    # Null log-likelihood
    p_bar = clamp(mean(y), T(1e-10), one(T) - T(1e-10))
    loglik_null = T(n) * (p_bar * log(p_bar) + (one(T) - p_bar) * log(one(T) - p_bar))

    pseudo_r2 = one(T) - loglik_final / loglik_null
    aic_val = -2 * loglik_final + 2 * T(n_params)
    bic_val = -2 * loglik_final + log(T(n)) * T(n_params)

    rho = sigma_u^2 / (sigma_u^2 + T(pi)^2 / 3)

    PanelLogitModel{T}(
        beta, vcov_mat, y, X_c, fitted,
        loglik_final, loglik_null, pseudo_r2, aic_val, bic_val,
        sigma_u, rho,
        vn, :cre, cov_type, converged, iterations, n, N, pd
    )
end
