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
    _clogit_dp_logsum(X_g, beta, s) -> (log_denom, prob)

Fixed-effects conditional-logit partition function via a forward/backward dynamic program
in LOG SPACE. Returns `log_denom = log Σ_{d: Σd=s} exp(d'X_gβ)` and, for each observation
`t`, `prob[t] = P(d_t = 1 ∣ Σd = s)`. The group conditional log-likelihood is
`y_g'X_gβ − log_denom` with gradient `X_g'(y_g − prob)` and Hessian
`−X_g' diag(prob(1−prob)) X_g`.

The recursion `f(t,j) = f(t-1,j) + f(t-1,j-1)·e^{η_t}` is carried on the log scale via
`logaddexp` so that `|η_t|` on the order of 700+ (raw `exp` overflows to `Inf` in Float64)
no longer produces `Inf`/`NaN`.
"""
function _clogit_dp_logsum(X_g::AbstractMatrix{T}, beta::Vector{T}, s::Int) where {T}
    T_g = size(X_g, 1)
    eta = X_g * beta
    NEG = T(-Inf)
    # Stable scalar log-add-exp with -Inf handling (logaddexp(-Inf,-Inf) = -Inf).
    la(a, b) = a == NEG ? b : (b == NEG ? a :
               (a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))))

    # Forward log-DP: lfwd[t+1][j+1] = log Σ over subsets of {1..t} with j ones of exp(Σ d·η).
    lfwd = Vector{Vector{T}}(undef, T_g + 1)
    lfwd[1] = fill(NEG, s + 1); lfwd[1][1] = zero(T)   # log f(0,0)=0, others -Inf
    for t in 1:T_g
        prev = lfwd[t]
        cur = fill(NEG, s + 1)
        cur[1] = prev[1]                                # j=0: exclude obs t
        for j in 1:min(s, t)
            cur[j + 1] = la(prev[j + 1], prev[j] + eta[t])
        end
        lfwd[t + 1] = cur
    end
    log_denom = lfwd[T_g + 1][s + 1]

    # Backward log-DP: lbwd[t][j+1] = log Σ over subsets of {t..T_g} with j ones.
    lbwd = Vector{Vector{T}}(undef, T_g + 1)
    lbwd[T_g + 1] = fill(NEG, s + 1); lbwd[T_g + 1][1] = zero(T)
    for t in T_g:-1:1
        nxt = lbwd[t + 1]
        cur = fill(NEG, s + 1)
        cur[1] = nxt[1]
        remaining = T_g - t + 1
        for j in 1:min(s, remaining)
            cur[j + 1] = la(nxt[j + 1], nxt[j] + eta[t])
        end
        lbwd[t] = cur
    end

    # prob[t] = exp(η_t + logf_{-t}(s-1) − log_denom), where lfwd[t] spans {1..t-1} and
    # lbwd[t+1] spans {t+1..T_g}: logf_{-t}(s-1) = logsumexp_j lfwd[t][j+1] + lbwd[t+1][s-1-j+1].
    prob = zeros(T, T_g)
    if isfinite(log_denom)
        for t in 1:T_g
            acc = NEG
            for j in 0:min(s - 1, t - 1)
                bidx = s - 1 - j
                (0 <= bidx <= T_g - t) || continue
                acc = la(acc, lfwd[t][j + 1] + lbwd[t + 1][bidx + 1])
            end
            prob[t] = acc == NEG ? zero(T) : exp(eta[t] + acc - log_denom)
        end
    end

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

"""
    _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, x_nodes, w_nodes;
                         agh_newton_iters=8) -> S

Adaptive Gauss-Hermite marginal log-likelihood for the RE/CRE logit. Generic in the element
type `S` of `theta = [β; log σ_u]` so ForwardDiff can differentiate through it. For each
group the integrand over `α ~ N(0, σ_u²)` is recentered at its posterior mode `μ̂` (scalar
Newton) and rescaled by the curvature `σ̂` (Liu & Pierce 1994; Rabe-Hesketh, Skrondal &
Pickles 2005): nodes `a_q = μ̂ + √2·σ̂·x_q`, log-weights `log(√2·σ̂) + log(w_q) + x_q²`
(the `+x_q²` cancels the `e^{-x²}` baked into the standard GH weights). Reduces to the
Laplace approximation at `length(x_nodes)==1`.
"""
function _re_logit_agh_loglik(theta::AbstractVector{S}, y::Vector{T}, X_c::Matrix{T},
                              unique_groups::Vector{Int}, group_obs::Dict{Int,Vector{Int}},
                              x_nodes::Vector{Float64}, w_nodes::Vector{Float64};
                              agh_newton_iters::Int=8) where {S,T}
    k = size(X_c, 2)
    beta = theta[1:k]
    sigma_u = exp(theta[k+1])
    s2 = sigma_u^2
    nq = length(x_nodes)
    half_log2pi = S(0.5 * log(2π))
    log_sqrt2 = S(0.5 * log(2.0))
    total = zero(S)

    for g in unique_groups
        idx = group_obs[g]
        eta = @view(X_c[idx, :]) * beta           # Vector{S}
        yg = @view y[idx]

        # (a) posterior mode μ̂ of h(α) = ℓ_g(α) − α²/(2σ²) via scalar Newton from 0
        alpha = zero(S)
        for _ in 1:agh_newton_iters
            hp = -alpha / s2
            hpp = -one(S) / s2
            @inbounds for j in eachindex(eta)
                mu = one(S) / (one(S) + exp(-(eta[j] + alpha)))
                hp += yg[j] - mu
                hpp -= mu * (one(S) - mu)
            end
            alpha -= hp / hpp                      # h''<0 ⇒ stable Newton toward the mode
            abs(hp) < S(1e-10) && break
        end
        info = one(S) / s2                          # curvature σ̂ = 1/√(−h''(μ̂))
        @inbounds for j in eachindex(eta)
            mu = one(S) / (one(S) + exp(-(eta[j] + alpha)))
            info += mu * (one(S) - mu)
        end
        sighat = one(S) / sqrt(info)

        # (b,c) adaptive nodes + stable logsumexp of logω_q + ℓ_g(a_q) + logφ(a_q;0,σ²)
        logvals = Vector{S}(undef, nq)
        @inbounds for q in 1:nq
            a = alpha + sqrt(S(2)) * sighat * S(x_nodes[q])
            logw = log_sqrt2 + log(sighat) + log(S(w_nodes[q])) + S(x_nodes[q])^2
            ll = zero(S)
            for j in eachindex(eta)
                mu = clamp(one(S) / (one(S) + exp(-(eta[j] + a))), S(1e-12), one(S) - S(1e-12))
                ll += yg[j] * log(mu) + (one(S) - yg[j]) * log(one(S) - mu)
            end
            logphi = -half_log2pi - log(sigma_u) - a^2 / (2 * s2)
            logvals[q] = logw + ll + logphi
        end
        mx = maximum(logvals)
        acc = zero(S)
        @inbounds for q in 1:nq
            acc += exp(logvals[q] - mx)
        end
        total += mx + log(acc)
    end
    total
end

"""
    _agh_newton_polish(nll, theta) -> (theta, gnorm)

Damped-Newton polish of the adaptive-GH marginal negative loglik around the LBFGS
minimizer. Optim's `f_reltol` stopping can halt with `‖∇nll‖` marginally above the
honest-convergence FOC threshold (the exact stopping point varies across Optim
versions); a few backtracking Newton steps on the exact ForwardDiff Hessian drive
the gradient to numerical zero whenever the solution is a genuine local optimum.
Returns the (possibly improved) parameter vector and the final gradient norm.
"""
function _agh_newton_polish(nll::F, theta::Vector{T}) where {F,T<:AbstractFloat}
    g = ForwardDiff.gradient(nll, theta)
    for _ in 1:5
        gnorm = norm(g)
        (isfinite(gnorm) && gnorm >= T(1e-8)) || break
        H = ForwardDiff.hessian(nll, theta)
        step = try
            -(Hermitian((H .+ H') ./ 2) \ g)
        catch
            break
        end
        all(isfinite, step) || break
        f0 = nll(theta)
        lam = one(T)
        accepted = false
        for _ in 1:20
            cand = theta .+ lam .* step
            fc = nll(cand)
            if isfinite(fc) && fc <= f0
                theta = cand
                accepted = true
                break
            end
            lam /= 2
        end
        accepted || break
        g = ForwardDiff.gradient(nll, theta)
    end
    return theta, norm(g)
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
    theta0 = vcat(beta_init, zero(T))  # [beta; log_sigma_u]

    # Maximize the adaptive-GH marginal loglik with LBFGS on ForwardDiff gradients (replaces
    # the ad-hoc gradient ascent whose function-value stopping rule reported false convergence).
    nll(th) = -_re_logit_agh_loglik(th, y, X_c, unique_groups, group_obs, nodes, weights)
    g!(G, th) = (G .= ForwardDiff.gradient(nll, th))
    res = Optim.optimize(nll, g!, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol=T(1e-8), iterations=maxiter, f_reltol=tol))
    theta = Optim.minimizer(res)
    iterations = Optim.iterations(res)
    theta, gnorm = _agh_newton_polish(nll, theta)

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    loglik_final = _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, nodes, weights)

    # Honest convergence: reported only when the true gradient norm is near zero.
    converged = isfinite(gnorm) && gnorm < T(1e-5)

    # SEs from observed information = Hessian of the NEGATIVE loglik (no sign flip).
    n_params = k_full + 1
    H = ForwardDiff.hessian(nll, theta)
    vcov_full = try
        Matrix{T}(robust_inv(Hermitian((H .+ H') ./ 2)))
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
    theta0 = vcat(beta_init, zero(T))

    nll(th) = -_re_logit_agh_loglik(th, y, X_c, unique_groups, group_obs, nodes, weights)
    g!(G, th) = (G .= ForwardDiff.gradient(nll, th))
    res = Optim.optimize(nll, g!, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol=T(1e-8), iterations=maxiter, f_reltol=tol))
    theta = Optim.minimizer(res)
    iterations = Optim.iterations(res)
    theta, gnorm = _agh_newton_polish(nll, theta)

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    # Recompute the loglik at the optimum (the old code reused loglik_old — a stale value).
    loglik_final = _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, nodes, weights)

    converged = isfinite(gnorm) && gnorm < T(1e-5)

    n_params = k_full + 1
    H = ForwardDiff.hessian(nll, theta)
    vcov_full = try
        Matrix{T}(robust_inv(Hermitian((H .+ H') ./ 2)))
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
