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
                     maxiter=2000, tol=1e-8, n_quadrature=12) -> PanelLogitModel{T}

Estimate a panel logistic regression model.

# Arguments
- `pd::PanelData{T}` -- panel data container (created via `xtset`)
- `depvar::Symbol` -- binary dependent variable (0/1)
- `indepvars::Vector{Symbol}` -- independent variable names

# Keyword Arguments
- `model::Symbol` -- `:pooled`, `:fe`, `:re`, or `:cre` (default: `:pooled`)
- `cov_type::Symbol` -- covariance type: `:ols`, `:cluster` (default)
- `maxiter::Int` -- maximum iterations (default 2000; FE conditional logit often needs 1000+)
- `tol` -- convergence tolerance (default 1e-8). Under `model=:fe` it is applied to the
  sup-norm of the conditional score, floored at `1e-5`; standard errors come from the
  conditional information (`:ols`) or its entity-clustered sandwich (`:cluster`)
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
                          maxiter::Int=2000, tol::Real=1e-8,
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
        return _xtlogit_fe(pd, y, X, groups, unique_groups, N, n, k, indepvars, cov_type,
                           maxiter, T(tol))
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

"""
    _clogit_ll_grad(X, y, keep_groups, group_obs, beta, k) -> (loglik, grad, scores)

Conditional log-likelihood, its exact score `Σ_g X_g'(y_g − p_g)`, and the per-group
score contributions, with `p_g` the forward-backward DP conditional probabilities.
"""
function _clogit_ll_grad(X::Matrix{T}, y::Vector{T}, keep_groups::Vector{Int},
                         group_obs::Dict{Int,Vector{Int}}, beta::Vector{T},
                         k::Int) where {T<:AbstractFloat}
    loglik = zero(T)
    grad = zeros(T, k)
    scores = Vector{Vector{T}}(undef, length(keep_groups))
    for (j, g) in enumerate(keep_groups)
        idx = group_obs[g]
        X_g = X[idx, :]
        y_g = y[idx]
        s_g = round(Int, sum(y_g))
        log_denom, prob = _clogit_dp_logsum(X_g, beta, s_g)
        loglik += dot(y_g, X_g * beta) - log_denom
        score_g = X_g' * (y_g .- prob)
        scores[j] = score_g
        grad .+= score_g
    end
    (loglik, grad, scores)
end

"""
    _clogit_information(X, y, keep_groups, group_obs, beta, k) -> Matrix{T}

Observed conditional information `−∇²ℓ` by central differences of the exact DP score.
The analytic second derivative needs the pairwise probabilities `Pr(y_t=1, y_s=1 | Σy)`,
which the forward-backward recursion does not return; differencing the exact gradient is
accurate to `O(h²)` at a cost of `2k` DP passes (the technique `_re_probit_vcov` uses).
Unlike the independent-Bernoulli matrix `X_g' Diag(p(1−p)) X_g` this is invariant to
within-group shifts of `X`, as the conditional likelihood itself is (#543).
"""
function _clogit_information(X::Matrix{T}, y::Vector{T}, keep_groups::Vector{Int},
                             group_obs::Dict{Int,Vector{Int}}, beta::Vector{T},
                             k::Int) where {T<:AbstractFloat}
    H = zeros(T, k, k)
    for j in 1:k
        h = T(1e-4) * max(one(T), abs(beta[j]))
        beta_p = copy(beta); beta_p[j] += h
        beta_m = copy(beta); beta_m[j] -= h
        _, g_p, _ = _clogit_ll_grad(X, y, keep_groups, group_obs, beta_p, k)
        _, g_m, _ = _clogit_ll_grad(X, y, keep_groups, group_obs, beta_m, k)
        H[:, j] .= (g_p .- g_m) ./ (2h)
    end
    -(H .+ H') ./ 2
end

function _xtlogit_fe(pd::PanelData{T}, y::Vector{T}, X::Matrix{T},
                     groups::Vector{Int}, unique_groups::Vector{Int},
                     N::Int, n::Int, k::Int, indepvars::Vector{Symbol},
                     cov_type::Symbol, maxiter::Int, tol::T) where {T}
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

    # Newton-Raphson on the conditional log-likelihood (globally concave), stepping with
    # the conditional information. Convergence is judged on ‖∇‖∞: the loglik-change rule
    # declared convergence at points where the score was still far from zero. The 1e-5
    # floor matches the RE path and the precision attainable for a score summed over
    # thousands of groups (#543).
    gtol = max(T(tol), T(1e-5))
    beta = zeros(T, k)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        loglik, grad, _ = _clogit_ll_grad(X, y, keep_groups, group_obs, beta, k)

        if maximum(abs, grad) < gtol
            converged = true
            break
        end

        info = _clogit_information(X, y, keep_groups, group_obs, beta, k)
        (any(isnan, info) || any(isinf, info)) && break
        step = try
            info \ grad
        catch
            break
        end
        # Backtrack if the full Newton step does not improve the likelihood (guards
        # against a near-singular information on weakly identified designs).
        t_step = one(T)
        for _ in 1:20
            _clogit_ll_grad(X, y, keep_groups, group_obs, beta .+ t_step .* step, k)[1] > loglik && break
            t_step /= 2
        end
        beta .+= t_step .* step
    end

    if !converged
        @warn "FE conditional logit did not converge in $iterations iterations " *
              "(maxiter=$maxiter, score tolerance $gtol). Coefficient estimates may be " *
              "unreliable. Raise maxiter or relax tol." maxlog=1
    end

    # Covariance from the conditional information (Stata `clogit`), optionally sandwiched
    # with the per-group score meat when entity-clustered SEs are requested. The
    # independent-Bernoulli matrix used previously is not the conditional information: it
    # scales with the LEVEL of X, which the conditional likelihood conditions away (#543).
    loglik_final, _, scores = _clogit_ll_grad(X, y, keep_groups, group_obs, beta, k)
    info_final = _clogit_information(X, y, keep_groups, group_obs, beta, k)

    fitted_all = zeros(T, n)
    for g in keep_groups
        idx = group_obs[g]
        _, prob = _clogit_dp_logsum(X[idx, :], beta, round(Int, sum(y[idx])))
        # Store conditional probabilities as fitted values
        for (j, i) in enumerate(idx)
            fitted_all[i] = prob[j]
        end
    end

    vcov_mat = try
        bread = Matrix{T}(robust_inv(Hermitian((info_final .+ info_final') ./ 2)))
        if cov_type == :cluster
            meat = zeros(T, k, k)
            for s_g in scores
                meat .+= s_g * s_g'
            end
            G = length(keep_groups)
            n_used_c = sum(length(group_obs[g]) for g in keep_groups)
            meat .*= T(G) / T(G - 1) * T(n_used_c - 1) / T(max(n_used_c - k, 1))
            V = bread * meat * bread
            Matrix{T}((V .+ V') ./ 2)
        else
            bread
        end
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
        vn, :fe, cov_type, converged, iterations, n_used, length(keep_groups), pd
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

"""Overflow-safe logistic. `exp(-z)` overflows to `Inf` for `z ≲ -709`; the value survives
(`1/(1+Inf) = 0`) but the ForwardDiff dual carries `Inf/Inf = NaN` into the gradient. (#600)"""
_agh_logistic(z::S) where {S} =
    z >= zero(S) ? one(S) / (one(S) + exp(-z)) : (e = exp(z); e / (one(S) + e))

"""Overflow-safe `log(1 + exp(z))`, used for the Bernoulli log-likelihood in the form
`y·z − log1pexp(z)`. Never forms `exp(large)`, so neither the value nor its derivative
overflows and no probability clamp is needed. (#600)"""
_agh_log1pexp(z::S) where {S} =
    z > zero(S) ? z + log1p(exp(-z)) : log1p(exp(z))

"""
    _agh_group_mode(eta, yg, inv_s2; maxiter=200) -> alpha

Posterior mode μ̂ of `h(α) = ℓ_g(α) − α²/(2σ²)` for one group, by safeguarded Newton:
`h'` is strictly decreasing (h'' < 0 everywhere), so the root is bracketed by doubling
and every Newton step that leaves the bracket is replaced by bisection. The mode MUST
be converged tightly, not run for a fixed iteration count: a truncated inner iteration
makes the marginal loglik VALUE jagged in θ (on ddcg, O(1) jumps under 1e-3 parameter
moves), which sent LBFGS to noise-dependent pseudo-optima several loglik units apart
across Optim versions and made finite-difference Hessians indefinite (#542).

When `inv_s2` underflows and the group is degenerate (all-0/all-1), `h'` never crosses
zero; the bracket expansion then caps at ~2⁶⁰ and returns the capped point — the
integrand is flat out there and the quadrature value stays finite (#600 regime).
Generic in `S` so ForwardDiff duals pass through; at a converged root the residual is
~0, so the dual parts agree with the implicit-function derivative.
"""
function _agh_group_mode(eta::AbstractVector{S}, yg::AbstractVector{<:Real},
                         inv_s2::S; maxiter::Int=200) where {S}
    # Bernoulli part only: c(α) = Σ_j (y_j − μ_j), c'(α) = −Σ_j μ_j(1−μ_j). Kept separate
    # from the prior term −α·inv_s2 so the Newton update can be evaluated in the
    # cancellation-free form α₊ = (c − α·c')/(inv_s2 − c'): the naive α − h'/h'' rounds to
    # exactly α (i.e. a zero step) whenever α·inv_s2 dominates c in Float64, which turned
    # the loop into pure arithmetic bisection that cannot reach a root at 1e-295 (σ_u → 0).
    bern = function (alpha::S)
        c = zero(S)
        cp = zero(S)
        @inbounds for j in eachindex(eta)
            mu = _agh_logistic(eta[j] + alpha)
            c += yg[j] - mu
            cp -= mu * (one(S) - mu)
        end
        c, cp
    end
    alpha = zero(S)
    c, cp = bern(alpha)
    hp = c - alpha * inv_s2
    abs(hp) < S(1e-10) && return alpha
    # Bracket [lo, hi] with h'(lo) > 0 > h'(hi); h' is strictly decreasing.
    local lo::S, hi::S
    if hp > 0
        lo = alpha
        step = one(S)
        hi = lo + step
        hp_hi = (ch = bern(hi)[1]; ch - hi * inv_s2)
        nexp = 0
        while hp_hi > 0 && nexp < 60
            lo = hi
            step *= 2
            hi = lo + step
            hp_hi = (ch = bern(hi)[1]; ch - hi * inv_s2)
            nexp += 1
        end
        hp_hi > 0 && return hi          # no root (degenerate group, σ_u → ∞): capped
    else
        hi = alpha
        step = one(S)
        lo = hi - step
        hp_lo = (cl = bern(lo)[1]; cl - lo * inv_s2)
        nexp = 0
        while hp_lo < 0 && nexp < 60
            hi = lo
            step *= 2
            lo = hi - step
            hp_lo = (cl = bern(lo)[1]; cl - lo * inv_s2)
            nexp += 1
        end
        hp_lo < 0 && return lo
    end
    for _ in 1:maxiter
        denom = max(inv_s2 - cp, eps(one(S)))       # −h'' > 0 always
        cand = (c - alpha * cp) / denom             # Newton step, cancellation-free form
        alpha_new = (lo < cand < hi) ? cand : (lo + hi) / 2
        alpha_new == alpha && break                 # no representable progress left
        alpha = alpha_new
        c, cp = bern(alpha)
        hp = c - alpha * inv_s2
        abs(hp) < S(1e-10) && break
        if hp > 0
            lo = alpha
        else
            hi = alpha
        end
    end
    alpha
end

"""
    _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, x_nodes, w_nodes;
                         agh_newton_iters=200) -> S

Adaptive Gauss-Hermite marginal log-likelihood for the RE/CRE logit. Generic in the element
type `S` of `theta = [β; log σ_u]` so ForwardDiff can differentiate through it. For each
group the integrand over `α ~ N(0, σ_u²)` is recentered at its posterior mode `μ̂`
([`_agh_group_mode`](@ref)) and rescaled by the curvature `σ̂` (Liu & Pierce 1994;
Rabe-Hesketh, Skrondal & Pickles 2005): nodes `a_q = μ̂ + √2·σ̂·x_q`, log-weights
`log(√2·σ̂) + log(w_q) + x_q²` (the `+x_q²` cancels the `e^{-x²}` baked into the standard
GH weights). Reduces to the Laplace approximation at `length(x_nodes)==1`.

Everything below is written so that neither the value nor the ForwardDiff gradient can go
non-finite anywhere the optimizer can reach (#600): probabilities go through
[`_agh_logistic`](@ref)/[`_agh_log1pexp`](@ref) rather than a clamp on `1/(1+exp(-z))`, and
the prior enters as `inv_s2 = exp(-2 log σ_u)` and the additive `log σ_u` rather than
`s2 = exp(2 log σ_u)` — a large trial `log σ_u` then underflows `inv_s2` to zero instead of
producing `Inf/Inf`. `HagerZhang` asserts finiteness of the trial value *and* its
directional derivative, so a single `NaN` partial aborted the whole fit.
"""
function _re_logit_agh_loglik(theta::AbstractVector{S}, y::Vector{T}, X_c::Matrix{T},
                              unique_groups::Vector{Int}, group_obs::Dict{Int,Vector{Int}},
                              x_nodes::Vector{Float64}, w_nodes::Vector{Float64};
                              agh_newton_iters::Int=200) where {S,T}
    k = size(X_c, 2)
    beta = theta[1:k]
    # Carry the *inverse* variance: exp(-2ℓ) underflows to 0 for a large trial ℓ, whereas
    # s2 = exp(2ℓ) overflows to Inf and turns 1/s2 and a²/(2s2) into NaN partials (#600).
    # ℓ is clamped to a range where exp(±2ℓ) is representable; the clamp must apply to
    # EVERY use of ℓ, because the −log σ_u normalizer in logφ cancels against log σ̂ in the
    # quadrature weight — clamping one and not the other sends the objective to +∞ as
    # σ_u → 0 and manufactures an optimum at the degenerate point. σ_u = e^{±340} is
    # numerically 0/∞ anyway, so a flat objective out there is the honest limit.
    log_sigma_u = clamp(theta[k+1], S(-340), S(340))
    inv_s2 = exp(-2 * log_sigma_u)
    nq = length(x_nodes)
    half_log2pi = S(0.5 * log(2π))
    log_sqrt2 = S(0.5 * log(2.0))
    total = zero(S)

    for g in unique_groups
        idx = group_obs[g]
        eta = @view(X_c[idx, :]) * beta           # Vector{S}
        yg = @view y[idx]

        # (a) posterior mode μ̂ of h(α) = ℓ_g(α) − α²/(2σ²), converged to |h'| < 1e-10
        # by safeguarded Newton — see _agh_group_mode for why truncation is not an option.
        alpha = _agh_group_mode(eta, yg, inv_s2; maxiter=agh_newton_iters)
        info = inv_s2                               # curvature σ̂ = 1/√(−h''(μ̂))
        @inbounds for j in eachindex(eta)
            mu = _agh_logistic(eta[j] + alpha)
            info += mu * (one(S) - mu)
        end
        sighat = one(S) / sqrt(max(info, eps(one(S))))

        # (b,c) adaptive nodes + stable logsumexp of logω_q + ℓ_g(a_q) + logφ(a_q;0,σ²)
        logvals = Vector{S}(undef, nq)
        @inbounds for q in 1:nq
            a = alpha + sqrt(S(2)) * sighat * S(x_nodes[q])
            logw = log_sqrt2 + log(sighat) + log(S(w_nodes[q])) + S(x_nodes[q])^2
            ll = zero(S)
            for j in eachindex(eta)
                z = eta[j] + a                      # y·z − log(1+e^z): exact, never overflows
                ll += yg[j] * z - _agh_log1pexp(z)
            end
            logphi = -half_log2pi - log_sigma_u - a^2 * inv_s2 / 2
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
    _re_logit_agh_score_info(theta, y, X_c, unique_groups, group_obs, x_nodes, w_nodes)
        -> (score, info, group_scores)

Marginal score and Louis (1982) observed information of the RE/CRE logit marginal
loglik, via posterior expectations over each group's random effect computed on the
same adaptive Gauss–Hermite rule as the loglik. For group g with complete-data loglik
`ℓ_c(α) = Σ_j [y_j z_j − log(1+e^{z_j})] + log φ(α; 0, σ_u²)`, `z_j = x_j'β + α`:

    ∇ℓ_g = E[∇ℓ_c | y_g]                      (Fisher's identity)
    −∇²ℓ_g = E[−∇²ℓ_c | y_g] − Var[∇ℓ_c | y_g]  (missing-information principle)

with complete-data derivatives w.r.t. θ = [β; ℓ], ℓ = log σ_u:

    s_β = Σ_j (y_j − μ_j) x_j     s_ℓ = α²/σ² − 1
    −H_ββ = Σ_j μ_j(1−μ_j) x_j x_j'   −H_ℓℓ = 2α²/σ²   −H_βℓ = 0

This deliberately never differentiates through the inner mode search: the ForwardDiff
Hessian of the AGH objective carried a spurious O(1e11) eigenvalue on ddcg and produced
standard errors below the complete-data information bound — mathematically impossible
values like z = 2674 for a logit slope (#542). Returns the total score, the observed
information (positive definite at a genuine optimum), and the per-group scores
(n_params × G) for the cluster sandwich.

Note these approximate the EXACT marginal score/information (verified against dense
numerical integration to 1e-6): at the optimum of the AGH-`n_quadrature` objective the
returned score is O(quadrature error), not 0 — the right target for inference, but not
the objective gradient. Use the objective's own gradient for FOC checks.
"""
function _re_logit_agh_score_info(theta::Vector{T}, y::Vector{T}, X_c::Matrix{T},
                                  unique_groups::Vector{Int},
                                  group_obs::Dict{Int,Vector{Int}},
                                  x_nodes::Vector{Float64},
                                  w_nodes::Vector{Float64}) where {T<:AbstractFloat}
    k = size(X_c, 2)
    n_params = k + 1
    log_sigma_u = clamp(theta[k+1], T(-340), T(340))
    inv_s2 = exp(-2 * log_sigma_u)
    beta = theta[1:k]
    nq = length(x_nodes)
    half_log2pi = T(0.5 * log(2π))
    log_sqrt2 = T(0.5 * log(2.0))
    G = length(unique_groups)

    score = zeros(T, n_params)
    info = zeros(T, n_params, n_params)
    group_scores = zeros(T, n_params, G)
    logvals = Vector{T}(undef, nq)
    a_nodes = Vector{T}(undef, nq)
    s_node = zeros(T, n_params)
    Es = zeros(T, n_params)
    Ess = zeros(T, n_params, n_params)
    EnegH = zeros(T, n_params, n_params)

    for (gi, g) in enumerate(unique_groups)
        idx = group_obs[g]
        X_g = @view X_c[idx, :]
        eta = X_g * beta
        yg = @view y[idx]

        alpha = _agh_group_mode(eta, yg, inv_s2)
        info_h = inv_s2
        @inbounds for j in eachindex(eta)
            mu = _agh_logistic(eta[j] + alpha)
            info_h += mu * (one(T) - mu)
        end
        sighat = one(T) / sqrt(max(info_h, eps(one(T))))

        @inbounds for q in 1:nq
            a = alpha + sqrt(T(2)) * sighat * T(x_nodes[q])
            a_nodes[q] = a
            logw = log_sqrt2 + log(sighat) + log(T(w_nodes[q])) + T(x_nodes[q])^2
            ll = zero(T)
            for j in eachindex(eta)
                z = eta[j] + a
                ll += yg[j] * z - _agh_log1pexp(z)
            end
            logphi = -half_log2pi - log_sigma_u - a^2 * inv_s2 / 2
            logvals[q] = logw + ll + logphi
        end
        mx = maximum(logvals)
        acc = zero(T)
        @inbounds for q in 1:nq
            acc += exp(logvals[q] - mx)
        end

        fill!(Es, zero(T)); fill!(Ess, zero(T)); fill!(EnegH, zero(T))
        @inbounds for q in 1:nq
            p_q = exp(logvals[q] - mx) / acc
            p_q > 0 || continue
            a = a_nodes[q]
            # complete-data score at (β, ℓ; α = a)
            fill!(s_node, zero(T))
            for j in eachindex(eta)
                mu = _agh_logistic(eta[j] + a)
                r = yg[j] - mu
                w = mu * (one(T) - mu)
                for c in 1:k
                    s_node[c] += r * X_g[j, c]
                    for c2 in c:k
                        EnegH[c, c2] += p_q * w * X_g[j, c] * X_g[j, c2]
                    end
                end
            end
            s_node[n_params] = a^2 * inv_s2 - one(T)
            EnegH[n_params, n_params] += p_q * 2 * a^2 * inv_s2
            for c in 1:n_params
                Es[c] += p_q * s_node[c]
                for c2 in c:n_params
                    Ess[c, c2] += p_q * s_node[c] * s_node[c2]
                end
            end
        end
        # I_g = E[−H_c] − (E[s s'] − E[s]E[s]')  (upper triangles accumulated)
        @inbounds for c in 1:n_params, c2 in c:n_params
            v = EnegH[c, c2] - (Ess[c, c2] - Es[c] * Es[c2])
            info[c, c2] += v
            c2 > c && (info[c2, c] += v)
        end
        score .+= Es
        group_scores[:, gi] .= Es
    end
    score, info, group_scores
end

"""
    _agh_newton_polish(nll, theta) -> (theta, gnorm)

Damped-Newton polish of the adaptive-GH marginal negative loglik around the LBFGS
minimizer. Optim's `f_reltol` stopping can halt with `‖∇nll‖` marginally above the
honest-convergence FOC threshold (the exact stopping point varies across Optim
versions); a few backtracking Newton steps on the exact ForwardDiff Hessian drive
the gradient to numerical zero whenever the solution is a genuine local optimum.

Returns the (possibly improved) parameter vector and the final gradient VECTOR.
The scale-aware FOC verdict lives in [`_agh_foc_stationary`](@ref), computed on
the Louis observed information rather than the AD Hessian (#542).
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
    return theta, g
end

"""
    _agh_foc_stationary(g, info, theta) -> Bool

Scale-aware first-order-condition check for stiff marginal-likelihood surfaces: `g`
is the gradient of the OBJECTIVE actually minimized (the AGH-n negative loglik) and
`info` the Louis observed information used as the curvature metric. With `I ≻ 0`,
`‖I⁻¹·g‖ ≤ √eps·(1 + ‖θ‖)` means the remaining Newton step is below parameter
resolution — θ IS the stationary point to machine precision even when the raw
gradient floats on rounding noise above an absolute cutoff (ddcg RE logit: the
objective value is O(1e3), so `eps(f)`-level noise in ∇ can exceed 1e-5). Do NOT
pass the Louis score here: at the AGH-n optimum the exact-marginal score is
O(quadrature error), not 0, so the check would never fire.
"""
function _agh_foc_stationary(g::Vector{T}, info::Matrix{T},
                             theta::Vector{T}) where {T<:AbstractFloat}
    F_c = cholesky(Hermitian((info .+ info') ./ 2); check=false)
    issuccess(F_c) || return false
    step = F_c \ g
    all(isfinite, step) && norm(step) <= sqrt(eps(T)) * (one(T) + norm(theta))
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
    theta, g_pol = _agh_newton_polish(nll, theta)
    gnorm = norm(g_pol)

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    loglik_final = _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, nodes, weights)

    # Louis score/information: one pass serves the honest-convergence FOC check and the
    # covariance (bread + cluster meat) without differentiating through the mode search.
    _, info_L, group_scores = _re_logit_agh_score_info(theta, y, X_c, unique_groups,
                                                       group_obs, nodes, weights)

    # Honest convergence: the true gradient norm is near zero, or the FOC holds in the
    # information metric (stiff surfaces float the raw gradient on rounding noise > 1e-5).
    converged = isfinite(gnorm) && (gnorm < T(1e-5) ||
                                    _agh_foc_stationary(g_pol, info_L, theta))
    if !converged
        @warn "RE logit did not reach a stationary point (‖∇nll‖=$gnorm after " *
              "$iterations LBFGS iterations + Newton polish). Raise maxiter/tol." maxlog=1
    end

    # Covariance: Louis observed-information bread, optionally sandwich-clustered by entity.
    n_params = k_full + 1
    vcov_full = _re_logit_vcov(info_L, group_scores, n, cov_type)
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

"""
    _re_logit_vcov(info, group_scores, n, cov_type)

Entity-clustered (or pure-information) VCE for the RE/CRE logit `theta = [β; log σ_u]`
from the Louis observed information and per-group marginal scores computed by
[`_re_logit_agh_score_info`](@ref). Bread = I⁻¹; under `cov_type == :cluster` (the
default) forms the group-score sandwich

    V = I⁻¹ (Σ_g s_g s_g') I⁻¹

carrying the pooled path's `G/(G−1)·(n−1)/(n−k)` finite-sample correction, so the dof
convention does not silently change between `:pooled` and `:re`/`:cre` (#542). This is
the standard panel-robust VCE (Stata `xtlogit, re vce(robust)`); the pure information
inverse overstates precision on moderately long panels.
"""
function _re_logit_vcov(info::Matrix{T}, group_scores::Matrix{T}, n::Int,
                        cov_type::Symbol) where {T<:AbstractFloat}
    n_params = size(info, 1)
    bread = Matrix{T}(robust_inv(Hermitian((info .+ info') ./ 2)))
    cov_type == :cluster || return Matrix{T}((bread .+ bread') ./ 2)

    meat = group_scores * group_scores'
    G = size(group_scores, 2)
    if G > 1
        meat .*= T(G) / T(G - 1) * T(n - 1) / T(max(n - n_params, 1))
    end
    V = bread * meat * bread
    Matrix{T}((V .+ V') ./ 2)
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
    theta, g_pol = _agh_newton_polish(nll, theta)
    gnorm = norm(g_pol)

    beta = theta[1:k_full]
    sigma_u = exp(theta[k_full + 1])
    # Recompute the loglik at the optimum (the old code reused loglik_old — a stale value).
    loglik_final = _re_logit_agh_loglik(theta, y, X_c, unique_groups, group_obs, nodes, weights)

    _, info_L, group_scores = _re_logit_agh_score_info(theta, y, X_c, unique_groups,
                                                       group_obs, nodes, weights)
    converged = isfinite(gnorm) && (gnorm < T(1e-5) ||
                                    _agh_foc_stationary(g_pol, info_L, theta))
    if !converged
        @warn "CRE logit did not reach a stationary point (‖∇nll‖=$gnorm after " *
              "$iterations LBFGS iterations + Newton polish). Raise maxiter/tol." maxlog=1
    end

    n_params = k_full + 1
    vcov_full = _re_logit_vcov(info_L, group_scores, n, cov_type)
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
