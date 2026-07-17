# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Penalized (regularized) cross-sectional regression: ridge, LASSO, elastic net,
adaptive LASSO (Zou 2006) and post-LASSO OLS refit (Belloni-Chernozhukov).

Ridge (`alpha = 0`) is solved in closed form; LASSO / elastic net (`alpha > 0`) use
cyclic coordinate descent with soft-thresholding, active-set cycling, and strong-rule
screening over a warm-started log-spaced lambda path. All fitting is done on
standardized regressors (unit variance) and a centered response following the `glmnet`
convention: the objective is

    (1/2n) * ||yc - Xs*beta||^2 + lambda * [ alpha*sum(w.*|beta|) + (1-alpha)/2*sum(w.*beta.^2) ]

so lambda values line up with R `glmnet` and Python `sklearn` (penalty scaled by n).
Coefficients are unstandardized before return and the (never-penalized) intercept is
recovered as `beta0 = ybar - xbar'beta`.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Soft-thresholding
# =============================================================================

"""`_soft(z, γ) = sign(z) * max(|z| - γ, 0)` — the LASSO soft-threshold operator."""
@inline function _soft(z::T, γ::T) where {T<:AbstractFloat}
    z > γ ? z - γ : (z < -γ ? z + γ : zero(T))
end

# =============================================================================
# Standardization
# =============================================================================

"""
    _standardize(X, y; standardize=true) -> (Xs, xbar, xsd, yc, ybar)

Center every column of `X` and scale to population unit variance (`1/n` normalization,
matching `glmnet`); center `y`. When `standardize=false`, columns are centered but not
scaled (`xsd .= 1`). Constant columns get `xsd = 1` (they contribute nothing after
centering). No intercept column may be present.
"""
function _standardize(X::AbstractMatrix{T}, y::AbstractVector{T};
                      standardize::Bool=true) where {T<:AbstractFloat}
    n, p = size(X)
    xbar = vec(mean(X; dims=1))
    Xs = X .- xbar'
    if standardize
        xsd = [sqrt(sum(abs2, @view Xs[:, j]) / n) for j in 1:p]
        for j in 1:p
            xsd[j] <= sqrt(eps(T)) && (xsd[j] = one(T))   # constant column guard
            @views Xs[:, j] ./= xsd[j]
        end
    else
        xsd = ones(T, p)
    end
    ybar = mean(y)
    yc = y .- ybar
    (Xs, xbar, xsd, yc, ybar)
end

# =============================================================================
# Lambda path
# =============================================================================

"""
    _lambda_max(Xs, yc, α, w) -> λ_max

Smallest lambda that zeros every coefficient: `max_j |x_j'yc| / (n * α * w_j)`. `α` is
floored at a small value so the (nearly-)ridge case still yields a finite cap.
"""
function _lambda_max(Xs::AbstractMatrix{T}, yc::AbstractVector{T},
                     α::T, w::AbstractVector{T}) where {T<:AbstractFloat}
    n = size(Xs, 1)
    αf = max(α, T(1e-3))
    λ = zero(T)
    @inbounds for j in 1:size(Xs, 2)
        g = abs(dot(@view(Xs[:, j]), yc)) / (n * αf * w[j])
        g > λ && (λ = g)
    end
    λ <= zero(T) ? one(T) : λ
end

"""Log-spaced descending lambda path from `λ_max` to `ε*λ_max`."""
function _lambda_path(λ_max::T, nlambda::Int, lambda_min_ratio::T) where {T<:AbstractFloat}
    nlambda == 1 && return [λ_max]
    lo = log(λ_max * lambda_min_ratio)
    hi = log(λ_max)
    T[exp(hi + (lo - hi) * (k - 1) / (nlambda - 1)) for k in 1:nlambda]
end

# =============================================================================
# Coordinate descent (single lambda, warm-started)
# =============================================================================

"""
    _cd_single!(β, r, Xs, λα, λ1, w; tol, maxit) -> β

In-place cyclic coordinate descent with soft-thresholding, active-set cycling, and
strong-rule / KKT-checked admission of new coordinates for a single lambda. `r` is the
working residual `yc - Xs*β` (kept in sync). Because columns have `(1/n)Σx_ij² = 1`,
the coordinate denominator is `1 + λ1*w_j`.
"""
function _cd_single!(β::Vector{T}, r::Vector{T}, Xs::AbstractMatrix{T},
                     λα::T, λ1::T, w::AbstractVector{T};
                     tol::T=T(1e-9), maxit::Int=100_000) where {T<:AbstractFloat}
    n, p = size(Xs)
    invn = one(T) / n
    active = [β[j] != zero(T) for j in 1:p]
    ktol = tol

    function update_coord!(j)
        xj = @view Xs[:, j]
        βj_old = β[j]
        z = invn * dot(xj, r) + βj_old         # (1/n)x_j'r_partial
        βj_new = _soft(z, λα * w[j]) / (one(T) + λ1 * w[j])
        Δ = βj_new - βj_old
        if Δ != zero(T)
            @inbounds @simd for i in 1:n
                r[i] -= xj[i] * Δ
            end
            β[j] = βj_new
        end
        abs(Δ)
    end

    outer = 0
    while outer < maxit
        outer += 1
        # ---- inner loop: cycle active set to convergence ----
        while true
            maxΔ = zero(T)
            @inbounds for j in 1:p
                active[j] || continue
                Δ = update_coord!(j)
                Δ > maxΔ && (maxΔ = Δ)
                β[j] == zero(T) && (active[j] = false)
            end
            maxΔ < tol && break
            outer += 1
            outer >= maxit && break
        end
        # ---- KKT / strong-rule sweep: admit violating coordinates ----
        changed = false
        @inbounds for j in 1:p
            active[j] && continue
            g = invn * dot(@view(Xs[:, j]), r)
            if abs(g) > λα * w[j] + ktol
                active[j] = true
                update_coord!(j)
                β[j] != zero(T) && (changed = true)
            end
        end
        changed || break
    end
    β
end

# =============================================================================
# Ridge (closed form)
# =============================================================================

"""
    _ridge_fit(Xs, yc, λ) -> (β_std, df)

Ridge on the standardized scale via `robust_inv(Hermitian(Xs'Xs + n*λ*I)) * Xs'yc`
(the `glmnet` `(1/2n)`-scaled normal equations). Effective degrees of freedom
`df = tr(Xs (Xs'Xs + n*λ*I)^{-1} Xs')`.
"""
function _ridge_fit(Xs::AbstractMatrix{T}, yc::AbstractVector{T}, λ::T) where {T<:AbstractFloat}
    n, p = size(Xs)
    XtX = Symmetric(Xs' * Xs)
    A = Matrix{T}(XtX) + (n * λ) * I
    Ainv = Matrix{T}(robust_inv(Hermitian(A)))
    β = Ainv * (Xs' * yc)
    # df = tr(Xs Ainv Xs') = tr(Ainv Xs'Xs)
    df = tr(Ainv * Matrix{T}(XtX))
    (β, df)
end

# =============================================================================
# Path fit (dispatch ridge vs coordinate descent)
# =============================================================================

"""
    _fit_path(Xs, yc, α, λpath, w; tol, maxit) -> (B, dfs)

Fit the full standardized-scale coefficient path `B` (`p × L`) and per-lambda degrees of
freedom `dfs`. `α == 0` ⇒ ridge closed form per lambda; `α > 0` ⇒ warm-started coordinate
descent (active-set df). `λ == 0` short-circuits to the exact least-squares solution.
"""
function _fit_path(Xs::AbstractMatrix{T}, yc::AbstractVector{T}, α::T,
                   λpath::AbstractVector{T}, w::AbstractVector{T};
                   tol::T=T(1e-9), maxit::Int=100_000) where {T<:AbstractFloat}
    n, p = size(Xs)
    L = length(λpath)
    B = zeros(T, p, L)
    dfs = zeros(T, L)
    if α == zero(T)
        for k in 1:L
            β, df = _ridge_fit(Xs, yc, λpath[k])
            B[:, k] = β
            dfs[k] = df
        end
        return (B, dfs)
    end
    β = zeros(T, p)
    r = copy(yc)
    for k in 1:L
        λ = λpath[k]
        if λ == zero(T)
            β = Xs \ yc                       # exact OLS at λ=0
            r .= yc .- Xs * β
        else
            _cd_single!(β, r, Xs, λ * α, λ * (one(T) - α), w; tol=tol, maxit=maxit)
        end
        B[:, k] = β
        dfs[k] = count(!iszero, β)
    end
    (B, dfs)
end

# =============================================================================
# Fit statistics for a natural-scale coefficient vector
# =============================================================================

"""Return `(β0, fitted, residuals, ssr)` on the original scale from a natural-scale slope
vector `β_nat`, using `X`, `y`, and the response mean `ybar` / column means `xbar`."""
function _natural_fit(X::AbstractMatrix{T}, y::AbstractVector{T}, β_nat::Vector{T},
                      xbar::Vector{T}, ybar::T) where {T<:AbstractFloat}
    β0 = ybar - dot(xbar, β_nat)
    fitted = X * β_nat .+ β0
    resid = y .- fitted
    ssr = sum(abs2, resid)
    (β0, fitted, resid, ssr)
end

"""Gaussian log-likelihood, AIC, BIC, EBIC for `ssr` with `k = df + 1` parameters
(intercept + active slopes). EBIC (Chen-Chen 2008) uses `γ=1`: `BIC + 2*log C(p, df)`."""
function _ic_stats(ssr::T, n::Int, df::Real, p::Int) where {T<:AbstractFloat}
    σ2 = ssr / n
    ll = -T(n) / 2 * (log(2 * T(π)) + log(max(σ2, floatmin(T))) + one(T))
    k = df + 1                                        # slopes df + intercept
    aic = -2 * ll + 2 * k
    bic = -2 * ll + log(T(n)) * k
    # EBIC extended penalty: 2γ log C(p, df), γ=1
    dfi = round(Int, df)
    lchoose = (dfi <= 0 || dfi >= p) ? zero(T) :
              T(loggamma(p + 1.0) - loggamma(dfi + 1.0) - loggamma(p - dfi + 1.0))
    ebic = bic + 2 * lchoose
    (ll, aic, bic, ebic)
end

# =============================================================================
# Cross-validation
# =============================================================================

"""Build CV fold assignments: `:kfold` uses shuffled random folds (seeded RNG for
reproducibility); `:timeseries` uses contiguous, non-shuffled blocks (no look-ahead)."""
function _cv_folds(n::Int, nfolds::Int, cv::Symbol, rng)
    if cv == :timeseries
        # contiguous blocks preserving time order
        edges = round.(Int, range(0, n; length=nfolds + 1))
        [collect((edges[f]+1):edges[f+1]) for f in 1:nfolds]
    else
        perm = randperm(rng, n)
        [perm[f:nfolds:end] for f in 1:nfolds]
    end
end

"""
    _cv_curve(...) -> (cv_mse, cv_se)

Mean and standard error of held-out MSE along the (fixed, full-data) lambda path via
`nfolds`-fold CV. Standardization and (for adaptive) weights are recomputed within each
training fold; predictions use the natural-scale coefficients so the intercept is honored.
"""
function _cv_curve(X::AbstractMatrix{T}, y::AbstractVector{T}, α::T,
                   λpath::Vector{T}, cv::Symbol, nfolds::Int,
                   adaptive::Bool, adaptive_γ::T, standardize::Bool,
                   tol::T, maxit::Int, rng) where {T<:AbstractFloat}
    n = length(y)
    L = length(λpath)
    folds = _cv_folds(n, nfolds, cv, rng)
    fold_mse = fill(T(NaN), nfolds, L)
    for (f, test_idx) in enumerate(folds)
        isempty(test_idx) && continue
        train_idx = setdiff(1:n, test_idx)
        length(train_idx) <= 1 && continue
        Xtr = X[train_idx, :]; ytr = y[train_idx]
        Xte = X[test_idx, :];  yte = y[test_idx]
        Xs, xbar, xsd, yc, ybar = _standardize(Xtr, ytr; standardize=standardize)
        w = _weights(Xs, yc, α, adaptive, adaptive_γ)
        B, _ = _fit_path(Xs, yc, α, λpath, w; tol=tol, maxit=maxit)
        for k in 1:L
            β_nat = B[:, k] ./ xsd
            β0 = ybar - dot(xbar, β_nat)
            pred = Xte * β_nat .+ β0
            fold_mse[f, k] = mean(abs2, yte .- pred)
        end
    end
    cv_mse = [mean(skipmissing_nan(@view fold_mse[:, k])) for k in 1:L]
    cv_se = [begin
                 v = collect(Iterators.filter(isfinite, @view fold_mse[:, k]))
                 length(v) > 1 ? std(v) / sqrt(length(v)) : zero(T)
             end for k in 1:L]
    (cv_mse, cv_se)
end

skipmissing_nan(v) = Iterators.filter(isfinite, v)

# =============================================================================
# Adaptive weights (Zou 2006)
# =============================================================================

"""Adaptive-LASSO penalty weights `ŵ_j = 1/|β̂_j^{init}|^γ` from a ridge first stage on the
standardized scale. Returns `ones` when `adaptive=false`."""
function _weights(Xs::AbstractMatrix{T}, yc::AbstractVector{T}, α::T,
                  adaptive::Bool, γ::T) where {T<:AbstractFloat}
    p = size(Xs, 2)
    adaptive || return ones(T, p)
    n = size(Xs, 1)
    # small-ridge first stage (well-defined for p>=n); scale-free enough for weights
    λ0 = T(1e-3) * (sum(abs2, yc) / n)
    β_init, _ = _ridge_fit(Xs, yc, λ0)
    w = similar(β_init)
    @inbounds for j in 1:p
        w[j] = one(T) / (abs(β_init[j]) + sqrt(eps(T)))^γ
    end
    w
end

# =============================================================================
# PenalizedRegModel construction from a chosen lambda index
# =============================================================================

function _build_penalized(X, y, α, λpath, B, dfs, xbar, xsd, ybar,
                          idx, select, cv_mse, cv_se, λ_min, λ_1se,
                          adaptive, post, standardize, varnames, cv, nfolds)
    T = eltype(y)
    n, p = size(X)
    β_std = B[:, idx]
    β_nat = β_std ./ xsd

    post_flag = post
    if post && α > zero(T)
        # Post-LASSO / post-elastic-net: OLS refit on the selected support.
        supp = findall(!iszero, β_nat)
        if !isempty(supp)
            Xsup = hcat(ones(T, n), X[:, supp])
            b = Xsup \ y
            β_nat = zeros(T, p)
            β_nat[supp] = b[2:end]
            # intercept recomputed below from centered form for consistency
        end
    end

    β0, fitted, resid, ssr = _natural_fit(X, y, β_nat, xbar, ybar)
    tss = sum(abs2, y .- ybar)
    r2 = tss > 0 ? one(T) - ssr / tss : zero(T)
    active = findall(!iszero, β_nat)
    df_star = α == zero(T) ? dfs[idx] : T(length(active))
    ll, aic, bic, ebic = _ic_stats(ssr, n, df_star, p)

    # full natural-scale coefficient path (for plotting)
    coef_path = B ./ xsd
    beta0_path = [ybar - dot(xbar, @view coef_path[:, k]) for k in 1:length(λpath)]

    PenalizedRegModel{T}(
        Vector{T}(y), Matrix{T}(X), β_nat, β0, β_std, α, λpath[idx],
        Vector{T}(λpath), Matrix{T}(coef_path), Vector{T}(beta0_path),
        active, Vector{T}(dfs), df_star,
        fitted, resid, ssr, tss, r2, ll, aic, bic, ebic,
        cv_mse, cv_se, λ_min, λ_1se,
        select, cv, nfolds, adaptive, post_flag, standardize,
        Vector{T}(xbar), Vector{T}(xsd), ybar,
        varnames)
end

# =============================================================================
# Main entry: estimate_elastic_net
# =============================================================================

"""
    estimate_elastic_net(y, X; alpha=1.0, lambda=:cv, select=:cv, ...) -> PenalizedRegModel

Elastic-net penalized linear regression. `alpha=1` is LASSO, `alpha=0` is ridge; general
`alpha ∈ [0,1]` mixes an ℓ1 and an ℓ2 penalty. `X` must **not** include an intercept column
— the intercept is fitted separately and never penalized.

# Arguments
- `y::AbstractVector` — response (length `n`).
- `X::AbstractMatrix` — regressors (`n × p`), no constant column.
- `alpha::Real=1.0` — elastic-net mixing parameter (`1`=LASSO, `0`=ridge).
- `lambda` — `:cv` (default; build a path and select by CV), a single `Real`, or an explicit
  `AbstractVector` of lambdas (a path). Values are on the `glmnet` `(1/2n)`-SSE scale.
- `nlambda::Int=100`, `lambda_min_ratio=1e-4` — auto lambda-path length and range.
- `select::Symbol=:cv` — `:cv`, `:aic`, `:bic`, or `:ebic` (ignored when a single lambda is given).
- `cv::Symbol=:kfold` — `:kfold` (shuffled folds) or `:timeseries` (contiguous, non-shuffled
  folds — the correct choice for serially dependent macro data).
- `nfolds::Int=10` — number of CV folds.
- `adaptive::Bool=false` — adaptive LASSO weights `1/|β̂ⱼ^{init}|^γ` (Zou 2006) from a ridge
  first stage.
- `adaptive_gamma::Real=1.0` — adaptive-weight exponent `γ`.
- `post::Bool=false` — post-selection OLS refit on the selected support (Belloni-Chernozhukov).
  **Honest-inference caveat:** naive post-selection standard errors are invalid; no t-stats or
  p-values are reported for penalized fits, and post-LASSO SEs are trustworthy only under the
  sparsity/beta-min conditions of the honest-inference literature — this function does not
  compute them.
- `standardize::Bool=true` — scale regressors to unit variance before penalizing.
- `varnames`, `seed`, `tol`, `maxit` — names, CV RNG seed, and CD tolerance / iteration cap.

# Returns
`PenalizedRegModel{T}` with natural-scale coefficients, intercept, the full lambda path and
coefficient-path matrix, active set, degrees of freedom, CV curve, and fit statistics.

# References
- Hoerl & Kennard (1970); Tibshirani (1996); Zou & Hastie (2005); Zou (2006);
  Friedman, Hastie & Tibshirani (2010); Belloni & Chernozhukov (2013).
"""
function estimate_elastic_net(y::AbstractVector{T}, X::AbstractMatrix{T};
                              alpha::Real=1.0,
                              lambda::Union{Symbol,Real,AbstractVector}=:cv,
                              nlambda::Int=100,
                              lambda_min_ratio::Real=1e-4,
                              select::Symbol=:cv,
                              cv::Symbol=:kfold,
                              nfolds::Int=10,
                              adaptive::Bool=false,
                              adaptive_gamma::Real=1.0,
                              post::Bool=false,
                              standardize::Bool=true,
                              varnames::Union{Nothing,Vector{String}}=nothing,
                              seed::Int=1234,
                              tol::Real=1e-9,
                              maxit::Int=100_000) where {T<:AbstractFloat}
    n, p = size(X)
    length(y) == n || throw(ArgumentError("y has length $(length(y)); X has $n rows"))
    n > 1 || throw(ArgumentError("need n > 1"))
    (0 <= alpha <= 1) || throw(ArgumentError("alpha must be in [0,1]; got $alpha"))
    select in (:cv, :aic, :bic, :ebic) ||
        throw(ArgumentError("select must be :cv, :aic, :bic, or :ebic; got :$select"))
    cv in (:kfold, :timeseries) ||
        throw(ArgumentError("cv must be :kfold or :timeseries; got :$cv"))
    α = T(alpha)
    vnames = varnames === nothing ? ["x$j" for j in 1:p] : varnames
    length(vnames) == p || throw(ArgumentError("varnames must have length $p"))

    # ---- standardize ----
    Xs, xbar, xsd, yc, ybar = _standardize(X, y; standardize=standardize)
    w = _weights(Xs, yc, α, adaptive, T(adaptive_gamma))

    # ---- lambda path ----
    λpath = if lambda isa Symbol
        _lambda_path(_lambda_max(Xs, yc, α, w), nlambda, T(lambda_min_ratio))
    elseif lambda isa Real
        T[T(lambda)]
    else
        pv = sort(T.(collect(lambda)); rev=true)  # descending for warm starts
        isempty(pv) && throw(ArgumentError("lambda vector is empty"))
        pv
    end

    # ---- fit full path ----
    B, dfs = _fit_path(Xs, yc, α, λpath, w; tol=T(tol), maxit=maxit)

    # ---- selection ----
    L = length(λpath)
    cv_mse = nothing; cv_se = nothing; λ_min = λpath[1]; λ_1se = λpath[1]
    use_cv = (select == :cv)      # :cv chooses CV; :aic/:bic/:ebic choose IC over the path
    if L == 1
        idx = 1
        sel = lambda isa Union{Real,AbstractVector} ? :fixed : select
    elseif use_cv
        rng = MersenneTwister(seed)
        cv_mse, cv_se = _cv_curve(X, y, α, λpath, cv, nfolds, adaptive,
                                  T(adaptive_gamma), standardize, T(tol), maxit, rng)
        imin = argmin(cv_mse)
        λ_min = λpath[imin]
        # 1-SE rule: largest lambda (most parsimonious) within 1 SE of the min
        thresh = cv_mse[imin] + cv_se[imin]
        i1se = imin
        for k in 1:imin                       # path descends, so earlier = larger lambda
            if cv_mse[k] <= thresh
                i1se = k
                break
            end
        end
        λ_1se = λpath[i1se]
        idx = imin
        sel = :cv
    else
        # IC selection over the path
        ics = zeros(T, L)
        for k in 1:L
            β_nat = B[:, k] ./ xsd
            _, _, _, ssr = _natural_fit(X, y, β_nat, xbar, ybar)
            df_k = α == zero(T) ? dfs[k] : count(!iszero, β_nat)
            ll, aic, bic, ebic = _ic_stats(ssr, n, df_k, p)
            ics[k] = select == :aic ? aic : select == :bic ? bic : ebic
        end
        idx = argmin(ics)
        sel = select
    end

    _build_penalized(X, y, α, λpath, B, dfs, xbar, xsd, ybar,
                     idx, sel, cv_mse, cv_se, λ_min, λ_1se,
                     adaptive, post, standardize, vnames, cv, nfolds)
end

# convenience: accept Integer/other element types
function estimate_elastic_net(y::AbstractVector, X::AbstractMatrix; kwargs...)
    estimate_elastic_net(float.(y), float.(X); kwargs...)
end

# =============================================================================
# Wrappers: LASSO (alpha=1) and ridge (alpha=0)
# =============================================================================

"""
    estimate_lasso(y, X; kwargs...) -> PenalizedRegModel

LASSO regression: `estimate_elastic_net(y, X; alpha=1, kwargs...)`. Coordinate descent with
soft-thresholding over a warm-started lambda path; sparse solutions. See
[`estimate_elastic_net`](@ref) for keyword arguments and the honest-inference caveat.
"""
estimate_lasso(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_elastic_net(y, X; alpha=1.0, kwargs...)

"""
    estimate_ridge(y, X; kwargs...) -> PenalizedRegModel

Ridge regression: `estimate_elastic_net(y, X; alpha=0, kwargs...)`. Closed-form
`(Xs'Xs + nλI)^{-1}Xs'yc` on the standardized scale (`glmnet` `(1/2n)` convention); dense
solutions with effective df `tr(Xs(Xs'Xs+nλI)^{-1}Xs')`. See [`estimate_elastic_net`](@ref).
"""
estimate_ridge(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_elastic_net(y, X; alpha=0.0, kwargs...)

# =============================================================================
# StatsAPI interface + predict
# =============================================================================

StatsAPI.coef(m::PenalizedRegModel) = m.beta
StatsAPI.residuals(m::PenalizedRegModel) = m.residuals
StatsAPI.predict(m::PenalizedRegModel) = m.fitted
StatsAPI.nobs(m::PenalizedRegModel) = length(m.y)
StatsAPI.dof(m::PenalizedRegModel) = m.df_star + 1
StatsAPI.loglikelihood(m::PenalizedRegModel) = m.loglik
StatsAPI.aic(m::PenalizedRegModel) = m.aic
StatsAPI.bic(m::PenalizedRegModel) = m.bic
StatsAPI.islinear(::PenalizedRegModel) = true
StatsAPI.r2(m::PenalizedRegModel) = m.r2

"""
    predict(m::PenalizedRegModel, Xnew) -> Vector

Predict on new data: `Xnew * β_natural .+ β0`. `Xnew` has the same `p` columns as the
training regressors (no intercept column).
"""
function StatsAPI.predict(m::PenalizedRegModel{T}, Xnew::AbstractMatrix) where {T}
    size(Xnew, 2) == length(m.beta) ||
        throw(ArgumentError("Xnew must have $(length(m.beta)) columns"))
    Matrix{T}(Xnew) * m.beta .+ m.beta0
end
