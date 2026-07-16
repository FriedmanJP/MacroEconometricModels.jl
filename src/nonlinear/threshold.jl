# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Two-regime threshold least squares, SETAR, Hansen (1996) linearity test and
Hansen (2000) threshold confidence interval.

References
- Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach.*
- Hansen, B. E. (1996). Inference when a nuisance parameter is not identified
  under the null hypothesis. *Econometrica* 64(2), 413–430.
- Hansen, B. E. (2000). Sample splitting and threshold estimation.
  *Econometrica* 68(3), 575–603.
"""

# =============================================================================
# OLS helpers
# =============================================================================

"""
    _ols_fit(y, X) -> (β, resid, ssr, XtX_inv)

Plain OLS via the normal equations with a robust inverse. Returns coefficients,
residuals, sum of squared residuals, and `(X'X)⁻¹`.
"""
function _ols_fit(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    XtX = Symmetric(X' * X)
    XtX_inv = robust_inv(XtX; silent=true)
    beta = XtX_inv * (X' * y)
    resid = y .- X * beta
    ssr = dot(resid, resid)
    return beta, resid, ssr, Matrix{T}(XtX_inv)
end

# =============================================================================
# Threshold grid over the order statistics of q
# =============================================================================

"""
    _threshold_grid(q, trim) -> Vector

Candidate threshold values: the sorted unique values of `q` lying strictly
between the `trim` and `1−trim` sample quantiles. Searching the order statistics
of `q` (rather than an even grid) hits every distinct sample split and never
wastes a point on a split that reproduces a neighbour.
"""
function _threshold_grid(q::AbstractVector{T}, trim::Real) where {T<:AbstractFloat}
    (0 < trim < 0.5) || throw(ArgumentError("trim must satisfy 0 < trim < 0.5; got $trim."))
    qs = sort(q)
    n = length(qs)
    lo = qs[max(1, floor(Int, trim * n))]
    hi = qs[min(n, ceil(Int, (1 - trim) * n))]
    cand = unique(qs)
    # Keep candidates in [lo, hi); a candidate γ splits into {q ≤ γ} and {q > γ},
    # so the largest value is dropped (it would put every observation in regime 1).
    grid = T[g for g in cand if lo <= g <= hi && g < qs[end]]
    isempty(grid) && throw(ArgumentError(
        "Empty threshold grid — increase the sample size or decrease `trim`."))
    return grid
end

# =============================================================================
# Concentrated SSR at a single threshold
# =============================================================================

"""
    _split_ssr(y, X, q, γ; min_obs) -> (ssr, mask1, ok)

Split the sample at `γ` (`mask1 = q .≤ γ`), fit both regime OLS problems, and
return the pooled SSR. `ok=false` flags a rank-deficient split (a regime with
fewer than `min_obs` observations), in which case `ssr = Inf`.
"""
function _split_ssr(y::AbstractVector{T}, X::AbstractMatrix{T}, q::AbstractVector{T},
                    gamma::T; min_obs::Int) where {T<:AbstractFloat}
    mask1 = q .<= gamma
    n1 = count(mask1)
    n2 = length(q) - n1
    if n1 < min_obs || n2 < min_obs
        return T(Inf), mask1, false
    end
    _, _, ssr1, _ = _ols_fit(y[mask1], X[mask1, :])
    mask2 = .!mask1
    _, _, ssr2, _ = _ols_fit(y[mask2], X[mask2, :])
    return ssr1 + ssr2, mask1, true
end

# =============================================================================
# Two-regime threshold least squares
# =============================================================================

"""
    estimate_threshold(y, X, q; trim=0.15, linearity=true, reps=1000,
                       ci_level=0.95, het=false, rng=Random.default_rng())

Estimate a two-regime threshold regression by conditional least squares.

The model is `yₜ = Xₜ'β₁·1{qₜ ≤ γ} + Xₜ'β₂·1{qₜ > γ} + uₜ`. The threshold γ is
chosen by grid search over the trimmed order statistics of `q`, minimising the
concentrated SSR `S(γ) = SSR₁(γ) + SSR₂(γ)`; each regime is then fit by OLS.

# Arguments
- `y::AbstractVector`: dependent variable.
- `X::AbstractMatrix`: `n × k` regressor matrix (include an intercept column).
- `q::AbstractVector`: threshold variable (length `n`).

# Keywords
- `trim`: fraction of extreme `q` values excluded from the γ grid (default `0.15`).
- `linearity`: run [`hansen_linearity_test`](@ref) and attach the result (default `true`).
- `reps`: bootstrap replications for the linearity test.
- `ci_level`: confidence level for the Hansen (2000) threshold CI (`0.90`/`0.95`/`0.99`).
- `het`: heteroskedasticity-correct the threshold CI (default `false`).
- `rng`: random number generator for the linearity bootstrap.

Returns a [`ThresholdModel`](@ref).
"""
function estimate_threshold(y::AbstractVector, X::AbstractMatrix, q::AbstractVector;
                            trim::Real=0.15, linearity::Bool=true, reps::Int=1000,
                            ci_level::Real=0.95, het::Bool=false,
                            rng::Random.AbstractRNG=Random.default_rng(),
                            xnames::Union{Nothing,Vector{String}}=nothing,
                            qname::String="q",
                            p::Int=0, d::Int=0, is_setar::Bool=false)
    T = float(promote_type(eltype(y), eltype(X), eltype(q)))
    yv = Vector{T}(y)
    Xm = Matrix{T}(X)
    qv = Vector{T}(q)
    n, k = size(Xm)
    (length(yv) == n == length(qv)) ||
        throw(DimensionMismatch("y, X rows and q must have matching length."))

    min_obs = k + 1
    grid = _threshold_grid(qv, trim)
    (n >= 2 * min_obs) || throw(ArgumentError(
        "Sample too small: need at least $(2*min_obs) observations for two $(k)-regressor regimes."))

    # Grid search for the SSR-minimising threshold.
    best_ssr = T(Inf)
    best_gamma = grid[1]
    for g in grid
        s, _, ok = _split_ssr(yv, Xm, qv, g; min_obs=min_obs)
        if ok && s < best_ssr
            best_ssr = s
            best_gamma = g
        end
    end
    isfinite(best_ssr) || throw(ArgumentError(
        "No admissible threshold split (every candidate leaves a rank-deficient regime); " *
        "decrease `trim` or supply more data."))

    # Per-regime OLS at γ̂.
    mask1 = qv .<= best_gamma
    mask2 = .!mask1
    n1 = count(mask1); n2 = count(mask2)
    beta1, resid1, ssr1, XtXinv1 = _ols_fit(yv[mask1], Xm[mask1, :])
    beta2, resid2, ssr2, XtXinv2 = _ols_fit(yv[mask2], Xm[mask2, :])

    # Classical per-regime standard errors.
    s2_1 = ssr1 / max(n1 - k, 1)
    s2_2 = ssr2 / max(n2 - k, 1)
    se1 = sqrt.(max.(s2_1 .* diag(XtXinv1), zero(T)))
    se2 = sqrt.(max.(s2_2 .* diag(XtXinv2), zero(T)))

    # Pooled residuals in the original observation order.
    resid = zeros(T, n)
    resid[mask1] .= resid1
    resid[mask2] .= resid2
    ssr = ssr1 + ssr2
    sigma2 = ssr / n

    # Information criteria: npar = 2k regime coefficients + 1 threshold.
    npar = 2 * k + 1
    loglik = -T(n) / 2 * (log(T(2π)) + log(sigma2) + one(T))
    aic = -2 * loglik + 2 * npar
    bic = -2 * loglik + log(T(n)) * npar

    # Hansen (2000) threshold confidence interval.
    gamma_ci = _hansen2000_ci(yv, Xm, qv, grid, best_gamma, ssr, beta1, beta2, resid;
                              level=ci_level, het=het)

    # Hansen (1996) linearity test.
    lin = linearity ?
        hansen_linearity_test(yv, Xm, qv; trim=T(trim), reps=reps, rng=rng) : nothing

    xn = xnames === nothing ? ["x$i" for i in 1:k] : xnames

    return ThresholdModel{T}(yv, Xm, qv, best_gamma, gamma_ci, T(ci_level),
        beta1, beta2, se1, se2, mask1, ssr1, ssr2, ssr, sigma2, resid,
        n, k, n1, n2, p, d, is_setar, aic, bic, xn, qname, T(trim), lin)
end

# =============================================================================
# SETAR
# =============================================================================

"""
    estimate_setar(y, p, d=1; trim=0.15, kwargs...)

Estimate a two-regime self-exciting threshold autoregression, SETAR(2; p, p).

Sets `qₜ = y_{t−d}` and `Xₜ = [1, y_{t−1}, …, y_{t−p}]`, then calls
[`estimate_threshold`](@ref). If `d` is a range (or `:auto`, meaning `1:p`), the
delay is selected jointly with γ by minimising the pooled SSR over the `(d, γ)`
grid; the selected `d` is stored in the returned model.

# Arguments
- `y::AbstractVector`: the series.
- `p::Int`: autoregressive order (per regime).
- `d`: threshold delay — an `Int`, an `AbstractRange`, or `:auto`.

Returns a [`ThresholdModel`](@ref) with `is_setar = true`.
"""
function estimate_setar(y::AbstractVector, p::Int, d=1;
                        trim::Real=0.15, linearity::Bool=true, reps::Int=1000,
                        ci_level::Real=0.95, het::Bool=false,
                        rng::Random.AbstractRNG=Random.default_rng())
    p >= 1 || throw(ArgumentError("SETAR order p must be ≥ 1; got $p."))
    Tf = float(eltype(y))
    yv = Vector{Tf}(y)

    delays = if d === :auto
        1:p
    elseif d isa AbstractRange
        d
    elseif d isa Integer
        d:d
    else
        throw(ArgumentError("d must be an Int, an AbstractRange, or :auto; got $(typeof(d))."))
    end
    all(dd -> dd >= 1, delays) || throw(ArgumentError("all delays must be ≥ 1."))

    # Select the delay by minimising the concentrated SSR of the threshold fit,
    # over a common effective sample (t > max(p, dmax)) so SSRs are comparable.
    m0 = max(p, maximum(delays))
    n_full = length(yv)
    (n_full > m0 + 2 * (p + 2)) || throw(ArgumentError(
        "Series too short for SETAR(p=$p): need more than $(m0 + 2*(p+2)) observations."))

    best_d = first(delays)
    best_ssr = Tf(Inf)
    if length(delays) > 1
        for dd in delays
            yy, XX, qq = _setar_design(yv, p, dd, m0)
            grid = _threshold_grid(qq, trim)
            min_obs = size(XX, 2) + 1
            s_min = Tf(Inf)
            for g in grid
                s, _, ok = _split_ssr(yy, XX, qq, g; min_obs=min_obs)
                ok && s < s_min && (s_min = s)
            end
            if s_min < best_ssr
                best_ssr = s_min
                best_d = dd
            end
        end
    else
        best_d = first(delays)
    end

    yy, XX, qq = _setar_design(yv, p, best_d, m0)
    xn = vcat("const", ["y[t-$i]" for i in 1:p])
    return estimate_threshold(yy, XX, qq; trim=trim, linearity=linearity, reps=reps,
                              ci_level=ci_level, het=het, rng=rng,
                              xnames=xn, qname="y[t-$best_d]",
                              p=p, d=best_d, is_setar=true)
end

"""
    _setar_design(y, p, d, m0) -> (y_eff, X, q)

Build the SETAR regression design over the effective sample `t = m0+1 … T`:
dependent `yₜ`, regressors `[1, y_{t−1}, …, y_{t−p}]`, threshold `y_{t−d}`.
"""
function _setar_design(y::AbstractVector{T}, p::Int, d::Int, m0::Int) where {T<:AbstractFloat}
    n = length(y)
    idx = (m0 + 1):n
    m = length(idx)
    y_eff = y[idx]
    X = Matrix{T}(undef, m, p + 1)
    X[:, 1] .= one(T)
    for j in 1:p
        X[:, j + 1] .= y[idx .- j]
    end
    q = y[idx .- d]
    return y_eff, X, q
end

# =============================================================================
# Hansen (1996) linearity test (fixed-regressor bootstrap)
# =============================================================================

"""
    hansen_linearity_test(y, X, q; trim=0.15, reps=1000, rng=Random.default_rng())

Hansen's (1996) sup-LM / sup-Wald test of linearity against a two-regime
threshold alternative, testing H₀: β₁ = β₂.

The heteroskedasticity-robust LM statistic at threshold γ is
`LM(γ) = S(γ)' V(γ)⁻¹ S(γ)`, where `S(γ) = Z(γ)'ê` is the score of the regime
interaction `Z(γ) = X ⊙ 1{q ≤ γ}` evaluated at the linear-model residuals `ê`,
and `V(γ)` is the White heteroskedasticity-robust score variance. The Wald
statistic is `n·(S₀ − S(γ))/S(γ)`, the split-sample F. Both are maximised over
the trimmed γ grid.

Because γ is unidentified under H₀ (the Davies problem), p-values are computed by
the **fixed-regressor bootstrap** of Hansen (1996): draw iid `N(0,1)` weights
`e*ₜ`, form the simulated score `S*(γ) = Z(γ)'(ê ⊙ e*)`, recompute the sup
statistics over the grid, and report the exceedance frequency across `reps`
replications.

Returns a [`HansenLinearityTest`](@ref).
"""
function hansen_linearity_test(y::AbstractVector, X::AbstractMatrix, q::AbstractVector;
                               trim::Real=0.15, reps::Int=1000,
                               rng::Random.AbstractRNG=Random.default_rng())
    T = float(promote_type(eltype(y), eltype(X), eltype(q)))
    yv = Vector{T}(y)
    Xm = Matrix{T}(X)
    qv = Vector{T}(q)
    n, k = size(Xm)

    # Restricted (linear) fit: residuals ê and restricted SSR S₀.
    _, resid0, ssr0, XtX_inv = _ols_fit(yv, Xm)

    grid = _threshold_grid(qv, trim)
    min_obs = k + 1

    # Residual maker components: M Z = Z − X (X'X)⁻¹ X'Z. Precompute X (X'X)⁻¹.
    XtXinvXt = XtX_inv * Xm'   # k × n

    # Precompute per-γ: orthogonalised interaction Z*(γ) = M Z(γ), robust V(γ)⁻¹.
    Zstar_list = Vector{Matrix{T}}(undef, length(grid))
    Vinv_list = Vector{Matrix{T}}(undef, length(grid))
    obs_lm = fill(T(-Inf), length(grid))
    obs_wald = fill(T(-Inf), length(grid))
    valid = falses(length(grid))

    for (gi, g) in enumerate(grid)
        mask1 = qv .<= g
        n1 = count(mask1)
        (n1 < min_obs || n - n1 < min_obs) && continue
        # Interaction regressors Z(γ) = X ⊙ 1{q ≤ γ}.
        Z = Xm .* T.(mask1)
        Zstar = Z .- Xm * (XtXinvXt * Z)   # M Z(γ), n × k
        Zstar_list[gi] = Zstar
        # Robust score variance V(γ) = Σₜ (z*ₜ êₜ)(z*ₜ êₜ)'.
        Ze = Zstar .* resid0               # n × k
        V = Symmetric(Ze' * Ze)
        Vinv = Matrix{T}(robust_inv(V; silent=true))
        Vinv_list[gi] = Vinv
        # Observed LM(γ).
        score = Zstar' * resid0            # = Z(γ)'ê  (since ê ⟂ X)
        obs_lm[gi] = dot(score, Vinv * score)
        # Observed Wald(γ) via the split-sample SSR.
        s, _, ok = _split_ssr(yv, Xm, qv, g; min_obs=min_obs)
        obs_wald[gi] = ok ? T(n) * (ssr0 - s) / s : T(-Inf)
        valid[gi] = true
    end

    any(valid) || throw(ArgumentError("No admissible threshold split for the linearity test."))
    sup_lm = maximum(obs_lm)
    sup_wald = maximum(obs_wald[valid])
    gamma_sup = grid[argmax(obs_lm)]

    # Fixed-regressor bootstrap. The bootstrap replaces the errors by ê ⊙ e*,
    # e* ~ N(0,1) iid, holding X, q and the weighting V(γ) fixed (Hansen 1996).
    exceed_lm = 0
    exceed_wald = 0
    _suppress_warnings() do
        for _ in 1:reps
            estar = randn(rng, T, n)
            u = resid0 .* estar
            sup_lm_b = T(-Inf)
            sup_wald_b = T(-Inf)
            # Restricted bootstrap SSR: S₀* = ||M u||² = u'u − (X'u)'(X'X)⁻¹(X'u).
            Xtu = Xm' * u
            ssr0_b = dot(u, u) - dot(Xtu, XtX_inv * Xtu)
            for gi in eachindex(grid)
                valid[gi] || continue
                score = Zstar_list[gi]' * u
                lm_b = dot(score, Vinv_list[gi] * score)
                lm_b > sup_lm_b && (sup_lm_b = lm_b)
                # Bootstrap Wald: refit split model on the simulated errors u.
                g = grid[gi]
                mask1 = qv .<= g
                _, _, s1, _ = _ols_fit(u[mask1], Xm[mask1, :])
                mask2 = .!mask1
                _, _, s2, _ = _ols_fit(u[mask2], Xm[mask2, :])
                sb = s1 + s2
                wald_b = T(n) * (ssr0_b - sb) / sb
                wald_b > sup_wald_b && (sup_wald_b = wald_b)
            end
            sup_lm_b >= sup_lm && (exceed_lm += 1)
            sup_wald_b >= sup_wald && (exceed_wald += 1)
        end
    end

    pval_lm = T(exceed_lm) / T(reps)
    pval_wald = T(exceed_wald) / T(reps)
    return HansenLinearityTest{T}(sup_lm, sup_wald, pval_lm, pval_wald, gamma_sup,
                                  reps, T(trim), count(valid))
end

# =============================================================================
# Hansen (2000) threshold confidence interval
# =============================================================================

"""
    _hansen2000_ci(y, X, q, grid, γ̂, S_hat, β₁, β₂, resid; level, het) -> (lo, hi)

Confidence interval for the threshold γ by inverting the likelihood-ratio
statistic `LR(γ) = n·(S(γ) − S(γ̂))/S(γ̂)` (Hansen 2000). The no-rejection region
is `{γ : LR(γ) ≤ c(level)}` with the tabulated non-standard critical values
`c(.90)=5.94`, `c(.95)=7.35`, `c(.99)=10.59`.

With `het=true` the statistic is scaled by an estimate `η̂²` of the
heteroskedasticity ratio at the threshold (Hansen 2000 §3.4), evaluated with a
Gaussian kernel in `q`; under homoskedasticity `η̂² ≈ 1`.
"""
function _hansen2000_ci(y::AbstractVector{T}, X::AbstractMatrix{T}, q::AbstractVector{T},
                        grid::AbstractVector{T}, gamma_hat::T, S_hat::T,
                        beta1::AbstractVector{T}, beta2::AbstractVector{T},
                        resid::AbstractVector{T};
                        level::Real=0.95, het::Bool=false) where {T<:AbstractFloat}
    n = length(y)
    crit = T(_hansen2000_crit(level))
    min_obs = size(X, 2) + 1

    # Optional heteroskedasticity scaling η̂² at the threshold.
    eta2 = one(T)
    if het
        eta2 = _hansen_eta2(X, q, gamma_hat, beta1, beta2, resid, S_hat / n)
    end

    lo = gamma_hat
    hi = gamma_hat
    for g in grid
        s, _, ok = _split_ssr(y, X, q, g; min_obs=min_obs)
        ok || continue
        lr = T(n) * (s - S_hat) / S_hat / eta2
        if lr <= crit
            g < lo && (lo = g)
            g > hi && (hi = g)
        end
    end
    return (lo, hi)
end

"""
    _hansen_eta2(X, q, γ̂, β₁, β₂, resid, σ̂²) -> η̂²

Heteroskedasticity ratio `η² = (δ'V₁δ)/(σ² δ'Vδ)` (Hansen 2000, §3.4), with
`δ = β₁ − β₂`, `V = E[xx'|q=γ]`, `V₁ = E[xx'e²|q=γ]`, estimated by Gaussian-kernel
smoothing in `q` at `γ̂` (Silverman bandwidth). Equals 1 under homoskedasticity.
"""
function _hansen_eta2(X::AbstractMatrix{T}, q::AbstractVector{T}, gamma_hat::T,
                      beta1::AbstractVector{T}, beta2::AbstractVector{T},
                      resid::AbstractVector{T}, sigma2::T) where {T<:AbstractFloat}
    n = length(q)
    delta = beta1 .- beta2
    sq = std(q)
    h = T(1.06) * sq * T(n)^(-T(1) / T(5))   # Silverman rule of thumb
    h <= 0 && return one(T)
    u = (q .- gamma_hat) ./ h
    w = exp.(-u .^ 2 ./ 2)                    # unnormalised Gaussian kernel weights
    sw = sum(w)
    sw <= 0 && return one(T)
    # δ'V δ  and  δ'V₁ δ  as kernel-weighted averages of (x'δ)² and (x'δ)² e².
    xd = X * delta                            # n-vector
    num = zero(T); den = zero(T)
    @inbounds for t in 1:n
        c = w[t] * xd[t]^2
        den += c
        num += c * resid[t]^2
    end
    den <= 0 && return one(T)
    eta2 = (num / den) / sigma2
    return isfinite(eta2) && eta2 > 0 ? eta2 : one(T)
end

# =============================================================================
# Forecast (bootstrap simulation of a SETAR model)
# =============================================================================

"""
    forecast(m::ThresholdModel, h; reps=1000, level=0.90, rng=Random.default_rng())

Multi-step bootstrap-simulation forecast of a SETAR [`ThresholdModel`](@ref).

Iterates the fitted piecewise-linear model forward `h` steps, resampling
residuals **within the realised regime** at each step, over `reps` simulated
paths. Returns a [`ThresholdForecast`](@ref) with the mean path, per-horizon
standard deviations, and central `level` percentile bands. Only SETAR models are
supported (a generic threshold model would require future exogenous `X`/`q`).
"""
function forecast(m::ThresholdModel{T}, h::Int; reps::Int=1000, level::Real=0.90,
                  rng::Random.AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    m.is_setar || throw(ArgumentError(
        "forecast is only defined for SETAR models (from estimate_setar)."))
    h >= 1 || throw(ArgumentError("horizon h must be ≥ 1."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))

    p = m.p; d = m.d
    # Reconstruct the full observed series: X[:,1] is the constant, X[:,2] = y[t-1],
    # so the initial condition for simulation is the last p observed levels plus the
    # regressand tail. The regressand m.y already holds y over the effective sample.
    # Build history = [ earliest lags … , m.y ].
    hist0 = vcat(m.X[1, 2:end][end:-1:1], m.y)   # length = p + n

    reg1_resid = m.residuals[m.regime]
    reg2_resid = m.residuals[.!m.regime]
    isempty(reg1_resid) && (reg1_resid = m.residuals)
    isempty(reg2_resid) && (reg2_resid = m.residuals)

    paths = Matrix{T}(undef, reps, h)
    for r in 1:reps
        hist = copy(hist0)
        for step in 1:h
            L = length(hist)
            qval = hist[L + 1 - d]
            beta = qval <= m.gamma ? m.beta1 : m.beta2
            resids = qval <= m.gamma ? reg1_resid : reg2_resid
            xt = Vector{T}(undef, p + 1)
            xt[1] = one(T)
            for j in 1:p
                xt[j + 1] = hist[L + 1 - j]
            end
            yhat = dot(xt, beta) + resids[rand(rng, 1:length(resids))]
            push!(hist, yhat)
            paths[r, step] = yhat
        end
    end

    alpha = (1 - level) / 2
    fmean = vec(mean(paths; dims=1))
    fse = vec(std(paths; dims=1))
    lo = T[Statistics.quantile(view(paths, :, s), alpha) for s in 1:h]
    hi = T[Statistics.quantile(view(paths, :, s), 1 - alpha) for s in 1:h]
    return ThresholdForecast{T}(fmean, lo, hi, fse, h, T(level), reps)
end
