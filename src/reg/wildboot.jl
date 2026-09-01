# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Wild cluster bootstrap for few-cluster inference (T243 / #342).

Cluster-robust standard errors are justified asymptotically in the number of clusters
`G`. With few clusters — the common difference-in-differences / policy-evaluation case —
the cluster-robust `t` over-rejects badly. The remedy is the **wild cluster restricted
(WCR) bootstrap** of Cameron, Gelbach & Miller (2008), as implemented in Stata `boottest`
and R `fwildclusterboot`.

Provides:
- `WildClusterBootstrap` — the result container
- `wild_cluster_bootstrap(model, coefficient, value; ...)` — the test and its inverted CI

References:
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-Based Improvements for
  Inference with Clustered Errors. *Review of Economics and Statistics*, 90(3), 414-427.
- MacKinnon, J. G. & Webb, M. D. (2018). The Wild Bootstrap for Few (Treated) Clusters.
  *The Econometrics Journal*, 21(2), 114-135.
- Roodman, D., MacKinnon, J. G., Nielsen, M. Ø. & Webb, M. D. (2019). Fast and Wild:
  Bootstrap Inference in Stata Using boottest. *The Stata Journal*, 19(1), 4-60.
"""

using LinearAlgebra
using Random
using Statistics

# =============================================================================
# Result container
# =============================================================================

"""
    WildClusterBootstrap{T}

Wild cluster bootstrap test of a single linear restriction, with the confidence interval
obtained by inverting the test.

# Fields
- `coefname::String` — name of the tested coefficient
- `coefindex::Int` — its column index in the design
- `estimate::T` — the unrestricted point estimate ``\\hat\\beta_j``
- `null_value::T` — the hypothesized value ``r`` in ``H_0: \\beta_j = r``
- `t_stat::T` — observed cluster-robust `t` statistic for the restriction
- `p_value::T` — symmetric bootstrap p-value, ``P(|t^*| \\ge |t_{obs}|)``
- `p_value_equaltail::T` — equal-tail bootstrap p-value
- `p_value_asymptotic::T` — the cluster-robust normal-approximation p-value, for comparison
- `ci_lower::T` / `ci_upper::T` — bootstrap confidence interval (`NaN` when `ci=false`)
- `level::T` — CI coverage
- `t_boot::Vector{T}` — the bootstrap `t` distribution at the null
- `n_boot::Int` — number of bootstrap replications actually used
- `n_clusters::Int` — number of clusters
- `weighttype::Symbol` — `:rademacher` or `:webb`
- `imposenull::Bool` — whether the null was imposed (WCR) or not (WCU)
- `enumerated::Bool` — whether all ``2^G`` Rademacher sign vectors were enumerated exactly
"""
struct WildClusterBootstrap{T<:AbstractFloat}
    coefname::String
    coefindex::Int
    estimate::T
    null_value::T
    t_stat::T
    p_value::T
    p_value_equaltail::T
    p_value_asymptotic::T
    ci_lower::T
    ci_upper::T
    level::T
    t_boot::Vector{T}
    n_boot::Int
    n_clusters::Int
    weighttype::Symbol
    imposenull::Bool
    enumerated::Bool
    manifest::Union{ReproManifest,Nothing}
end

WildClusterBootstrap{T}(coefname, coefindex, estimate, null_value, t_stat, p_value,
                        p_value_equaltail, p_value_asymptotic, ci_lower, ci_upper, level,
                        t_boot, n_boot, n_clusters, weighttype, imposenull, enumerated;
                        manifest=nothing) where {T<:AbstractFloat} =
    WildClusterBootstrap{T}(coefname, coefindex, estimate, null_value, t_stat, p_value,
                            p_value_equaltail, p_value_asymptotic, ci_lower, ci_upper, level,
                            t_boot, n_boot, n_clusters, weighttype, imposenull, enumerated,
                            manifest)

function Base.show(io::IO, b::WildClusterBootstrap{T}) where {T}
    spec = Any[
        "Coefficient"       b.coefname;
        "H₀"                "β = $(_fmt(b.null_value))";
        "Clusters"          b.n_clusters;
        "Replications"      b.n_boot;
        "Weights"           string(b.weighttype);
        "Null imposed"      b.imposenull ? "yes (WCR)" : "no (WCU)";
        "Enumerated"        b.enumerated ? "yes (exact 2^G)" : "no"
    ]
    _pretty_table(io, spec;
        title = "Wild Cluster Bootstrap (Cameron-Gelbach-Miller)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    ci_pct = round(Int, 100 * b.level)
    res = Any[
        "Estimate"                    _fmt(b.estimate);
        "t statistic"                 _fmt(b.t_stat);
        "p (bootstrap, symmetric)"    _fmt(b.p_value; digits=4);
        "p (bootstrap, equal-tail)"   _fmt(b.p_value_equaltail; digits=4);
        "p (cluster-robust normal)"   _fmt(b.p_value_asymptotic; digits=4);
        "$(ci_pct)% CI"               isnan(b.ci_lower) ? "—" :
                                      "[$(_fmt(b.ci_lower)), $(_fmt(b.ci_upper))]"
    ]
    _pretty_table(io, res;
        column_labels = ["Result", "Value"],
        alignment = [:l, :r],
    )
    return nothing
end

# =============================================================================
# Bootstrap weights
# =============================================================================

"""
    _wcb_weight_draw!(v, weighttype, rng)

Fill `v` with one bootstrap weight per cluster. `:rademacher` draws ``\\pm 1`` with equal
probability; `:webb` draws from the 6-point distribution
``\\{\\pm\\sqrt{1/2}, \\pm 1, \\pm\\sqrt{3/2}\\}`` with equal probability, which has more
support points and is preferred when `G` is very small (MacKinnon & Webb 2018).
"""
function _wcb_weight_draw!(v::Vector{T}, weighttype::Symbol, rng::AbstractRNG) where {T}
    if weighttype === :rademacher
        for i in eachindex(v)
            v[i] = rand(rng, Bool) ? one(T) : -one(T)
        end
    else  # :webb
        pts = T[-sqrt(T(3) / 2), -one(T), -sqrt(T(1) / 2),
                sqrt(T(1) / 2), one(T), sqrt(T(3) / 2)]
        for i in eachindex(v)
            v[i] = pts[rand(rng, 1:6)]
        end
    end
    return v
end

"""
    _wcb_weight_matrix(G, n_boot, weighttype, rng) -> (V, enumerated)

Build the `G × B` matrix of cluster weights, shared across every null value evaluated so
the inverted-test p-value function is smooth in the null (the `boottest` convention).

With Rademacher weights and `2^G ≤ n_boot` (and `G ≤ 20`), all `2^G` sign vectors are
**enumerated exactly** instead of drawn, which removes bootstrap simulation error entirely.
Pass `enumerate=false` to force random draws even when enumeration is available, or
`enumerate=true` to require it.
"""
function _wcb_weight_matrix(G::Int, n_boot::Int, weighttype::Symbol,
                            rng::AbstractRNG, ::Type{T};
                            enumerate::Union{Nothing,Bool}=nothing) where {T<:AbstractFloat}
    can_enumerate = weighttype === :rademacher && G <= 20 && 2^G <= n_boot
    want_enumerate = enumerate === nothing ? can_enumerate : enumerate
    if want_enumerate && !can_enumerate
        throw(ArgumentError(
            "enumerate=true requires Rademacher weights, G ≤ 20 and 2^G ≤ n_boot " *
            "(G=$G, n_boot=$n_boot, weights=:$weighttype)"))
    end
    if want_enumerate
        B = 2^G
        V = Matrix{T}(undef, G, B)
        for b in 0:(B-1)
            for g in 1:G
                V[g, b+1] = ((b >> (g - 1)) & 1) == 1 ? -one(T) : one(T)
            end
        end
        return V, true
    end
    V = Matrix{T}(undef, G, n_boot)
    col = Vector{T}(undef, G)
    for b in 1:n_boot
        _wcb_weight_draw!(col, weighttype, rng)
        @views V[:, b] .= col
    end
    return V, false
end

# =============================================================================
# Core kernel
# =============================================================================

"""
    _wcb_cluster_t(X, resid, XtXinv, cluster_idx, j) -> T

Cluster-robust `t` denominator: the standard error of coefficient `j` under the Arellano
(1987) sandwich with the usual `G/(G-1) · (n-1)/(n-k)` correction, matching
`_cluster_vcov`. `cluster_idx` is the precomputed per-cluster row-index vector.
"""
function _wcb_cluster_se(X::Matrix{T}, resid::Vector{T}, XtXinv::Matrix{T},
                         cluster_idx::Vector{Vector{Int}}, j::Int) where {T<:AbstractFloat}
    n, k = size(X)
    G = length(cluster_idx)
    B = zeros(T, k, k)
    score = Vector{T}(undef, k)
    for idx in cluster_idx
        fill!(score, zero(T))
        @inbounds for i in idx
            ri = resid[i]
            for c in 1:k
                score[c] += X[i, c] * ri
            end
        end
        BLAS.ger!(one(T), score, score, B)
    end
    B .*= T(G) / T(G - 1) * T(n - 1) / T(n - k)
    v = dot(@view(XtXinv[j, :]), B * @view(XtXinv[:, j]))
    return sqrt(max(v, zero(T)))
end

"""
    _wcb_run(y, X, cluster_idx, j, r0, V; imposenull) -> (t_obs, t_boot)

The wild cluster bootstrap kernel for `H₀: βⱼ = r0` on a fixed design.

With `imposenull=true` (WCR, the default and the recommended variant) the bootstrap data
generating process uses the **restricted** fit `β̃` — OLS subject to `βⱼ = r0` — and its
residuals `ẽ`. Each replication forms `y* = Xβ̃ + v_{g(i)} ẽᵢ` and recomputes the `t`
statistic for the same null. With `imposenull=false` (WCU) the unrestricted fit is used.

Since the design is fixed, `A = (X'X)⁻¹X'` is formed once and every replication is a pair
of `O(nk)` products; no factorization is repeated.
"""
function _wcb_run(y::Vector{T}, X::Matrix{T}, cluster_idx::Vector{Vector{Int}},
                  j::Int, r0::T, V::Matrix{T};
                  imposenull::Bool=true) where {T<:AbstractFloat}
    n, k = size(X)
    XtX = X' * X
    # robust_inv(Symmetric(...)) returns a Symmetric wrapper; the kernels take Matrix{T}
    XtXinv = Matrix{T}(robust_inv(Symmetric(XtX)))
    A = XtXinv * X'                       # k × n
    beta = A * y
    resid = y .- X * beta

    t_obs = (beta[j] - r0) / _wcb_cluster_se(X, resid, XtXinv, cluster_idx, j)

    # Restricted least squares: β̃ = β̂ − (X'X)⁻¹Rᵀ[R(X'X)⁻¹Rᵀ]⁻¹(Rβ̂ − r), R = eⱼᵀ
    beta_dgp = copy(beta)
    if imposenull
        adj = (beta[j] - r0) / XtXinv[j, j]
        @inbounds for c in 1:k
            beta_dgp[c] -= XtXinv[c, j] * adj
        end
    end
    fit_dgp = X * beta_dgp
    resid_dgp = y .- fit_dgp

    B = size(V, 2)
    t_boot = Vector{T}(undef, B)
    ystar = Vector{T}(undef, n)
    rstar = Vector{T}(undef, n)
    bstar = Vector{T}(undef, k)
    for b in 1:B
        @inbounds for (g, idx) in enumerate(cluster_idx)
            w = V[g, b]
            for i in idx
                ystar[i] = fit_dgp[i] + w * resid_dgp[i]
            end
        end
        mul!(bstar, A, ystar)
        mul!(rstar, X, bstar)
        @inbounds for i in 1:n
            rstar[i] = ystar[i] - rstar[i]
        end
        se_b = _wcb_cluster_se(X, rstar, XtXinv, cluster_idx, j)
        # Under WCR the bootstrap DGP satisfies βⱼ = r0, so the bootstrap statistic is
        # centered at r0. Under WCU it must be recentered on the sample estimate.
        center = imposenull ? r0 : beta[j]
        t_boot[b] = se_b > 0 ? (bstar[j] - center) / se_b : T(NaN)
    end
    return t_obs, t_boot
end

"""
    _wcb_pvalues(t_obs, t_boot) -> (p_symmetric, p_equaltail)

Symmetric and equal-tail bootstrap p-values.

The comparison uses a **relative** tolerance rather than an absolute one because the wild
cluster bootstrap has exact ties by construction: the all-`+1` sign vector reproduces the
sample (`y* = Xβ̃ + ẽ = y` under WCR), so one bootstrap statistic equals `t_obs` and, since
`t*(-v) = -t*(v)`, its negation equals `-t_obs`. Both belong in the count. An absolute
`eps` threshold drops or keeps that pair depending on rounding in how `β*` was formed,
which shifts the p-value by `2/(B+1)`.
"""
function _wcb_pvalues(t_obs::T, t_boot::Vector{T}) where {T<:AbstractFloat}
    finite = filter(isfinite, t_boot)
    B = length(finite)
    B == 0 && return (T(NaN), T(NaN))
    tol = max(abs(t_obs), one(T)) * T(1e-9)
    p_sym = (1 + count(t -> abs(t) >= abs(t_obs) - tol, finite)) / (B + 1)
    p_left = (1 + count(t -> t <= t_obs + tol, finite)) / (B + 1)
    p_right = (1 + count(t -> t >= t_obs - tol, finite)) / (B + 1)
    p_et = min(one(T), 2 * min(p_left, p_right))
    return (T(p_sym), T(p_et))
end

"""
    _wcb_invert(y, X, cluster_idx, j, V, level; imposenull, gridpoints, se_hat, beta_hat)

Invert the bootstrap test to get a confidence interval: find the null values `r₀` at which
the symmetric bootstrap p-value crosses `1 − level`. A coarse grid spanning `β̂ ± 6·se`
brackets each crossing, then 30 bisection steps refine it. The same weight matrix `V` is
reused at every `r₀` so the p-value function is smooth and the crossings well defined.

Returns `(lower, upper)`, with `NaN` on a side whose crossing is not bracketed by the grid.
"""
function _wcb_invert(y::Vector{T}, X::Matrix{T}, cluster_idx::Vector{Vector{Int}},
                     j::Int, V::Matrix{T}, level::T;
                     imposenull::Bool, gridpoints::Int,
                     se_hat::T, beta_hat::T) where {T<:AbstractFloat}
    alpha = one(T) - level
    pfun = r0 -> begin
        t_obs, t_boot = _wcb_run(y, X, cluster_idx, j, r0, V; imposenull=imposenull)
        _wcb_pvalues(t_obs, t_boot)[1]
    end

    isfinite(se_hat) && se_hat > 0 || return (T(NaN), T(NaN))
    lo_end = beta_hat - 6 * se_hat
    hi_end = beta_hat + 6 * se_hat
    grid = collect(range(lo_end, hi_end; length=max(gridpoints, 5)))
    pv = [pfun(g) for g in grid]

    inside = findall(>=(alpha), pv)
    isempty(inside) && return (T(NaN), T(NaN))

    function bisect(a::T, b::T)
        # p(a) < alpha ≤ p(b); shrink toward the crossing
        for _ in 1:30
            m = (a + b) / 2
            if pfun(m) >= alpha
                b = m
            else
                a = m
            end
        end
        return b
    end

    i_lo, i_hi = first(inside), last(inside)
    lower = i_lo == 1 ? T(NaN) : bisect(grid[i_lo-1], grid[i_lo])
    upper = i_hi == length(grid) ? T(NaN) : bisect(grid[i_hi+1], grid[i_hi])
    return (lower, upper)
end

# =============================================================================
# Public API
# =============================================================================

"""
    wild_cluster_bootstrap(model, coefficient, null_value=0.0; clusters, kwargs...)
        -> WildClusterBootstrap

Wild cluster bootstrap test of `H₀: βⱼ = null_value` with the confidence interval obtained
by inverting the test — the few-cluster inference procedure of Cameron, Gelbach & Miller
(2008), matching Stata `boottest`.

The default is the **restricted** (WCR) variant: the bootstrap data generating process is
built from the fit that imposes the null, `y* = Xβ̃ + v_{g(i)} ẽᵢ`, with one weight `v_g`
per cluster. Cameron–Gelbach–Miller and MacKinnon–Webb both show WCR dominates the
unrestricted WCU variant, which is available via `imposenull=false` but is not the default.

With Rademacher weights and `2^G ≤ n_boot`, all `2^G` sign vectors are enumerated exactly,
so the test carries no bootstrap simulation error at all.

# Arguments
- `model` — a fitted [`RegModel`](@ref) or [`PanelRegModel`](@ref) (`model=:fe`,
  including HDFE fits from `absorb=`, which are re-absorbed with the fit's own
  dimensions and tolerance)
- `coefficient` — the coefficient to test: a column index, name `String`, or `Symbol`
- `null_value::Real=0.0` — the hypothesized value `r` in `H₀: βⱼ = r`

# Keywords
- `clusters::AbstractVector` — cluster assignment per observation. Required for `RegModel`;
  defaults to the panel entity ids for `PanelRegModel`
- `n_boot::Int=999` — bootstrap replications (ignored when the sign space is enumerated)
- `weights::Symbol=:rademacher` — `:rademacher` or `:webb` (6-point; more support points
  when `G` is very small)
- `imposenull::Bool=true` — impose the null (WCR); `false` gives WCU
- `ci::Bool=true` — compute the inverted-test confidence interval
- `level::Real=0.95` — CI coverage
- `ci_gridpoints::Int=25` — grid used to bracket the CI crossings
- `enumerate::Union{Nothing,Bool}=nothing` — force (`true`) or forbid (`false`) exact
  enumeration of the `2^G` Rademacher sign vectors; `nothing` enumerates whenever possible
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
[`WildClusterBootstrap`](@ref).

# References
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). *Review of Economics and
  Statistics*, 90(3), 414-427.
- Roodman, D., MacKinnon, J. G., Nielsen, M. Ø. & Webb, M. D. (2019). *The Stata Journal*,
  19(1), 4-60.
"""
function wild_cluster_bootstrap(model::RegModel{T}, coefficient, null_value::Real=0.0;
                                clusters::Union{Nothing,AbstractVector}=nothing,
                                kwargs...) where {T<:AbstractFloat}
    clusters === nothing && throw(ArgumentError(
        "clusters is required for a RegModel — pass the same cluster vector used for the " *
        "cluster-robust covariance"))
    return _wild_cluster_bootstrap(Vector{T}(model.y), Matrix{T}(model.X),
                                   model.varnames, clusters, coefficient,
                                   T(null_value); kwargs...)
end

function wild_cluster_bootstrap(model::PanelRegModel{T}, coefficient, null_value::Real=0.0;
                                clusters::Union{Nothing,AbstractVector}=nothing,
                                kwargs...) where {T<:AbstractFloat}
    model.method === :fe || throw(ArgumentError(
        "wild_cluster_bootstrap on a PanelRegModel currently supports model=:fe " *
        "(the within estimator), got :$(model.method). For other panel estimators, run " *
        "the bootstrap on the corresponding cross-sectional RegModel."))
    groups = model.data.group_id
    unique_groups = sort(unique(groups))
    if model.hdfe !== nothing
        # HDFE (T272, #371): re-absorb with the same dimensions and tolerance the
        # fit used, so the bootstrap design matches the estimated one exactly.
        h = model.hdfe
        fe_ids = AbstractVector[_hdfe_dimension(model.data, d) for d in h.absorb]
        # Reusing the fit's own iteration count matters when it did NOT converge:
        # the bootstrap must then reproduce the same truncated design, not a
        # better-converged one.
        ab = absorb_fe(Vector{T}(model.y), Matrix{T}(model.X), fe_ids;
                       tol=h.tol, maxiter=max(h.iterations, 1), accel=h.accel)
        y_dm, X_dm = ab.y, ab.X
    else
        y_dm, _ = _within_demean(Vector{T}(model.y), groups, unique_groups)
        X_dm, _ = _within_demean_matrix(Matrix{T}(model.X), groups, unique_groups)
        if model.twoway
            times = model.data.time_id
            unique_times = sort(unique(times))
            _twoway_demean!(y_dm, X_dm, Vector{T}(model.y), Matrix{T}(model.X),
                            groups, times, unique_groups, unique_times)
        end
    end
    cl = clusters === nothing ? groups : clusters
    return _wild_cluster_bootstrap(y_dm, X_dm, model.varnames, cl, coefficient,
                                   T(null_value); kwargs...)
end

"""Resolve a coefficient selector (index / name) against the design's variable names."""
function _wcb_resolve_coef(coefficient, varnames::Vector{String}, k::Int)
    if coefficient isa Integer
        1 <= coefficient <= k || throw(ArgumentError(
            "coefficient index $coefficient out of range 1:$k"))
        return Int(coefficient)
    end
    name = string(coefficient)
    idx = findfirst(==(name), varnames)
    idx === nothing && throw(ArgumentError(
        "coefficient '$name' not found. Available: $varnames"))
    return idx
end

function _wild_cluster_bootstrap(y::Vector{T}, X::Matrix{T}, varnames::Vector{String},
                                 clusters::AbstractVector, coefficient, null_value::T;
                                 n_boot::Int=999,
                                 weights::Symbol=:rademacher,
                                 imposenull::Bool=true,
                                 ci::Bool=true,
                                 level::Real=0.95,
                                 ci_gridpoints::Int=25,
                                 enumerate::Union{Nothing,Bool}=nothing,
                                 seed::Union{Integer,Nothing}=nothing,
                                 rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    n, k = size(X)
    length(y) == n || throw(ArgumentError("y and X must have the same number of rows"))
    length(clusters) == n || throw(ArgumentError(
        "clusters must have length $n, got $(length(clusters))"))
    weights in (:rademacher, :webb) ||
        throw(ArgumentError("weights must be :rademacher or :webb, got :$weights"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be positive"))
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1)"))

    j = _wcb_resolve_coef(coefficient, varnames, k)

    uc = unique(clusters)
    G = length(uc)
    G >= 2 || throw(ArgumentError("Need at least 2 clusters, got $G"))
    cluster_idx = [findall(==(g), clusters) for g in uc]

    V, enumerated = _wcb_weight_matrix(G, n_boot, weights, rng, T; enumerate=enumerate)

    t_obs, t_boot = _wcb_run(y, X, cluster_idx, j, null_value, V; imposenull=imposenull)
    p_sym, p_et = _wcb_pvalues(t_obs, t_boot)

    # Reference asymptotic p-value from the same cluster-robust t
    p_asy = 2 * (1 - cdf(Normal(), abs(t_obs)))

    XtXinv = Matrix{T}(robust_inv(Symmetric(X' * X)))
    beta = XtXinv * (X' * y)
    resid = y .- X * beta
    se_hat = _wcb_cluster_se(X, resid, XtXinv, cluster_idx, j)

    lo, hi = if ci
        _wcb_invert(y, X, cluster_idx, j, V, T(level);
                    imposenull=imposenull, gridpoints=ci_gridpoints,
                    se_hat=se_hat, beta_hat=beta[j])
    else
        (T(NaN), T(NaN))
    end

    result = WildClusterBootstrap{T}(varnames[j], j, beta[j], null_value, t_obs,
                                     p_sym, p_et, T(p_asy), lo, hi, T(level),
                                     t_boot, size(V, 2), G, weights, imposenull, enumerated)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_boot" => n_boot, "weights" => String(weights),
                                  "imposenull" => imposenull)))
end
