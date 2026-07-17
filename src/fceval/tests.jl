# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-39 (#447): Forecast-comparison tests.
#
# References: Diebold & Mariano (1995, JBES); Harvey, Leybourne & Newbold
# (1997, IJF; 1998, JBES); Clark & West (2007, J. Econometrics);
# Mincer & Zarnowitz (1969).
#
# Long-run-variance reuse (EV-12 / src/core/covariance.jl):
#   * MZ / encompassing HAC covariances reuse `newey_west` (bandwidth=L,
#     bartlett) — its meat `S` is built WITHOUT ÷n and the bread is
#     (X'X)^{-1}, giving the coefficient sandwich directly.
#   * The DM / CW scalar long-run variance is computed by `_dm_lrvar`, which
#     matches R `forecast::dm.test`'s truncated `acf` estimator: γ₀ + 2Σγⱼ with
#     each γⱼ divided by T, so `V̂/T` scaling is exact (see the scaling note).

"""
    _dm_lrvar(d, h; kernel=:rectangular) -> T

Truncated long-run variance `V̂ = γ₀ + 2·Σ_{j=1}^{h-1} w_j γ_j` of the loss
differential, with autocovariances `γ_j = (1/T) Σ (d_t−d̄)(d_{t+j}−d̄)`. The
default rectangular (`:rectangular`) kernel (`w_j = 1`) reproduces the `acf`
estimator of R `forecast::dm.test`; `:bartlett` applies `w_j = 1 − j/h`. The
Diebold–Mariano statistic then divides `V̂` by `T`.
"""
function _dm_lrvar(d::AbstractVector{T}, h::Int; kernel::Symbol=:rectangular) where {T<:AbstractFloat}
    n = length(d)
    dc = d .- mean(d)
    V = sum(abs2, dc) / n                      # γ₀
    @inbounds for j in 1:(h-1)
        w = kernel === :rectangular ? one(T) :
            kernel === :bartlett    ? one(T) - T(j) / T(h) :
            throw(ArgumentError("kernel must be :rectangular or :bartlett; got :$kernel"))
        gamma_j = zero(T)
        for t in 1:(n-j)
            gamma_j += dc[t+j] * dc[t]
        end
        V += 2 * w * (gamma_j / n)
    end
    return V
end

_loss_fun(loss::Symbol) = loss === :se ? (e -> e^2) :
                          loss === :ad ? (e -> abs(e)) :
                          throw(ArgumentError("loss must be :se, :ad, or a function; got :$loss"))
_loss_fun(loss) = loss   # already a function

"""
    diebold_mariano(e1, e2; h=1, loss=:se, hln=true, kernel=:rectangular,
                    alternative=:two_sided) -> DMTestResult

Diebold–Mariano (1995) test of equal predictive accuracy between two forecast
error series `e1`, `e2` (`e = actual − forecast`). The loss differential is
`d_t = g(e1_t) − g(e2_t)` with `g` the squared (`loss=:se`), absolute
(`loss=:ad`), or a user-supplied loss. The statistic is
`DM = d̄ / √(V̂/T)` where `V̂` is the truncated HAC long-run variance of `d_t`
at lag `h−1`.

With `hln=true` (default) the Harvey–Leybourne–Newbold (1997) small-sample
factor `√((T+1−2h+h(h−1)/T)/T)` is applied and the statistic is referenced to
`t_{T−1}` (matching R `forecast::dm.test`); with `hln=false` it uses `N(0,1)`.

A positive `DM` means model 1 has the *larger* average loss (is worse).
`alternative ∈ (:two_sided, :less, :greater)`.

!!! warning
    The DM test is **invalid for nested models** — under the null the forecast
    error difference is degenerate. Use [`clark_west`](@ref) instead.
"""
function diebold_mariano(e1::AbstractVector{<:Real}, e2::AbstractVector{<:Real};
                         h::Int=1, loss=:se, hln::Bool=true,
                         kernel::Symbol=:rectangular, alternative::Symbol=:two_sided)
    T = float(promote_type(eltype(e1), eltype(e2)))
    length(e1) == length(e2) || throw(DimensionMismatch("e1 and e2 must have equal length"))
    h >= 1 || throw(ArgumentError("h must be ≥ 1"))
    alternative in (:two_sided, :less, :greater) ||
        throw(ArgumentError("alternative must be :two_sided, :less, or :greater"))
    g = _loss_fun(loss)
    n = length(e1)
    d = T[g(T(e1[t])) - g(T(e2[t])) for t in 1:n]
    dbar = mean(d)
    V = _dm_lrvar(d, h; kernel=kernel)
    V > 0 || throw(ArgumentError("long-run variance of the loss differential is non-positive"))
    stat = dbar / sqrt(V / n)

    if hln
        k = sqrt((n + 1 - 2h + (h * (h - 1)) / n) / n)
        stat *= k
    end
    loss_sym = loss isa Symbol ? loss : :custom
    dof = n - 1
    dist = hln ? TDist(dof) : Normal()
    pval = alternative === :two_sided ? 2 * (1 - cdf(dist, abs(stat))) :
           alternative === :greater   ? (1 - cdf(dist, stat)) :
                                         cdf(dist, stat)
    DMTestResult{T}(T(stat), T(pval), dbar, V, h, loss_sym, hln, alternative, n)
end

"""
    clark_west(e_small, e_big, f_adj; h=1, alternative=:greater) -> ClarkWestResult

Clark–West (2007) adjusted-MSPE test for **nested** models, where the small
(restricted) model is nested in the big (unrestricted) one. Forms the adjusted
differential

    f̂_t = e_small_t² − ( e_big_t² − (ŷ_small,t − ŷ_big,t)² )

with `f_adj_t = ŷ_small,t − ŷ_big,t` the gap between the two point forecasts, and
tests `E[f̂] ≤ 0` (the big model does **not** improve MSPE) against the one-sided
`greater` alternative. The statistic is `mean(f̂)/√(V̂/T)` with `V̂` the truncated
HAC long-run variance of `f̂_t`, referenced to the standard normal (Clark–West
2007). This is the correct test when [`diebold_mariano`](@ref) is invalid because
the models are nested.
"""
function clark_west(e_small::AbstractVector{<:Real}, e_big::AbstractVector{<:Real},
                    f_adj::AbstractVector{<:Real};
                    h::Int=1, alternative::Symbol=:greater)
    T = float(promote_type(eltype(e_small), eltype(e_big), eltype(f_adj)))
    n = length(e_small)
    (length(e_big) == n && length(f_adj) == n) ||
        throw(DimensionMismatch("e_small, e_big, f_adj must have equal length"))
    h >= 1 || throw(ArgumentError("h must be ≥ 1"))
    alternative in (:two_sided, :less, :greater) ||
        throw(ArgumentError("alternative must be :two_sided, :less, or :greater"))
    fhat = T[T(e_small[t])^2 - (T(e_big[t])^2 - T(f_adj[t])^2) for t in 1:n]
    fbar = mean(fhat)
    V = _dm_lrvar(fhat, h)
    V > 0 || throw(ArgumentError("long-run variance of the CW differential is non-positive"))
    stat = fbar / sqrt(V / n)
    dist = Normal()
    pval = alternative === :two_sided ? 2 * (1 - cdf(dist, abs(stat))) :
           alternative === :greater   ? (1 - cdf(dist, stat)) :
                                         cdf(dist, stat)
    ClarkWestResult{T}(T(stat), T(pval), fbar, V, h, alternative, n)
end

# HAC coefficient covariance for the level regressions. Reuses `newey_west`
# (bandwidth=L, bartlett) for L≥1; L=0 gives the White (HC0) sandwich.
function _mz_vcov(X::AbstractMatrix{T}, u::AbstractVector{T}, L::Int, kernel::Symbol) where {T<:AbstractFloat}
    if L >= 1
        return newey_west(X, u; bandwidth=L, kernel=kernel)
    else
        XtXi = robust_inv(X' * X)
        G = X .* u
        S = G' * G
        V = XtXi * S * XtXi
        return Matrix{T}((V + V') / 2)
    end
end

"""
    mincer_zarnowitz(actual, fc; lags=0, kernel=:bartlett) -> MincerZarnowitzResult

Mincer–Zarnowitz (1969) forecast-efficiency test. Runs the regression
`actual = a + b·fc + u` and jointly tests `(a, b) = (0, 1)` with a Newey–West
HAC covariance (truncation lag `lags`; `lags=0` gives the White sandwich).
Reports the χ²(2) Wald statistic and the equivalent `F(2, T−2)`. A weakly
efficient forecast satisfies `a = 0`, `b = 1`.
"""
function mincer_zarnowitz(actual::AbstractVector{<:Real}, fc::AbstractVector{<:Real};
                          lags::Int=0, kernel::Symbol=:bartlett)
    T = float(promote_type(eltype(actual), eltype(fc)))
    n = length(actual)
    n == length(fc) || throw(DimensionMismatch("actual and fc must have equal length"))
    lags >= 0 || throw(ArgumentError("lags must be ≥ 0"))
    y = collect(T, actual)
    X = hcat(ones(T, n), collect(T, fc))
    beta = (X' * X) \ (X' * y)
    u = y .- X * beta
    V = _mz_vcov(X, u, lags, kernel)
    a, b = beta[1], beta[2]
    se = sqrt.(diag(V))
    d = T[a - 0, b - 1]
    wald = (d' * robust_inv(Matrix{T}(V)) * d)
    fstat = wald / 2
    pval_wald = 1 - cdf(Chisq(2), wald)
    pval_f = 1 - cdf(FDist(2, n - 2), fstat)
    MincerZarnowitzResult{T}(a, b, se, T(wald), T(pval_wald), T(fstat), T(pval_f),
                             lags, kernel, n)
end

"""
    forecast_encompassing(actual, fc1, fc2; lags=0, kernel=:bartlett) -> ForecastEncompassingResult

Regression-based forecast-encompassing test (Harvey, Leybourne & Newbold 1998).
Estimates `actual = a + b₁·fc1 + b₂·fc2 + u` with a Newey–West HAC covariance and
tests `b₂ = 0`. Non-rejection means forecast 1 **encompasses** forecast 2 —
forecast 2 carries no incremental information. The two-sided p-value references
`t_{T−3}`.
"""
function forecast_encompassing(actual::AbstractVector{<:Real}, fc1::AbstractVector{<:Real},
                               fc2::AbstractVector{<:Real};
                               lags::Int=0, kernel::Symbol=:bartlett)
    T = float(promote_type(eltype(actual), eltype(fc1), eltype(fc2)))
    n = length(actual)
    (length(fc1) == n && length(fc2) == n) ||
        throw(DimensionMismatch("actual, fc1, fc2 must have equal length"))
    lags >= 0 || throw(ArgumentError("lags must be ≥ 0"))
    y = collect(T, actual)
    X = hcat(ones(T, n), collect(T, fc1), collect(T, fc2))
    beta = (X' * X) \ (X' * y)
    u = y .- X * beta
    V = _mz_vcov(X, u, lags, kernel)
    b1, b2 = beta[2], beta[3]
    se_b2 = sqrt(V[3, 3])
    tstat = b2 / se_b2
    pval = 2 * (1 - cdf(TDist(n - 3), abs(tstat)))
    ForecastEncompassingResult{T}(b1, b2, se_b2, T(tstat), T(pval), lags, kernel, n)
end

# --- Display -----------------------------------------------------------------

function Base.show(io::IO, r::DMTestResult{T}) where {T}
    dname = r.hln ? "t($(r.T_obs - 1))" : "N(0,1)"
    data = Any[
        "DM statistic"        _fmt(r.statistic);
        "p-value"             _format_pvalue(r.pvalue);
        "Mean loss diff (d̄)"  _fmt(r.dbar);
        "Horizon h"           string(r.h);
        "Loss"                string(r.loss);
        "HLN correction"      (r.hln ? "yes" : "no");
        "Reference dist"      dname;
        "Alternative"         string(r.alternative);
        "Observations"        string(r.T_obs);
    ]
    _pretty_table(io, data; title = "Diebold–Mariano Test (H₀: equal predictive accuracy)",
        column_labels = ["", ""], alignment = [:l, :r])
    return nothing
end

function Base.show(io::IO, r::ClarkWestResult{T}) where {T}
    data = Any[
        "CW statistic"       _fmt(r.statistic);
        "p-value (1-sided)"  _format_pvalue(r.pvalue);
        "Mean adj. diff"     _fmt(r.fbar);
        "Horizon h"          string(r.h);
        "Reference dist"     "N(0,1)";
        "Observations"       string(r.T_obs);
    ]
    _pretty_table(io, data; title = "Clark–West Test (nested models; H₀: no MSPE improvement)",
        column_labels = ["", ""], alignment = [:l, :r])
    return nothing
end

function Base.show(io::IO, r::MincerZarnowitzResult{T}) where {T}
    _coef_table(io, "Mincer–Zarnowitz Efficiency Regression (actual = a + b·fc)",
                ["a (intercept)", "b (slope)"], T[r.a, r.b], r.se;
                dist=:z)
    data = Any[
        "Joint H₀: (a,b)=(0,1)"  "";
        "Wald χ²(2)"             string(_fmt(r.wald), "  [p=", _format_pvalue(r.pvalue_wald), "]");
        "F(2, $(r.T_obs-2))"     string(_fmt(r.fstat), "  [p=", _format_pvalue(r.pvalue_f), "]");
        "HAC lags"               string(r.lags);
        "Observations"           string(r.T_obs);
    ]
    _pretty_table(io, data; title = "", column_labels = ["", ""], alignment = [:l, :r])
    return nothing
end

function Base.show(io::IO, r::ForecastEncompassingResult{T}) where {T}
    data = Any[
        "b₁ (fc1 weight)"        _fmt(r.b1);
        "b₂ (fc2 weight)"        _fmt(r.b2);
        "se(b₂)"                 _fmt(r.se_b2);
        "t (H₀: b₂=0)"           _fmt(r.tstat);
        "p-value"                _format_pvalue(r.pvalue);
        "HAC lags"               string(r.lags);
        "Observations"           string(r.T_obs);
    ]
    _pretty_table(io, data;
        title = "Forecast Encompassing (H₀: fc1 encompasses fc2, b₂=0)",
        column_labels = ["", ""], alignment = [:l, :r])
    return nothing
end

for TT in (:DMTestResult, :ClarkWestResult, :MincerZarnowitzResult, :ForecastEncompassingResult)
    @eval Base.show(io::IO, ::MIME"text/plain", r::$TT) = show(io, r)
    @eval report(r::$TT) = show(stdout, r)
    @eval report(io::IO, r::$TT) = show(io, r)
end

# StatsAPI interface
StatsAPI.pvalue(r::DMTestResult) = r.pvalue
StatsAPI.pvalue(r::ClarkWestResult) = r.pvalue
StatsAPI.pvalue(r::MincerZarnowitzResult) = r.pvalue_wald
StatsAPI.pvalue(r::ForecastEncompassingResult) = r.pvalue
