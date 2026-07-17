# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Rank-correlation / association tests (EV-34, #442): Pearson, Spearman, and
Kendall τ_a / τ_b, exposed through `cor_test(x, y; method=...)`.

- **Pearson**: `r`, `t = r√((n−2)/(1−r²)) ~ t(n−2)`, two-sided; Fisher-z confidence
  interval `tanh(atanh(r) ± z_{α/2}/√(n−3))`.
- **Spearman**: Pearson correlation of the (tie-averaged) ranks; reports the
  `S = Σ dᵢ² = (1−ρ)(n³−n)/6` statistic and the `t(n−2)` approximation p-value
  (matches R `cor.test(method="spearman", exact=FALSE)`).
- **Kendall**: the concordant−discordant count `C−D` is obtained in `O(n log n)`
  from a **merge-sort inversion counter** (Knight 1966), not the `O(n²)` double
  loop. τ_a `= (C−D)/[n(n−1)/2]` (ignores ties); τ_b `= (C−D)/√((n₀−n₁)(n₀−n₂))`
  with `n₀ = n(n−1)/2`, `n₁,n₂` the pair-tie counts in `x` and `y`. Small-`n`
  tie-free data use the exact null (Mahonian recursion, R's `ckendall`);
  otherwise the normal approximation with the full tie-adjusted variance.

References:
- Kendall, M. G. (1938). "A New Measure of Rank Correlation." Biometrika 30(1/2).
- Knight, W. R. (1966). "A Computer Method for Calculating Kendall's Tau with
  Ungrouped Data." JASA 61(314).
- Spearman, C. (1904). "The Proof and Measurement of Association between Two Things."
"""

using Statistics, Distributions

# =============================================================================
# Kendall exact null: Mahonian recursion (R's ckendall / cor.test kendall exact)
# =============================================================================

"""Number of permutations of `n` items with exactly `k` discordant pairs
(`ckendall`). Memoized recursion `c(k,n) = Σ_{i=0}^{n−1} c(k−i, n−1)`."""
function _ckendall(k::Int, n::Int, memo::Dict{Tuple{Int,Int},Float64})
    u = n * (n - 1) ÷ 2
    (k < 0 || k > u) && return 0.0
    n == 1 && return k == 0 ? 1.0 : 0.0
    haskey(memo, (k, n)) && return memo[(k, n)]
    s = 0.0
    @inbounds for i in 0:(n - 1)
        s += _ckendall(k - i, n - 1, memo)
    end
    memo[(k, n)] = s
    return s
end

"""`P(K ≤ q)` under the Kendall null for `n` items (cumulative Mahonian / n!)."""
function _pkendall(q::Int, n::Int)
    q < 0 && return 0.0
    u = n * (n - 1) ÷ 2
    q >= u && return 1.0
    memo = Dict{Tuple{Int,Int},Float64}()
    total = exp(sum(log, 1:n))            # n!
    s = 0.0
    for k in 0:q
        s += _ckendall(k, n, memo)
    end
    return s / total
end

# =============================================================================
# Concordant−discordant via merge-sort inversion counting (Knight 1966)
# =============================================================================

"""Count strict inversions (`y[i] > y[j]` for `i < j`) of `y` in `O(n log n)`
by merge sort. `y` is destroyed in a scratch copy; the original is untouched."""
function _merge_inversions(y::Vector{Int})
    n = length(y)
    a = copy(y)
    buf = similar(a)
    inv = 0
    width = 1
    while width < n
        i = 1
        while i <= n
            l = i
            m = min(i + width - 1, n)
            r = min(i + 2 * width - 1, n)
            # merge a[l:m] and a[m+1:r] into buf[l:r], counting inversions
            p = l; q = m + 1; k = l
            @inbounds while p <= m && q <= r
                if a[p] <= a[q]
                    buf[k] = a[p]; p += 1
                else
                    inv += (m - p + 1)        # a[p..m] all > a[q]
                    buf[k] = a[q]; q += 1
                end
                k += 1
            end
            @inbounds while p <= m
                buf[k] = a[p]; p += 1; k += 1
            end
            @inbounds while q <= r
                buf[k] = a[q]; q += 1; k += 1
            end
            i += 2 * width
        end
        a, buf = buf, a
        width *= 2
    end
    return inv
end

"""Sum of `t(t−1)/2` over groups of identical values in a sorted vector `v`."""
function _pair_ties(v::AbstractVector)
    n = length(v)
    total = 0
    i = 1
    @inbounds while i <= n
        j = i
        while j < n && v[j + 1] == v[i]
            j += 1
        end
        t = j - i + 1
        total += t * (t - 1) ÷ 2
        i = j + 1
    end
    return total
end

# =============================================================================
# Average (tie-corrected) ranks
# =============================================================================

"""Tie-averaged ranks of `x` (Statistics-only; equivalent to `StatsBase.tiedrank`)."""
function _tiedrank(x::AbstractVector{<:Real})
    n = length(x)
    p = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    @inbounds while i <= n
        j = i
        while j < n && x[p[j + 1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2                 # 1-based average rank of the tie block
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

# =============================================================================
# cor_test
# =============================================================================

"""
    cor_test(x, y; method=:pearson, alpha=0.05) -> CorTestResult

Test the null of no association (`H₀: coefficient = 0`) between paired series `x`
and `y`. `method` ∈ {`:pearson`, `:spearman`, `:kendall`}.

- `:pearson` — Pearson `r`; `t = r√((n−2)/(1−r²)) ~ t(n−2)`; Fisher-z CI.
- `:spearman` — Pearson on ranks; `S = (1−ρ)(n³−n)/6`; `t(n−2)` approximation
  (matches R `exact=FALSE`).
- `:kendall` — τ_a and τ_b via the `O(n log n)` merge-sort concordance counter;
  exact null for small tie-free `n`, else the tie-adjusted normal approximation.

Returns a [`CorTestResult`](@ref). See also [`equality_test`](@ref) for the
by-group equality battery.

# Examples
```julia
cor_test(x, y; method=:pearson)
cor_test(x, y; method=:kendall)   # reports τ_b
```
"""
function cor_test(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                  method::Symbol=:pearson, alpha::Real=0.05)
    length(x) == length(y) || throw(ArgumentError("x and y must have equal length"))
    n = length(x)
    n >= 3 || throw(ArgumentError("need at least 3 observations, got $n"))
    xf = float.(collect(x)); yf = float.(collect(y))
    T = eltype(xf)

    if method == :pearson
        return _pearson_cor(xf, yf, T(alpha))
    elseif method == :spearman
        return _spearman_cor(xf, yf)
    elseif method == :kendall
        return _kendall_cor(xf, yf)
    else
        throw(ArgumentError("unknown method $method; use :pearson, :spearman, or :kendall"))
    end
end

function _pearson_cor(x::Vector{T}, y::Vector{T}, alpha::T) where {T<:AbstractFloat}
    n = length(x)
    r = cor(x, y)
    r = clamp(r, -one(T), one(T))
    df = T(n - 2)
    stat = r * sqrt(df / (one(T) - r^2))
    pval = T(2 * ccdf(TDist(df), abs(stat)))
    # Fisher-z confidence interval
    if n > 3 && abs(r) < 1
        z = atanh(r)
        se = one(T) / sqrt(T(n - 3))
        zc = T(quantile(Normal(), 1 - alpha / 2))
        lo = tanh(z - zc * se); hi = tanh(z + zc * se)
    else
        lo = T(NaN); hi = T(NaN)
    end
    CorTestResult{T}(:pearson, r, stat, pval, n, df, lo, hi, false,
                     "Pearson product-moment correlation")
end

function _spearman_cor(x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}
    n = length(x)
    rx = T.(_tiedrank(x)); ry = T.(_tiedrank(y))
    rho = clamp(cor(rx, ry), -one(T), one(T))
    S = (one(T) - rho) * (T(n)^3 - T(n)) / 6           # Σ dᵢ² form
    df = T(n - 2)
    # t-approximation (R cor.test method="spearman", exact=FALSE)
    denom = one(T) - rho^2
    stat = denom > 0 ? rho * sqrt(df / denom) : T(Inf) * sign(rho)
    pval = T(2 * ccdf(TDist(df), abs(stat)))
    CorTestResult{T}(:spearman, rho, S, pval, n, df, T(NaN), T(NaN), false,
                     "Spearman's rank correlation ρ")
end

function _kendall_cor(x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}
    n = length(x)
    # lexicographic sort by (x, y)
    perm = sortperm(collect(zip(x, y)); by = t -> (t[1], t[2]))
    xs = x[perm]; ys = y[perm]
    tot = n * (n - 1) ÷ 2
    xtie = _pair_ties(xs)                              # ties in x (n₁)
    ytie = _pair_ties(sort(y))                         # ties in y (n₂)
    # joint (x,y) ties on the lexsorted data
    ntie = _pair_ties(collect(zip(xs, ys)))
    # discordant pairs = strict inversions of the y-sequence (mapped to ranks)
    yranks = round.(Int, _dense_codes(ys))
    dis = _merge_inversions(yranks)
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    n0 = T(tot)
    tau_a = T(con_minus_dis) / n0
    denom_b = sqrt((n0 - T(xtie)) * (n0 - T(ytie)))
    tau_b = denom_b > 0 ? T(con_minus_dis) / denom_b : T(NaN)

    has_ties = (xtie > 0 || ytie > 0)
    if !has_ties && n < 50
        # exact null on the concordant count T = con_minus_dis/2 + tot/2
        q = (con_minus_dis + tot) ÷ 2                  # concordant pairs
        half = tot / 2
        pv = q > half ? (1 - _pkendall(q - 1, n)) : _pkendall(q, n)
        pval = T(min(2 * pv, 1.0))
        stat = T(q)                                    # R reports concordant count T
        return CorTestResult{T}(:kendall, tau_a, stat, pval, n, T(NaN), T(NaN), T(NaN),
                                true, "Kendall's τ_a (exact null)")
    else
        # tie-adjusted normal approximation (R cor.test method="kendall")
        xt = _tie_multiplicities(xs); yt = _tie_multiplicities(sort(y))
        nn = T(n)
        v0 = nn * (nn - 1) * (2nn + 5)
        vt = sum(T(t) * (t - 1) * (2t + 5) for t in xt; init = zero(T))
        vu = sum(T(u) * (u - 1) * (2u + 5) for u in yt; init = zero(T))
        v1 = (sum(T(t) * (t - 1) for t in xt; init = zero(T))) *
             (sum(T(u) * (u - 1) for u in yt; init = zero(T)))
        v2 = (sum(T(t) * (t - 1) * (t - 2) for t in xt; init = zero(T))) *
             (sum(T(u) * (u - 1) * (u - 2) for u in yt; init = zero(T)))
        var = (v0 - vt - vu) / 18 +
              v1 / (2 * nn * (nn - 1)) +
              v2 / (9 * nn * (nn - 1) * (nn - 2))
        z = var > 0 ? T(con_minus_dis) / sqrt(var) : zero(T)
        pval = T(2 * ccdf(Normal(), abs(z)))
        return CorTestResult{T}(:kendall, tau_b, z, pval, n, T(NaN), T(NaN), T(NaN),
                                false, "Kendall's τ_b (tie-adjusted normal approx.)")
    end
end

"""Dense integer codes (1..k) of the distinct values of `v`, preserving order,
so equal values map to equal codes (for inversion counting)."""
function _dense_codes(v::AbstractVector{<:Real})
    u = sort(unique(v))
    lut = Dict(val => i for (i, val) in enumerate(u))
    return Float64[lut[x] for x in v]
end

"""Multiplicities (> 1) of each distinct value in a sorted vector."""
function _tie_multiplicities(v::AbstractVector)
    n = length(v)
    out = Int[]
    i = 1
    @inbounds while i <= n
        j = i
        while j < n && v[j + 1] == v[i]
            j += 1
        end
        t = j - i + 1
        t > 1 && push!(out, t)
        i = j + 1
    end
    return out
end
