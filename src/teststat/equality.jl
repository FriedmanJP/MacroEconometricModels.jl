# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Equality-of-distribution "Basic statistics" battery (EV-34, #442) — the EViews
"Equality Tests by Classification" location and scale tests, plus convenience
wrappers, exposed through `equality_test(y, g; test=...)`.

**Location** (`H₀`: equal centers/distributions across groups):
- `:t` one-/two-sample t (pooled or Welch/Satterthwaite; `equal_var` toggles);
- `:anova` one-way ANOVA F (classic or Welch when `equal_var=false`);
- `:mann_whitney`, `:wilcoxon` (paired signed-rank), `:kruskal_wallis`,
  `:van_der_waerden`, `:median` (Mood χ²).

**Scale** (`H₀`: equal dispersion):
- `:f` two-group variance ratio, `:bartlett`, `:levene` (deviations from the group
  **mean**), `:brown_forsythe` (deviations from the group **median**),
  `:siegel_tukey`.

Rank tests use the **exact null** for small tie-free samples (Mann–Whitney and
Wilcoxon signed-rank via the R `cwilcox`/`csignrank` recursions, `n < 50`) and
otherwise the normal approximation **with continuity and tie corrections**,
mirroring R's refuse-exact-with-ties → normal fallback. Kruskal–Wallis uses the
tie-corrected χ² form with factor `C = 1 − Σ(tⱼ³−tⱼ)/(N³−N)`.

The grouped-data Levene / Brown–Forsythe tests here operate on **raw data split by
a classifier**; the regression-residual heteroskedasticity variants live in the
diagnostics module (EV-31). See also [`cor_test`](@ref) for the association tests.

References:
- Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods." Biometrics 1(6).
- Mann, H. B. & Whitney, D. R. (1947). "On a Test of Whether one of Two Random
  Variables is Stochastically Larger than the Other." Ann. Math. Stat. 18(1).
- Kruskal, W. H. & Wallis, W. A. (1952). "Use of Ranks in One-Criterion Variance
  Analysis." JASA 47(260).
- van der Waerden, B. L. (1952). "Order Tests for the Two-Sample Problem."
- Levene, H. (1960); Brown, M. B. & Forsythe, A. B. (1974), JASA 69(346).
- Bartlett, M. S. (1937); Welch, B. L. (1951); Satterthwaite, F. E. (1946).
"""

using Statistics, Distributions

# =============================================================================
# Exact null distributions (R-compatible recursions)
# =============================================================================

"""Mann–Whitney count `c(k,m,n) = c(k−n, m−1, n) + c(k, m, n−1)` (R's `cwilcox`)."""
function _cwilcox(k::Int, m::Int, n::Int, memo::Dict{Tuple{Int,Int,Int},Float64})
    (k < 0 || k > m * n) && return 0.0
    (m == 0 || n == 0) && return k == 0 ? 1.0 : 0.0
    haskey(memo, (k, m, n)) && return memo[(k, m, n)]
    v = _cwilcox(k - n, m - 1, n, memo) + _cwilcox(k, m, n - 1, memo)
    memo[(k, m, n)] = v
    return v
end

"""`P(U ≤ q)` under the Mann–Whitney null (`m`,`n`)."""
function _pwilcox(q::Int, m::Int, n::Int)
    q < 0 && return 0.0
    q >= m * n && return 1.0
    memo = Dict{Tuple{Int,Int,Int},Float64}()
    total = binomial(big(m + n), big(n))
    s = big(0)
    for k in 0:q
        s += round(BigInt, _cwilcox(k, m, n, memo))
    end
    return Float64(s / total)
end

"""Wilcoxon signed-rank count `c(k,n) = c(k−n, n−1) + c(k, n−1)` (R's `csignrank`)."""
function _csignrank(k::Int, n::Int, memo::Dict{Tuple{Int,Int},Float64})
    u = n * (n + 1) ÷ 2
    (k < 0 || k > u) && return 0.0
    n == 0 && return k == 0 ? 1.0 : 0.0
    haskey(memo, (k, n)) && return memo[(k, n)]
    v = _csignrank(k - n, n - 1, memo) + _csignrank(k, n - 1, memo)
    memo[(k, n)] = v
    return v
end

"""`P(V ≤ q)` under the signed-rank null for `n` nonzero differences."""
function _psignrank(q::Int, n::Int)
    q < 0 && return 0.0
    u = n * (n + 1) ÷ 2
    q >= u && return 1.0
    memo = Dict{Tuple{Int,Int},Float64}()
    total = big(2)^n
    s = big(0)
    for k in 0:q
        s += round(BigInt, _csignrank(k, n, memo))
    end
    return Float64(s / total)
end

# =============================================================================
# Grouping helpers
# =============================================================================

"""Split `y` into groups by the distinct (sorted) values of `g`; returns
`(labels, groups)`."""
function _split_groups(y::AbstractVector{<:Real}, g::AbstractVector)
    length(y) == length(g) || throw(ArgumentError("y and g must have equal length"))
    labels = sort(unique(g))
    groups = [Float64[y[i] for i in eachindex(g) if g[i] == lab] for lab in labels]
    return labels, groups
end

_tie_factor(counts) = sum(c^3 - c for c in counts)   # Σ(t³−t)

# =============================================================================
# Location tests
# =============================================================================

"""Two-sample t-test on groups `a`,`b`; `equal_var` toggles pooled vs Welch."""
function _two_sample_t(a::Vector{T}, b::Vector{T}, equal_var::Bool) where {T<:AbstractFloat}
    n1 = length(a); n2 = length(b)
    m1 = mean(a); m2 = mean(b)
    v1 = var(a); v2 = var(b)
    if equal_var
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = sqrt(sp2 * (one(T) / n1 + one(T) / n2))
        df = T(n1 + n2 - 2)
        name = :two_sample_t; detail = "pooled variance"
    else
        se = sqrt(v1 / n1 + v2 / n2)
        # Welch–Satterthwaite fractional df
        df = (v1 / n1 + v2 / n2)^2 /
             ((v1 / n1)^2 / (n1 - 1) + (v2 / n2)^2 / (n2 - 1))
        name = :welch_t; detail = "Welch/Satterthwaite df"
    end
    t = (m1 - m2) / se
    pval = T(2 * ccdf(TDist(df), abs(t)))
    EqualityTestResult{T}(name, t, pval, df, T(NaN), 2, [n1, n2], true, detail)
end

"""One-sample t-test of `H₀: mean(a) = μ₀`."""
function _one_sample_t(a::Vector{T}, mu0::T) where {T<:AbstractFloat}
    n = length(a)
    df = T(n - 1)
    t = (mean(a) - mu0) / (std(a) / sqrt(T(n)))
    pval = T(2 * ccdf(TDist(df), abs(t)))
    EqualityTestResult{T}(:one_sample_t, t, pval, df, T(NaN), 1, [n], true,
                          "one-sample (μ₀ = $(mu0))")
end

"""Paired t-test on equal-length groups `a`,`b`."""
function _paired_t(a::Vector{T}, b::Vector{T}) where {T<:AbstractFloat}
    length(a) == length(b) || throw(ArgumentError("paired t requires equal group sizes"))
    d = a .- b
    n = length(d)
    df = T(n - 1)
    t = mean(d) / (std(d) / sqrt(T(n)))
    pval = T(2 * ccdf(TDist(df), abs(t)))
    EqualityTestResult{T}(:paired_t, t, pval, df, T(NaN), 2, [n, n], true, "paired differences")
end

"""One-way ANOVA F (classic, equal-variance)."""
function _anova_classic(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    N = sum(ns)
    grand = mean(vcat(groups...))
    ssb = sum(ns[j] * (mean(groups[j]) - grand)^2 for j in 1:k)
    ssw = sum(sum((x - mean(groups[j]))^2 for x in groups[j]) for j in 1:k)
    df1 = T(k - 1); df2 = T(N - k)
    F = (ssb / df1) / (ssw / df2)
    pval = T(ccdf(FDist(df1, df2), F))
    EqualityTestResult{T}(:anova, F, pval, df1, df2, k, ns, true, "classic (equal variance)")
end

"""Welch one-way ANOVA F (unequal variances)."""
function _anova_welch(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    ms = mean.(groups)
    vs = var.(groups)
    w = T[ns[j] / vs[j] for j in 1:k]
    W = sum(w)
    xbar = sum(w[j] * ms[j] for j in 1:k) / W
    num = sum(w[j] * (ms[j] - xbar)^2 for j in 1:k) / (k - 1)
    tmp = sum((one(T) - w[j] / W)^2 / (ns[j] - 1) for j in 1:k)
    denom = one(T) + (2 * (k - 2) / T(k^2 - 1)) * tmp
    F = num / denom
    df1 = T(k - 1)
    df2 = T(k^2 - 1) / (3 * tmp)
    pval = T(ccdf(FDist(df1, df2), F))
    EqualityTestResult{T}(:welch_anova, F, pval, df1, df2, k, ns, true, "Welch (unequal variance)")
end

"""Mann–Whitney U test (two groups). Exact null when tie-free & small; else
continuity- and tie-corrected normal approximation."""
function _mann_whitney(a::Vector{T}, b::Vector{T}) where {T<:AbstractFloat}
    n1 = length(a); n2 = length(b)
    pooled = vcat(a, b)
    r = _tiedrank(pooled)
    R1 = sum(@view r[1:n1])
    U = R1 - n1 * (n1 + 1) / 2                         # R's W statistic
    counts = _value_counts(pooled)
    has_ties = any(c -> c > 1, values(counts))
    if !has_ties && n1 < 50 && n2 < 50
        Ui = Int(round(U))
        half = n1 * n2 / 2
        pv = Ui > half ? (1 - _pwilcox(Ui - 1, n1, n2)) : _pwilcox(Ui, n1, n2)
        pval = T(min(2 * pv, 1.0))
        return EqualityTestResult{T}(:mann_whitney, T(U), pval, T(n1), T(n2), 2, [n1, n2],
                                     true, "exact null")
    else
        N = n1 + n2
        tie = _tie_factor(values(counts))
        sigma = sqrt((n1 * n2 / 12) * ((N + 1) - tie / (N * (N - 1))))
        z0 = U - n1 * n2 / 2
        cc = sign(z0) * 0.5
        z = (z0 - cc) / sigma
        pval = T(min(2 * min(cdf(Normal(), z), ccdf(Normal(), z)), 1.0))
        return EqualityTestResult{T}(:mann_whitney, T(U), pval, T(n1), T(n2), 2, [n1, n2],
                                     false, "normal approx (continuity + tie corrected)")
    end
end

"""Wilcoxon signed-rank test on paired groups `a`,`b` (zeros dropped)."""
function _signed_rank(a::Vector{T}, b::Vector{T}) where {T<:AbstractFloat}
    length(a) == length(b) || throw(ArgumentError("signed-rank requires equal group sizes"))
    d = a .- b
    nz = d[d .!= 0]                                    # drop zero differences (Wilcoxon)
    n = length(nz)
    n >= 1 || throw(ArgumentError("all paired differences are zero"))
    ad = abs.(nz)
    r = _tiedrank(ad)
    V = sum(r[i] for i in 1:n if nz[i] > 0)            # sum of positive ranks
    counts = _value_counts(ad)
    has_ties = any(c -> c > 1, values(counts))
    if !has_ties && n < 50
        Vi = Int(round(V))
        half = n * (n + 1) / 4
        pv = Vi > half ? (1 - _psignrank(Vi - 1, n)) : _psignrank(Vi, n)
        pval = T(min(2 * pv, 1.0))
        return EqualityTestResult{T}(:wilcoxon_signed_rank, T(V), pval, T(n), T(NaN), 2,
                                     [length(a), length(b)], true, "exact null")
    else
        tie = _tie_factor(values(counts))
        sigma = sqrt(n * (n + 1) * (2n + 1) / 24 - tie / 48)
        z0 = V - n * (n + 1) / 4
        cc = sign(z0) * 0.5
        z = (z0 - cc) / sigma
        pval = T(min(2 * min(cdf(Normal(), z), ccdf(Normal(), z)), 1.0))
        return EqualityTestResult{T}(:wilcoxon_signed_rank, T(V), pval, T(n), T(NaN), 2,
                                     [length(a), length(b)], false,
                                     "normal approx (continuity + tie corrected)")
    end
end

"""Kruskal–Wallis H test (tie-corrected)."""
function _kruskal_wallis(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    N = sum(ns)
    pooled = vcat(groups...)
    r = _tiedrank(pooled)
    # rank sums per group (groups are contiguous in `pooled`)
    Hraw = zero(T)
    off = 0
    for j in 1:k
        Rj = sum(@view r[(off + 1):(off + ns[j])])
        Hraw += Rj^2 / ns[j]
        off += ns[j]
    end
    H = 12 / (T(N) * (N + 1)) * Hraw - 3 * (N + 1)
    counts = _value_counts(pooled)
    C = one(T) - _tie_factor(values(counts)) / (T(N)^3 - N)
    Hc = C > 0 ? H / C : H
    df = T(k - 1)
    pval = T(ccdf(Chisq(df), Hc))
    EqualityTestResult{T}(:kruskal_wallis, Hc, pval, df, T(NaN), k, ns, false, "tie-corrected")
end

"""van der Waerden normal-scores test."""
function _van_der_waerden(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    N = sum(ns)
    pooled = vcat(groups...)
    r = _tiedrank(pooled)
    A = T[quantile(Normal(), ri / (N + 1)) for ri in r]
    s2 = sum(a^2 for a in A) / (N - 1)
    stat = zero(T)
    off = 0
    for j in 1:k
        Aj = @view A[(off + 1):(off + ns[j])]
        stat += ns[j] * mean(Aj)^2
        off += ns[j]
    end
    stat /= s2
    df = T(k - 1)
    pval = T(ccdf(Chisq(df), stat))
    EqualityTestResult{T}(:van_der_waerden, stat, pval, df, T(NaN), k, ns, false, "normal scores")
end

"""Mood median (χ²) test: counts above vs at-or-below the grand median."""
function _mood_median(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    N = sum(ns)
    M = median(vcat(groups...))
    above = [count(>(M), gj) for gj in groups]         # x > M
    below = ns .- above
    tot_above = sum(above); tot_below = sum(below)
    stat = zero(T)
    for j in 1:k
        e_above = T(tot_above) * ns[j] / N
        e_below = T(tot_below) * ns[j] / N
        e_above > 0 && (stat += (above[j] - e_above)^2 / e_above)
        e_below > 0 && (stat += (below[j] - e_below)^2 / e_below)
    end
    df = T(k - 1)
    pval = T(ccdf(Chisq(df), stat))
    EqualityTestResult{T}(:median_chisq, stat, pval, df, T(NaN), k, ns, false, "Mood median χ²")
end

# =============================================================================
# Scale tests
# =============================================================================

"""Two-group variance-ratio F test."""
function _var_f_test(a::Vector{T}, b::Vector{T}) where {T<:AbstractFloat}
    n1 = length(a); n2 = length(b)
    F = var(a) / var(b)
    df1 = T(n1 - 1); df2 = T(n2 - 1)
    lower = cdf(FDist(df1, df2), F)
    pval = T(min(2 * min(lower, 1 - lower), 1.0))
    EqualityTestResult{T}(:variance_f, F, pval, df1, df2, 2, [n1, n2], true, "variance ratio")
end

"""Bartlett's χ² test of equal variances."""
function _bartlett(groups::Vector{Vector{T}}) where {T<:AbstractFloat}
    k = length(groups)
    ns = length.(groups)
    N = sum(ns)
    vs = var.(groups)
    sp2 = sum((ns[j] - 1) * vs[j] for j in 1:k) / (N - k)
    num = (N - k) * log(sp2) - sum((ns[j] - 1) * log(vs[j]) for j in 1:k)
    C = one(T) + (sum(one(T) / (ns[j] - 1) for j in 1:k) - one(T) / (N - k)) / (3 * (k - 1))
    stat = num / C
    df = T(k - 1)
    pval = T(ccdf(Chisq(df), stat))
    EqualityTestResult{T}(:bartlett, stat, pval, df, T(NaN), k, ns, false, "sensitive to non-normality")
end

"""Levene-type test: ANOVA F on absolute deviations from the group `center`
(`:mean` = Levene, `:median` = Brown–Forsythe)."""
function _levene(groups::Vector{Vector{T}}, center::Symbol) where {T<:AbstractFloat}
    cfun = center === :median ? median : mean
    devs = [abs.(gj .- cfun(gj)) for gj in groups]
    res = _anova_classic(devs)
    name = center === :median ? :brown_forsythe : :levene
    detail = center === :median ? "deviations from group median" : "deviations from group mean"
    EqualityTestResult{T}(name, res.statistic, res.pvalue, res.df1, res.df2,
                          length(groups), length.(groups), false, detail)
end

"""Siegel–Tukey test of scale (two groups): rank-sum on ranks assigned from the
extremes inward, then a continuity/tie-corrected normal (or exact) rank-sum test."""
function _siegel_tukey(a::Vector{T}, b::Vector{T}) where {T<:AbstractFloat}
    n1 = length(a); n2 = length(b)
    combined = vcat(a, b)
    N = n1 + n2
    ord = sortperm(combined)                           # ascending value order
    st = Vector{Float64}(undef, N)                     # Siegel–Tukey ranks
    lo = 1; hi = N; cnt = 1; side = 1
    while lo <= hi
        if side == 1
            st[ord[lo]] = cnt; cnt += 1; lo += 1
            if lo <= hi
                st[ord[hi]] = cnt; cnt += 1; hi -= 1
            end
            side = 2
        else
            st[ord[hi]] = cnt; cnt += 1; hi -= 1
            if lo <= hi
                st[ord[lo]] = cnt; cnt += 1; lo += 1
            end
            side = 1
        end
    end
    W1 = sum(@view st[1:n1])                            # group-1 ST rank sum
    U = W1 - n1 * (n1 + 1) / 2
    counts = _value_counts(combined)
    has_ties = any(c -> c > 1, values(counts))
    if !has_ties && n1 < 50 && n2 < 50
        Ui = Int(round(U))
        half = n1 * n2 / 2
        pv = Ui > half ? (1 - _pwilcox(Ui - 1, n1, n2)) : _pwilcox(Ui, n1, n2)
        pval = T(min(2 * pv, 1.0))
        return EqualityTestResult{T}(:siegel_tukey, T(W1), pval, T(n1), T(n2), 2, [n1, n2],
                                     true, "exact rank-sum on ST ranks")
    else
        sigma = sqrt(n1 * n2 * (N + 1) / 12)
        z0 = U - n1 * n2 / 2
        cc = sign(z0) * 0.5
        z = (z0 - cc) / sigma
        pval = T(min(2 * min(cdf(Normal(), z), ccdf(Normal(), z)), 1.0))
        return EqualityTestResult{T}(:siegel_tukey, T(W1), pval, T(n1), T(n2), 2, [n1, n2],
                                     false, "normal approx on ST ranks")
    end
end

"""Value → count map (for tie detection / correction)."""
function _value_counts(v::AbstractVector{<:Real})
    d = Dict{Float64,Int}()
    for x in v
        d[Float64(x)] = get(d, Float64(x), 0) + 1
    end
    return d
end

# =============================================================================
# equality_test dispatcher
# =============================================================================

const _TWO_GROUP_TESTS = (:t, :ttest, :mann_whitney, :wilcoxon, :f, :siegel_tukey)

"""
    equality_test(y, g; test::Symbol, equal_var=true, mu=0.0, alpha=0.05) -> EqualityTestResult

Group the response `y` by the distinct values of the classifier `g` and run the
requested equality-of-distribution test.

`test` selects the statistic:

| Location | Scale |
|---|---|
| `:t`, `:anova`, `:mann_whitney`, `:wilcoxon`, `:kruskal_wallis`, `:van_der_waerden`, `:median` | `:f`, `:bartlett`, `:levene`, `:brown_forsythe`, `:siegel_tukey` |

`equal_var` toggles pooled vs Welch (for `:t` and `:anova`). The two-sample /
paired / variance-ratio / Siegel–Tukey tests require exactly two groups; the
others accept `k ≥ 2`. Nonparametric rank tests use the exact null for small
tie-free samples and otherwise the tie- and continuity-corrected normal
approximation.

Also dispatches on `CrossSectionData`/`PanelData` — see the method taking column
symbols. See also [`ttest`](@ref), [`anova_test`](@ref), [`cor_test`](@ref).

# Examples
```julia
equality_test(y, g; test=:kruskal_wallis)
equality_test(y, g; test=:t, equal_var=false)   # Welch
equality_test(y, g; test=:brown_forsythe)
```
"""
function equality_test(y::AbstractVector{<:Real}, g::AbstractVector;
                       test::Symbol, equal_var::Bool=true, mu::Real=0.0, alpha::Real=0.05)
    labels, groups = _split_groups(y, g)
    k = length(groups)
    k >= 2 || throw(ArgumentError("need at least 2 groups, got $k"))
    if test in _TWO_GROUP_TESTS && k != 2
        throw(ArgumentError("test :$test requires exactly 2 groups, got $k"))
    end
    any(isempty, groups) && throw(ArgumentError("every group must be non-empty"))
    T = Float64
    a = groups[1]; b = k == 2 ? groups[2] : Float64[]

    if test === :t || test === :ttest
        return _two_sample_t(a, b, equal_var)
    elseif test === :anova
        return equal_var ? _anova_classic(groups) : _anova_welch(groups)
    elseif test === :mann_whitney
        return _mann_whitney(a, b)
    elseif test === :wilcoxon
        return _signed_rank(a, b)
    elseif test === :kruskal_wallis
        return _kruskal_wallis(groups)
    elseif test === :van_der_waerden
        return _van_der_waerden(groups)
    elseif test === :median
        return _mood_median(groups)
    elseif test === :f
        return _var_f_test(a, b)
    elseif test === :bartlett
        return _bartlett(groups)
    elseif test === :levene
        return _levene(groups, :mean)
    elseif test === :brown_forsythe
        return _levene(groups, :median)
    elseif test === :siegel_tukey
        return _siegel_tukey(a, b)
    else
        throw(ArgumentError("unknown test :$test"))
    end
end

# =============================================================================
# Convenience wrappers
# =============================================================================

"""
    ttest(x; mu=0.0)              -> EqualityTestResult    # one-sample
    ttest(x, y; paired=false, equal_var=true) -> EqualityTestResult

Student's t-test. One-sample tests `H₀: mean(x) = μ₀`; the two-argument form runs
a paired (`paired=true`) or two-sample (pooled `equal_var=true`, Welch otherwise)
test. See also [`equality_test`](@ref).
"""
function ttest(x::AbstractVector{<:Real}; mu::Real=0.0)
    xf = float.(collect(x))
    _one_sample_t(xf, Float64(mu))
end

function ttest(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
               paired::Bool=false, equal_var::Bool=true)
    xf = float.(collect(x)); yf = float.(collect(y))
    paired ? _paired_t(xf, yf) : _two_sample_t(xf, yf, equal_var)
end

"""
    anova_test(y, g; equal_var=true) -> EqualityTestResult

One-way ANOVA of `y` across the groups defined by `g`: the classic
equal-variance F when `equal_var=true`, else Welch's unequal-variance F.
"""
function anova_test(y::AbstractVector{<:Real}, g::AbstractVector; equal_var::Bool=true)
    equality_test(y, g; test=:anova, equal_var=equal_var)
end

# =============================================================================
# CrossSectionData / PanelData convenience methods (pull columns by symbol)
# =============================================================================

function equality_test(d::CrossSectionData, response::Symbol, class::Symbol; kwargs...)
    yi = findfirst(==(String(response)), d.varnames)
    yi === nothing && throw(ArgumentError("Variable '$(response)' not found. Available: $(d.varnames)"))
    gi = findfirst(==(String(class)), d.varnames)
    gi === nothing && throw(ArgumentError("Variable '$(class)' not found. Available: $(d.varnames)"))
    equality_test(d.data[:, yi], d.data[:, gi]; kwargs...)
end

function equality_test(d::PanelData, response::Symbol, class::Symbol; kwargs...)
    yi = _panel_varindex(d, response)
    gi = _panel_varindex(d, class)
    equality_test(d.data[:, yi], d.data[:, gi]; kwargs...)
end

function cor_test(d::CrossSectionData, x::Symbol, y::Symbol; kwargs...)
    xi = findfirst(==(String(x)), d.varnames)
    xi === nothing && throw(ArgumentError("Variable '$(x)' not found. Available: $(d.varnames)"))
    yi = findfirst(==(String(y)), d.varnames)
    yi === nothing && throw(ArgumentError("Variable '$(y)' not found. Available: $(d.varnames)"))
    cor_test(d.data[:, xi], d.data[:, yi]; kwargs...)
end

function cor_test(d::PanelData, x::Symbol, y::Symbol; kwargs...)
    xi = _panel_varindex(d, x)
    yi = _panel_varindex(d, y)
    cor_test(d.data[:, xi], d.data[:, yi]; kwargs...)
end
