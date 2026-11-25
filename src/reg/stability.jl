# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Parameter-stability and influence diagnostics for a fitted `RegModel` (EV-32, #440):

- Brown–Durbin–Evans (1975) recursive least-squares residuals (`recursive_residuals`).
- The recursive-residual CUSUM (`cusum_test`) and CUSUM-of-squares (`cusumsq_test`)
  parameter-stability tests, returning a `StabilityResult`.
- Chow (1960) known-break-date tests (`chow_test`): the breakpoint F test (single or
  multiple breaks) and the predictive-failure/forecast F test.
- Observation-level influence statistics (`influence_stats` → `InfluenceStats`):
  leverage, internally/externally studentized residuals, DFFITS, Cook's D, DFBETAS.

For the *unknown* break-date sup-Wald problem use [`andrews_test`](@ref)
(Quandt–Andrews), which this module cross-links rather than duplicates.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Recursive residuals — Brown–Durbin–Evans (1975)
# =============================================================================

# Non-mutating Sherman–Morrison recursive-least-squares pass. Given a design `X`
# (n×k) and response `y`, returns the standardized recursive residuals
#   w_t = (y_t − x_t' b_{t−1}) / √(1 + x_t' (X_{t−1}'X_{t−1})⁻¹ x_t),   t = k+1 … n
# updating the running `(X'X)⁻¹` and coefficient vector by rank-one formulae
# (O(nk²), no per-step matrix inverse).
function _recursive_residuals(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, k = size(X)
    n > k || throw(ArgumentError("recursive residuals need n > k (got n=$n, k=$k)"))
    # Initialize from the first k observations. If that block is rank-deficient,
    # extend the burn-in until the running design reaches full rank.
    X0 = Matrix{T}(X[1:k, :])
    A = X0' * X0
    Ainv = robust_inv(A)
    b = Ainv * (X0' * y[1:k])
    start = k + 1
    # Guard a singular initial block: grow the burn-in window to full rank.
    while start <= n && !(all(isfinite, Ainv) && cond(Symmetric(A)) < T(1e12))
        xt = Vector{T}(X[start, :])
        A = A + xt * xt'
        Ainv = robust_inv(A)
        b = Ainv * (X[1:start, :]' * y[1:start])
        start += 1
    end

    w = Vector{T}(undef, n - k)
    fill!(w, zero(T))
    @inbounds for t in start:n
        xt = Vector{T}(X[t, :])
        Ax = Ainv * xt
        f = one(T) + dot(xt, Ax)              # 1 + x_t' (X_{t-1}'X_{t-1})⁻¹ x_t
        pred_err = y[t] - dot(xt, b)
        w[t-k] = pred_err / sqrt(f)
        # Recursive-least-squares update (Sherman–Morrison):
        K = Ax ./ f                           # Kalman gain
        b = b .+ K .* pred_err
        Ainv = Ainv .- (Ax * Ax') ./ f        # (A + x x')⁻¹
    end
    w
end

"""
    recursive_residuals(m::RegModel{T}) -> Vector{T}

Standardized Brown–Durbin–Evans (1975) recursive least-squares residuals of a fitted
OLS/WLS [`RegModel`](@ref). Processing observations `t = k+1 … n` with a growing-window
OLS fit `b_{t−1}` on the first `t−1` rows, the `t`-th standardized recursive residual is

```
w_t = (y_t − x_t' b_{t−1}) / √(1 + x_t' (X_{t−1}'X_{t−1})⁻¹ x_t)
```

which under a stable model is i.i.d. `N(0, σ²)`. Returns a vector of length `n − k`
(one per post-burn-in observation, in time order). Their sum of squares equals the
full-sample OLS SSR — the basis of the CUSUM-of-squares test.

The running `(X'X)⁻¹` and coefficient vector are updated by rank-one Sherman–Morrison
formulae, not recomputed each step. See [`cusum_test`](@ref), [`cusumsq_test`](@ref).

# References
- Brown, R. L., Durbin, J. & Evans, J. M. (1975). *JRSS-B* 37(2), 149–192.
"""
recursive_residuals(m::RegModel{T}) where {T<:AbstractFloat} =
    _recursive_residuals(m.y, m.X)

# =============================================================================
# CUSUM — Brown–Durbin–Evans (1975)
# =============================================================================

# BDE 5%/1%/10% line coefficient `a`: the boundary is
#   ± a √(n−k) · (1 + 2(t−k)/(n−k)).
# Values from Brown, Durbin & Evans (1975, p. 154).
function _cusum_a(level::Real)
    lv = round(Float64(level); digits=4)
    lv == 0.05 && return 0.948
    lv == 0.01 && return 1.143
    lv == 0.10 && return 0.850
    throw(ArgumentError("cusum level must be 0.01, 0.05, or 0.10 (got $level)"))
end

"""
    cusum_test(m::RegModel{T}; level=0.05) -> StabilityResult{T}

Brown–Durbin–Evans (1975) CUSUM test for parameter stability of a fitted OLS/WLS
[`RegModel`](@ref). Builds the cumulative sum of standardized recursive residuals

```
W_t = (1/σ̂_w) Σ_{j=k+1}^{t} w_j,   σ̂_w = √(Σ w_j² / (n−k)),
```

and compares it with the pair of significance lines

```
± a √(n−k) · (1 + 2(t−k)/(n−k)),
```

with `a = 0.948` (5%), `1.143` (1%), or `0.850` (10%) from Brown, Durbin & Evans
(1975, p. 154). The bounds are a **pair of straight lines in `t`**, not a flat band;
`W_t` wandering outside them signals coefficient instability (typically a slow drift).

For the CUSUM-of-squares variant see [`cusumsq_test`](@ref); for the unknown-break
sup-Wald test see [`andrews_test`](@ref).

# References
- Brown, R. L., Durbin, J. & Evans, J. M. (1975). *JRSS-B* 37(2), 149–192.
"""
function cusum_test(m::RegModel{T}; level::Real=0.05) where {T<:AbstractFloat}
    w = _recursive_residuals(m.y, m.X)
    n = length(m.y)
    k = size(m.X, 2)
    nr = n - k
    a = T(_cusum_a(level))
    sigma_w = sqrt(dot(w, w) / T(nr))
    sigma_w = max(sigma_w, T(1e-300))
    Wt = cumsum(w) ./ sigma_w
    tindex = collect((k+1):n)
    upper = Vector{T}(undef, nr)
    lower = Vector{T}(undef, nr)
    crossed = false
    first_cross = nothing
    sqrtnr = sqrt(T(nr))
    @inbounds for i in 1:nr
        t = tindex[i]
        bnd = a * sqrtnr * (one(T) + T(2) * T(t - k) / T(nr))
        upper[i] = bnd
        lower[i] = -bnd
        if !crossed && abs(Wt[i]) > bnd
            crossed = true
            first_cross = t
        end
    end
    StabilityResult{T}(:cusum, tindex, Wt, upper, lower, crossed, first_cross,
                       T(level), w, n, k)
end

# =============================================================================
# CUSUM of squares — Brown–Durbin–Evans (1975) / Edgerton–Wells (1994)
# =============================================================================

# Edgerton–Wells (1994) approximation to the CUSUMSQ critical value c₀, as used by
# statsmodels' RecursiveLS. With m = (n−k)/2 − 1 and column `ix` selected by the
# two-sided level,
#   c₀ = s₀/√m + s₁/m + s₂/m^{3/2}.
# The band is (t−k)/(n−k) ± c₀. Scalars: Edgerton & Wells (1994, Table 1).
const _CUSUMSQ_SCALARS = [
    1.072983  1.2238734  1.3581015  1.5174271  1.6276236;
   -0.6698868 -0.6700069 -0.6701218 -0.6702672 -0.6703724;
   -0.5816458 -0.7351697 -0.8858694 -1.0847745 -1.2365861
]
# Column index by *two-sided* significance `level`. The five columns are the
# half-levels [0.10, 0.05, 0.025, 0.01, 0.005] (Edgerton–Wells / statsmodels),
# so a two-sided level `α` selects the column at half-level `α/2`:
#   α = 0.20→col1, 0.10→col2, 0.05→col3, 0.02→col4, 0.01→col5.
function _cusumsq_c0(nr::Int, level::Real)
    lv = round(Float64(level); digits=4)
    ix = lv == 0.20 ? 1 : lv == 0.10 ? 2 : lv == 0.05 ? 3 :
         lv == 0.02 ? 4 : lv == 0.01 ? 5 :
         throw(ArgumentError("cusumsq level must be 0.20/0.10/0.05/0.02/0.01 (got $level)"))
    mparam = 0.5 * nr - 1.0
    mparam <= 0 && throw(ArgumentError("CUSUMSQ needs n−k > 2 (got n−k=$nr)"))
    s = _CUSUMSQ_SCALARS[:, ix]
    s[1] / sqrt(mparam) + s[2] / mparam + s[3] / mparam^1.5
end

"""
    cusumsq_test(m::RegModel{T}; level=0.05) -> StabilityResult{T}

Brown–Durbin–Evans (1975) CUSUM-of-squares test for parameter/variance stability of a
fitted OLS/WLS [`RegModel`](@ref). Uses the normalized cumulative sum of squared
recursive residuals

```
S_t = Σ_{j=k+1}^{t} w_j² / Σ_{j=k+1}^{n} w_j²,
```

which rises monotonically from `0` to exactly `1` and has mean line `(t−k)/(n−k)` under
stability. The significance band is `(t−k)/(n−k) ± c₀`, where `c₀` is the
Edgerton–Wells (1994) approximation to Durbin's critical value

```
c₀ = s₀/√m + s₁/m + s₂/m^{3/2},   m = (n−k)/2 − 1,
```

(the same approximation used by statsmodels' `RecursiveLS`; scalars from Edgerton &
Wells 1994, Table 1). CUSUMSQ is sensitive to a *one-off* variance shift, complementing
the drift-sensitive [`cusum_test`](@ref).

# References
- Brown, R. L., Durbin, J. & Evans, J. M. (1975). *JRSS-B* 37(2), 149–192.
- Edgerton, D. & Wells, C. (1994). *Oxford Bull. Econ. Stat.* 56(3), 355–365.
"""
function cusumsq_test(m::RegModel{T}; level::Real=0.05) where {T<:AbstractFloat}
    w = _recursive_residuals(m.y, m.X)
    n = length(m.y)
    k = size(m.X, 2)
    nr = n - k
    c0 = T(_cusumsq_c0(nr, level))
    total = max(dot(w, w), T(1e-300))
    St = cumsum(w .^ 2) ./ total
    tindex = collect((k+1):n)
    upper = Vector{T}(undef, nr)
    lower = Vector{T}(undef, nr)
    crossed = false
    first_cross = nothing
    @inbounds for i in 1:nr
        t = tindex[i]
        mean_line = T(t - k) / T(nr)
        upper[i] = mean_line + c0
        lower[i] = mean_line - c0
        if !crossed && (St[i] > upper[i] || St[i] < lower[i])
            crossed = true
            first_cross = t
        end
    end
    StabilityResult{T}(:cusumsq, tindex, St, upper, lower, crossed, first_cross,
                       T(level), w, n, k)
end

# =============================================================================
# Chow (1960) known-break-date tests
# =============================================================================

# OLS SSR of `y` on `X` (with the model's own columns), collinearity-guarded.
function _ols_ssr(y::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    beta = robust_inv(X' * X) * (X' * y)
    resid = y .- X * beta
    dot(resid, resid)
end

"""
    chow_test(m::RegModel{T}, break_index; type=:breakpoint, level=0.05) -> RegDiagnosticResult{T}

Chow (1960) known-break-date structural-stability test on a fitted OLS
[`RegModel`](@ref). `break_index` is an observation index (or a sorted vector of
indices) that partitions the sample; each break index is the **last** observation of
the preceding segment.

- `type=:breakpoint` (default): estimate the model separately on each of the `m+1`
  segments and test coefficient equality. With `SSR_u = Σ SSR_seg` and restricted
  `SSR_r` from the pooled fit,
  `F = [(SSR_r − SSR_u)/(k·m)] / [SSR_u/(n − k·(m+1))] ~ F(k·m, n − k·(m+1))`.
  Each segment must contain at least `k` observations.
- `type=:forecast`: Chow's predictive-failure test with a single split at `n₁ =
  break_index`. The first `n₁` observations estimate the model and the remaining
  `n₂ = n − n₁` are the forecast period:
  `F = [(SSR_r − SSR₁)/n₂] / [SSR₁/(n₁ − k)] ~ F(n₂, n₁ − k)`,
  valid (and the only option) when a segment is too short to estimate `k` coefficients.

For an *unknown* break date, use [`andrews_test`](@ref) (Quandt–Andrews sup-Wald),
which searches over candidate break points instead of testing a pre-specified one.

# References
- Chow, G. C. (1960). *Econometrica* 28(3), 591–605.
"""
function chow_test(m::RegModel{T}, break_index::Union{Integer,AbstractVector{<:Integer}};
                   type::Symbol=:breakpoint, level::Real=0.05) where {T<:AbstractFloat}
    y = m.y
    X = Matrix{T}(m.X)
    n, k = size(X)
    breaks = sort(collect(Int, break_index isa Integer ? [break_index] : break_index))
    all(b -> 1 <= b < n, breaks) ||
        throw(ArgumentError("break index/indices must lie in 1:$(n-1) (got $breaks)"))
    ssr_r = _ols_ssr(y, X)

    if type == :breakpoint
        # Segment boundaries: 1..b1, b1+1..b2, …, bm+1..n.
        edges = vcat(0, breaks, n)
        nseg = length(edges) - 1
        ssr_u = zero(T)
        for s in 1:nseg
            lo = edges[s] + 1
            hi = edges[s+1]
            (hi - lo + 1) >= k ||
                throw(ArgumentError("segment $s has $(hi-lo+1) < k=$k observations; use type=:forecast"))
            ssr_u += _ols_ssr(y[lo:hi], X[lo:hi, :])
        end
        mbreaks = length(breaks)
        df1 = k * mbreaks
        df2 = n - k * (nseg)
        f = (df2 >= 1 && ssr_u > zero(T)) ?
            ((ssr_r - ssr_u) / T(df1)) / (ssr_u / T(df2)) : zero(T)
        pval = _f_p(f, df1, df2)
        name = mbreaks == 1 ? "Chow breakpoint test (break=$(breaks[1]))" :
               "Chow breakpoint test (breaks=$(breaks))"
        h0 = "Coefficients constant across the $(nseg) sub-samples"
        return RegDiagnosticResult{T}(name, h0, f, pval, (df1, df2),
                                      nothing, nothing, nothing, zero(T), n)
    elseif type == :forecast
        length(breaks) == 1 ||
            throw(ArgumentError("forecast Chow test takes a single break index (got $breaks)"))
        n1 = breaks[1]
        n1 > k || throw(ArgumentError("forecast test needs break index > k=$k (got $n1)"))
        n2 = n - n1
        n2 >= 1 || throw(ArgumentError("forecast test needs at least one out-of-sample obs"))
        ssr1 = _ols_ssr(y[1:n1], X[1:n1, :])
        df1 = n2
        df2 = n1 - k
        f = (df2 >= 1 && ssr1 > zero(T)) ?
            ((ssr_r - ssr1) / T(n2)) / (ssr1 / T(df2)) : zero(T)
        pval = _f_p(f, df1, df2)
        name = "Chow forecast test (split=$(n1))"
        h0 = "No predictive failure in the $(n2)-observation forecast period"
        return RegDiagnosticResult{T}(name, h0, f, pval, (df1, df2),
                                      nothing, nothing, nothing, zero(T), n)
    else
        throw(ArgumentError("chow_test type must be :breakpoint or :forecast (got $type)"))
    end
end

# =============================================================================
# Influence statistics — Belsley, Kuh & Welsch (1980)
# =============================================================================

"""
    influence_stats(m::RegModel{T}) -> InfluenceStats{T}

Observation-level leverage and influence diagnostics for a fitted OLS
[`RegModel`](@ref), following Belsley, Kuh & Welsch (1980). Reuses the fitted `(X'X)⁻¹`
(from the model's OLS covariance) — it never recomputes an inverse per observation.

Per observation `i` (with residual `r_i`, `σ̂ = √(SSR/(n−k))`, and hat diagonal
`h_ii = x_i'(X'X)⁻¹x_i`):

- `hat` — leverage `h_ii ∈ (0,1)`.
- `student_internal` — internally studentized residual `r_i / (σ̂√(1−h_ii))` (R `rstandard`).
- `student_external` — externally studentized residual `t*_i` using the
  leave-one-out variance `σ̂_{(i)}` (R `rstudent`).
- `dffits` — `t*_i √(h_ii/(1−h_ii))`.
- `cooksd` — Cook's distance `(r_i²/(k σ̂²)) · (h_ii/(1−h_ii)²)` = `(t_i²/k)·(h_ii/(1−h_ii))`.
- `dfbetas` — `n × k` matrix; column `j` is the standardized change in `β̂_j` from
  deleting observation `i`.

`high_leverage` flags `h_ii > 2k/n` and `influential` flags `|DFFITS_i| > 2√(k/n)`
(the BKW size-adjusted cutoffs). A leverage-1 point is guarded so its denominators do
not blow up; it is still flagged in `high_leverage`.

For the *unknown* break-date sup-Wald test see [`andrews_test`](@ref).

# References
- Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics*. Wiley.
- Cook, R. D. (1977). *Technometrics* 19(1), 15–18.
"""
function influence_stats(m::RegModel{T}) where {T<:AbstractFloat}
    X = Matrix{T}(m.X)
    n, k = size(X)
    resid = m.residuals
    ssr = m.ssr
    dfres = n - k
    dfres >= 1 || throw(ArgumentError("influence statistics need n > k (got n=$n, k=$k)"))
    sigma2 = ssr / T(dfres)
    sigma = sqrt(max(sigma2, T(0)))

    XtXinv = robust_inv(Symmetric(X' * X))
    # Hat diagonals in one pass: h_ii = x_i' (X'X)⁻¹ x_i.
    H = X * XtXinv                     # n × k, row i = x_i'(X'X)⁻¹
    hat = Vector{T}(undef, n)
    @inbounds for i in 1:n
        hat[i] = clamp(dot(view(H, i, :), view(X, i, :)), zero(T), one(T))
    end

    student_int = Vector{T}(undef, n)
    student_ext = Vector{T}(undef, n)
    dffits = Vector{T}(undef, n)
    cooksd = Vector{T}(undef, n)
    dfbetas = zeros(T, n, k)

    tiny = T(1e-12)
    @inbounds for i in 1:n
        omh = one(T) - hat[i]
        if omh <= tiny
            # Exact-fit / leverage-1 point: influence measures undefined. Flag via
            # leverage and leave the standardized quantities at zero rather than ±Inf.
            student_int[i] = zero(T)
            student_ext[i] = zero(T)
            dffits[i] = zero(T)
            cooksd[i] = zero(T)
            continue
        end
        ri = resid[i]
        # Internally studentized residual.
        s_int = ri / (sigma * sqrt(omh))
        student_int[i] = s_int
        # Leave-one-out variance σ̂²_{(i)} and externally studentized residual.
        s2_i = (T(dfres) * sigma2 - ri^2 / omh) / T(dfres - 1)
        s2_i = max(s2_i, T(0))
        sig_i = sqrt(s2_i)
        s_ext = sig_i > tiny ? ri / (sig_i * sqrt(omh)) : zero(T)
        student_ext[i] = s_ext
        dffits[i] = s_ext * sqrt(hat[i] / omh)
        cooksd[i] = (s_int^2 / T(k)) * (hat[i] / omh)
        # DFBETAS: column j = (X'X)⁻¹_{·,j}' x_i · r_i / (σ̂_{(i)} √((X'X)⁻¹_{jj} (1−h_ii))).
        if sig_i > tiny
            ci = XtXinv * view(X, i, :)          # (X'X)⁻¹ x_i, length k
            denom_scale = ri / (sig_i * omh)
            for j in 1:k
                gjj = XtXinv[j, j]
                dfbetas[i, j] = gjj > tiny ? ci[j] * denom_scale / sqrt(gjj) : zero(T)
            end
        end
    end

    hi_cut = T(2 * k) / T(n)
    inf_cut = T(2) * sqrt(T(k) / T(n))
    high_leverage = findall(h -> h > hi_cut, hat)
    influential = findall(d -> abs(d) > inf_cut, dffits)

    InfluenceStats{T}(hat, student_int, student_ext, dffits, cooksd, dfbetas,
                      sigma, high_leverage, influential, copy(m.varnames), n, k)
end

# =============================================================================
# Display — StabilityResult
# =============================================================================

function Base.show(io::IO, r::StabilityResult{T}) where {T}
    test_name = r.kind == :cusum ? "CUSUM test (recursive residuals)" :
                                    "CUSUM of squares test"
    lvl_pct = round(Int, 100 * r.level)
    decision = r.crossed ?
        "Reject stability — path crosses the $(lvl_pct)% bound" *
            (r.first_crossing === nothing ? "" : " at obs $(r.first_crossing)") :
        "Fail to reject stability — path within the $(lvl_pct)% bounds"
    max_abs = isempty(r.stat_path) ? zero(T) : maximum(abs, r.stat_path)
    rows = Any[
        "Test"             test_name;
        "Recursive resid." length(r.recursive_resid);
        "Observations"     r.n;
        "Regressors"       r.k;
        "Level"            "$(lvl_pct)%";
        (r.kind == :cusum ? "max |W_t|" : "max |S_t|")  _fmt(max_abs);
        "Crosses bound"    (r.crossed ? "Yes" : "No");
        "Conclusion"       decision;
    ]
    _pretty_table(io, rows; title = test_name,
                  column_labels = ["", ""], alignment = [:l, :r])
end

report(r::StabilityResult) = show(stdout, r)

# =============================================================================
# Display — InfluenceStats
# =============================================================================

function Base.show(io::IO, s::InfluenceStats{T}) where {T}
    spec = Any[
        "Observations"    s.n;
        "Regressors"      s.k;
        "σ̂ (resid. s.e.)" _fmt(s.sigma);
        "Leverage cutoff" _fmt(T(2 * s.k) / T(s.n)) * " (2k/n)";
        "DFFITS cutoff"   _fmt(T(2) * sqrt(T(s.k) / T(s.n))) * " (2√(k/n))";
        "High-leverage"   length(s.high_leverage);
        "Influential"     length(s.influential);
    ]
    _pretty_table(io, spec; title = "Influence Diagnostics",
                  column_labels = ["Summary", ""], alignment = [:l, :r])

    # Rank the most influential observations by |DFFITS| and tabulate.
    order = sortperm(abs.(s.dffits); rev = true)
    top = order[1:min(length(order), 10)]
    data = Matrix{Any}(undef, length(top), 5)
    for (row, i) in enumerate(top)
        data[row, 1] = i
        data[row, 2] = _fmt(s.hat[i])
        data[row, 3] = _fmt(s.student_external[i])
        data[row, 4] = _fmt(s.dffits[i])
        data[row, 5] = _fmt(s.cooksd[i])
    end
    _pretty_table(io, data; title = "Most influential observations (top $(length(top)) by |DFFITS|)",
                  column_labels = ["Obs", "Leverage", "Stud.resid", "DFFITS", "Cook's D"],
                  alignment = [:r, :r, :r, :r, :r])
end

report(s::InfluenceStats) = show(stdout, s)
