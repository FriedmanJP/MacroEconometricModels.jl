# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Westerlund (2007) error-correction-based panel cointegration tests. For each unit
an ECM `Δy_it = δ_i d_t + α_i y_{i,t-1} + λ_i' x_{i,t-1} + short-run + e_it` is
estimated; the four statistics test `H0: α_i = 0` (no error correction ⇒ no
cointegration). Two group-mean statistics (`Gt`, `Ga`) average per-unit
quantities; two panel statistics (`Pt`, `Pa`) pool a common error-correction
coefficient. All are standardized to N(0,1) with moments from Westerlund (2007,
Table 1) — as tabulated in Persyn & Westerlund's Stata `xtwest` — and are
**left-tailed** (large negative rejects). An optional seeded bootstrap gives
p-values robust to cross-sectional dependence.

References:
- Westerlund, J. (2007). Testing for Error Correction in Panel Data. Oxford
  Bulletin of Economics and Statistics, 69(6), 709-748.
- Persyn, D. & Westerlund, J. (2008). Error-correction-based cointegration tests
  for panel data. Stata Journal, 8(2), 232-241.
"""

# =============================================================================
# Westerlund (2007) Table 1 standardization moments (mean μ, variance v).
# Transcribed VERBATIM from the `xtwest`/`Westerlund` reference implementation
# (DisplayWesterlund: gtmean/gamean/ptmean/pamean, gtvar/gavar/ptvar/pavar).
# Rows: deterministic case (1=:none, 2=:constant, 3=:trend); columns: number of
# regressors nox ∈ 1:6. Standardization (all left-tailed, p = Φ(z)):
#   Z(Gt) = (√N·Gt − √N·μ)/√v,  Z(Ga) = (√N·Ga − √N·μ)/√v,
#   Z(Pt) = (Pt   − √N·μ)/√v,   Z(Pa) = (√N·Pa − √N·μ)/√v.
# =============================================================================

const _WEST_GT_MU = [
    -0.9763 -1.3816 -1.7093 -1.9789 -2.1985 -2.4262;
    -1.7776 -2.0349 -2.2332 -2.4453 -2.6462 -2.8358;
    -2.3664 -2.5284 -2.7040 -2.8639 -3.0146 -3.1710]
const _WEST_GA_MU = [
    -3.8022  -5.8239  -7.8108  -9.8791 -11.7239 -13.8581;
    -7.1423  -9.1249 -10.9667 -12.9561 -14.9752 -17.0673;
    -12.0116 -13.6324 -15.5262 -17.3648 -19.2533 -21.2479]
const _WEST_PT_MU = [
    -0.5105 -0.9370 -1.3169 -1.6167 -1.8815 -2.1256;
    -1.4476 -1.7131 -1.9206 -2.1484 -2.3730 -2.5765;
    -2.1124 -2.2876 -2.4633 -2.6275 -2.7858 -2.9537]
const _WEST_PA_MU = [
    -1.0263  -2.4988  -4.2699  -6.1141  -8.0317 -10.0074;
    -4.2303  -5.8650  -7.4599  -9.3057 -11.3152 -13.3180;
    -8.9326 -10.4874 -12.1672 -13.8889 -15.6815 -17.6515]
const _WEST_GT_V = [
    1.0823 1.0981 1.0489 1.0576 1.0351 1.0409;
    0.8071 0.8481 0.8886 0.9119 0.9083 0.9236;
    0.6603 0.7070 0.7586 0.8228 0.8477 0.8599]
const _WEST_GA_V = [
    20.6868 29.9016 39.0109 50.5741 58.9595 69.5967;
    29.6336 39.3428 49.4880 58.7035 67.9499 79.1093;
    46.2420 53.7428 64.5591 74.7403 84.7990 94.0024]
const _WEST_PT_V = [
    1.3624 1.7657 1.7177 1.6051 1.4935 1.4244;
    0.9885 1.0663 1.1168 1.1735 1.1684 1.1589;
    0.7649 0.8137 0.8857 0.9985 0.9918 0.9898]
const _WEST_PA_V = [
    8.3827 24.0223 39.8827 53.4518 63.2406 76.6757;
    19.7090 31.2637 42.9975 57.4844 69.4374 81.0384;
    37.5948 45.6890 57.9985 74.1258 81.3934 91.2392]

# Positional lag (NaN-filled) and first difference for a balanced series.
_wlag(v::AbstractVector{T}, k::Int) where {T} =
    k == 0 ? collect(v) : T[i > k ? v[i-k] : T(NaN) for i in eachindex(v)]
_wdiff(v::AbstractVector{T}) where {T} =
    T[i > 1 ? v[i] - v[i-1] : T(NaN) for i in eachindex(v)]
# Positional lead (NaN at the tail).
_wlead(v::AbstractVector{T}, k::Int) where {T} =
    k == 0 ? collect(v) : T[i + k <= length(v) ? v[i+k] : T(NaN) for i in eachindex(v)]

_finite_mask(cols::Vector{Vector{T}}) where {T} =
    [all(c -> isfinite(c[t]), cols) for t in 1:length(cols[1])]

# OLS on the finite rows of [dep, cols...]; returns (coef, se, resid_full_vec,
# fitted_minus?). resid is defined only on the finite rows (NaN elsewhere).
function _wols(dep::Vector{T}, cols::Vector{Vector{T}}) where {T<:AbstractFloat}
    n = length(dep)
    mask = [isfinite(dep[t]) && all(c -> isfinite(c[t]), cols) for t in 1:n]
    idx = findall(mask)
    m = length(idx)
    W = Matrix{T}(undef, m, length(cols))
    for (jj, c) in enumerate(cols), (ii, t) in enumerate(idx)
        W[ii, jj] = c[t]
    end
    yv = dep[idx]
    WtW_inv = robust_inv(W'W)
    b = WtW_inv * (W'yv)
    res = yv - W * b
    dof = max(m - length(cols), 1)
    s2 = dot(res, res) / dof
    se = T[sqrt(max(s2 * WtW_inv[j, j], zero(T))) for j in 1:length(cols)]
    resid_full = fill(T(NaN), n)
    resid_full[idx] .= res
    (b, se, resid_full, mask, dot(res, res))
end

# =============================================================================
# westerlund_test
# =============================================================================

"""
    westerlund_test(pd::PanelData, y::Symbol, xs::Symbol...; trend=:constant,
                    lags=1, leads=0, lrwindow=2, bootstrap=0, seed=20240716)
        -> WesterlundResult

Westerlund (2007) ECM panel cointegration test. Estimates a per-unit
error-correction model of `Δy` and reports the four statistics `Gt`, `Ga`, `Pt`,
`Pa` (H0: no cointegration).

# Keyword Arguments
- `trend`: `:none`, `:constant` (default, an intercept), or `:trend`
  (intercept + linear trend) in the ECM.
- `lags`: short-run lag order `p` (default 1).
- `leads`: short-run lead order `q` of `Δx` (default 0).
- `lrwindow`: Bartlett window for the long-run variances (default 2).
- `bootstrap`: if `> 0`, number of seeded bootstrap replications for
  CSD-robust p-values (stored alongside the asymptotic ones); `0` (default)
  skips the bootstrap.
- `seed`: RNG seed for the bootstrap (stored in the result for reproducibility).

# Example
```julia
pd = xtset(df, :country, :year)
res = westerlund_test(pd, :lny, :lnx; trend=:constant, lags=1, leads=0)
res.pvalues                 # asymptotic (Gt, Ga, Pt, Pa)
```

# References
- Westerlund (2007), OBES 69(6); Persyn & Westerlund (2008), Stata Journal 8(2).
"""
function westerlund_test(pd::PanelData{TT}, y::Symbol, xs::Symbol...;
                         trend::Symbol=:constant,
                         lags::Int=1, leads::Int=0, lrwindow::Int=2,
                         bootstrap::Int=0, seed::Int=20240716) where {TT}
    trend in (:none, :constant, :trend) || throw(ArgumentError(
        "trend must be :none, :constant, or :trend, got :$trend"))
    isempty(xs) && throw(ArgumentError("westerlund_test needs at least one regressor"))
    length(xs) <= 6 || throw(ArgumentError("Westerlund test supports at most 6 regressors"))
    T = float(TT)
    Y, X = _panel_coint_data(pd, y, xs)
    return _westerlund_core(Y, X, trend, lags, leads, lrwindow, bootstrap, seed)
end

function _westerlund_core(Y::Matrix{T}, X::Array{T,3}, trend::Symbol,
                          p::Int, q::Int, lrwindow::Int,
                          bootstrap::Int, seed::Int) where {T<:AbstractFloat}
    Tobs, N = size(Y)
    k = size(X, 3)
    N < 2 && throw(ArgumentError("Westerlund test needs at least N=2 units, got N=$N"))
    has_c = trend != :none
    has_tr = trend == :trend
    row_idx = Int(has_c) + Int(has_tr) + 1

    gt, ga, pt, pa = _west_raw_stats(Y, X, trend, p, q, lrwindow)

    sqrtN = sqrt(T(N))
    zgt = (sqrtN * gt - sqrtN * T(_WEST_GT_MU[row_idx, k])) / sqrt(T(_WEST_GT_V[row_idx, k]))
    zga = (sqrtN * ga - sqrtN * T(_WEST_GA_MU[row_idx, k])) / sqrt(T(_WEST_GA_V[row_idx, k]))
    zpt = (pt - sqrtN * T(_WEST_PT_MU[row_idx, k])) / sqrt(T(_WEST_PT_V[row_idx, k]))
    zpa = (sqrtN * pa - sqrtN * T(_WEST_PA_MU[row_idx, k])) / sqrt(T(_WEST_PA_V[row_idx, k]))
    zs = T[zgt, zga, zpt, zpa]
    pvals = T[cdf(Normal(), z) for z in zs]           # left-tailed

    boot_p = fill(T(NaN), 4)
    if bootstrap > 0
        boot_p = _west_bootstrap(Y, X, trend, p, q, lrwindow, bootstrap, seed,
                                 T[gt, ga, pt, pa])
    end

    WesterlundResult{T}(["Gt", "Ga", "Pt", "Pa"], T[gt, ga, pt, pa], zs, pvals,
                        boot_p, trend, k, p, q, lrwindow, bootstrap, seed, Tobs, N)
end

# Raw Gt, Ga, Pt, Pa for a balanced panel (ports Westerlund/xtwest `WesterlundPlain`
# for fixed lags/leads, non-`westerlund` normalization).
function _west_raw_stats(Y::Matrix{T}, X::Array{T,3}, trend::Symbol,
                         p::Int, q::Int, lrwindow::Int) where {T<:AbstractFloat}
    Tobs, N = size(Y)
    k = size(X, 3)
    has_c = trend != :none
    has_tr = trend == :trend
    nc = Int(has_c)
    ntr = Int(has_tr)
    kp = nc + ntr + k + p + k * (p + q + 1)

    alphas = T[]; seas = T[]; aonesemis = T[]; tnorms = T[]

    for i in 1:N
        yi = @view Y[:, i]
        Xi = @view X[:, i, :]
        dy = _wdiff(yi)
        ly = _wlag(yi, 1)
        dyc, wysq, aonesemi = _west_unit_pieces(yi, Xi, trend, p, q, lrwindow, k,
                                                nc, ntr, false)
        # final individual regression
        cols, alpha_idx = _west_design(yi, Xi, trend, p, q, k, nc, ntr, true)
        b, se, _, _, _ = _wols(dy, cols)
        alpha = b[alpha_idx]; sea = se[alpha_idx]
        tn = Tobs - p - q - 1 - kp - 1                # non-westerlund tnorm
        push!(alphas, alpha); push!(seas, sea)
        push!(aonesemis, aonesemi); push!(tnorms, T(tn))
    end

    gt = mean(alphas ./ seas)
    ga = mean(tnorms .* alphas ./ aonesemis)

    # Loop 2: pooled Pt, Pa
    ptop = zero(T); pbot = zero(T); sum_sisq = zero(T)
    for i in 1:N
        yi = @view Y[:, i]
        Xi = @view X[:, i, :]
        dy = _wdiff(yi)
        ly = _wlag(yi, 1)
        # wysq (same convention as loop 1)
        dytmp = collect(dy)
        if has_c && has_tr
            fin = findall(isfinite, dytmp)
            mu = mean(dytmp[fin]); dytmp[fin] .-= mu
        end
        wysq = _stata_lrv(filter(isfinite, dytmp), lrwindow)
        # Z_mat = design WITHOUT ly
        zcols, _ = _west_design(yi, Xi, trend, p, q, k, nc, ntr, false)
        _, _, dyresid, mdy, _ = _wols(dy, zcols)
        _, _, yresid, mly, _ = _wols(ly, zcols)
        # full_X = [ly, Z_mat]
        fullcols = vcat([ly], zcols)
        bf, _, residf, _, ssr_f = _wols(dy, fullcols)
        count_sub = 1 + nc + ntr + k + p
        # u = dy - full_X[:,1:count_sub] * cfs[1:count_sub]
        u = fill(T(NaN), Tobs)
        for t in 1:Tobs
            acc = dy[t]
            ok = isfinite(acc)
            for j in 1:count_sub
                v = fullcols[j][t]
                if !isfinite(v); ok = false; break; end
                acc -= bf[j] * v
            end
            u[t] = ok ? acc : T(NaN)
        end
        wusq = _stata_lrv(filter(isfinite, u), lrwindow)
        aonesemipool = sqrt(wusq / wysq)
        vp = [isfinite(dyresid[t]) && isfinite(yresid[t]) for t in 1:Tobs]
        for t in 1:Tobs
            if vp[t]
                ptop += (1 / aonesemipool) * yresid[t] * dyresid[t]
                pbot += yresid[t]^2
            end
        end
        sigmasqi = ssr_f
        tnorm = Tobs - p - q - 1 - kp - 1
        if tnorm > 0
            sum_sisq += (sqrt(sigmasqi / tnorm) / aonesemipool)^2
        end
    end
    pooledalpha = ptop / pbot
    se_pooled = sqrt(sum_sisq / N) / sqrt(pbot)
    pt = pooledalpha / se_pooled
    tnorm_pool = Tobs - p - q - 1 - kp - 1
    pa = tnorm_pool * pooledalpha
    (gt, ga, pt, pa)
end

# aonesemi and wysq for a single unit's ECM (loop-1 semiparametric scaling).
function _west_unit_pieces(yi, Xi, trend, p, q, lrwindow, k, nc, ntr, ::Bool)
    T = eltype(yi)
    Tobs = length(yi)
    has_c = nc == 1; has_tr = ntr == 1
    dy = _wdiff(yi); ly = _wlag(yi, 1)
    cols, alpha_idx = _west_design(yi, Xi, trend, p, q, k, nc, ntr, true)
    b, se, _, _, _ = _wols(dy, cols)
    # u = dy - (det + alpha*ly + lx + L.dy terms); NOT the dx terms.
    # cols order: [det..., ly, lx_1..k, L1..p dy, per-x(leads,dx,lags)]
    # count of "long-run + lagged dy" leading columns = nc+ntr + 1 + k + p
    count_sub = nc + ntr + 1 + k + p
    u = fill(T(NaN), Tobs)
    for t in 1:Tobs
        acc = dy[t]; ok = isfinite(acc)
        for j in 1:count_sub
            v = cols[j][t]
            if !isfinite(v); ok = false; break; end
            acc -= b[j] * v
        end
        u[t] = ok ? acc : T(NaN)
    end
    dytmp = collect(dy)
    if has_c && has_tr
        fin = findall(isfinite, dytmp); mu = mean(dytmp[fin]); dytmp[fin] .-= mu
    end
    wysq = _stata_lrv(filter(isfinite, dytmp), lrwindow)
    wusq = _stata_lrv(filter(isfinite, u), lrwindow)
    aonesemi = sqrt(wusq / wysq)
    (dytmp, wysq, aonesemi)
end

# Build the ECM design column list. `include_ly` toggles the error-correction
# level `y_{t-1}` (present in the individual regression; partialled out in the
# pooled loop). Returns (cols, alpha_idx) where alpha_idx points at `ly` (0 if
# absent). Column order mirrors xtwest.
function _west_design(yi, Xi, trend, p, q, k, nc, ntr, include_ly::Bool)
    T = eltype(yi)
    Tobs = length(yi)
    cols = Vector{Vector{T}}()
    if nc == 1; push!(cols, ones(T, Tobs)); end
    if ntr == 1; push!(cols, T.(1:Tobs)); end
    alpha_idx = 0
    if include_ly
        push!(cols, _wlag(yi, 1)); alpha_idx = length(cols)
    end
    for j in 1:k
        push!(cols, _wlag(@view(Xi[:, j]), 1))         # l.x_j
    end
    dy = _wdiff(yi)
    for kk in 1:p
        push!(cols, _wlag(dy, kk))                      # L(kk) dy
    end
    for j in 1:k
        dxj = _wdiff(@view Xi[:, j])
        for kk in q:-1:1
            push!(cols, _wlead(dxj, kk))                # F(kk) dx_j
        end
        push!(cols, dxj)                                # L0 dx_j
        for kk in 1:p
            push!(cols, _wlag(dxj, kk))                 # L(kk) dx_j
        end
    end
    (cols, alpha_idx)
end

# Seeded CSD-robust bootstrap: resample cross-sectional innovation vectors (whole
# time rows) with replacement to preserve contemporaneous cross-unit dependence,
# regenerate the panel under H0 (independent unit-root y with the observed Δx),
# and recompute the four statistics. p = (#{stat_b ≤ stat_obs}+1)/(B+1).
function _west_bootstrap(Y::Matrix{T}, X::Array{T,3}, trend, p, q, lrwindow,
                         B::Int, seed::Int, obs::Vector{T}) where {T<:AbstractFloat}
    Tobs, N = size(Y)
    k = size(X, 3)
    rng = MersenneTwister(seed)
    # innovations under H0: Δy purged of any deterministic drift (demean per unit)
    dY = Matrix{T}(undef, Tobs - 1, N)
    for i in 1:N
        dyi = diff(@view Y[:, i]); dY[:, i] = dyi .- mean(dyi)
    end
    counts = zeros(Int, 4)
    for b in 1:B
        Yb = Matrix{T}(undef, Tobs, N)
        Yb[1, :] .= zero(T)
        for t in 2:Tobs
            s = rand(rng, 1:(Tobs-1))                   # resample a whole time row
            @inbounds for i in 1:N
                Yb[t, i] = Yb[t-1, i] + dY[s, i]
            end
        end
        gt, ga, pt, pa = try
            _west_raw_stats(Yb, X, trend, p, q, lrwindow)
        catch
            (T(NaN), T(NaN), T(NaN), T(NaN))
        end
        sb = (gt, ga, pt, pa)
        for j in 1:4
            isfinite(sb[j]) && sb[j] <= obs[j] && (counts[j] += 1)
        end
    end
    T[(counts[j] + 1) / (B + 1) for j in 1:4]
end
