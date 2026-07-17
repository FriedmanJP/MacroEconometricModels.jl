# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-27 (#435): Variance-ratio / random-walk (martingale) tests.
#
# Lo & MacKinlay (1988) overlapping VR(q) with the unbiased normalizer
# m = q(N-q+1)(1-q/N); homoskedastic Z(q) and heteroskedasticity-robust Z*(q);
# Chow & Denning (1993) joint max|Z| statistic with an SMM-complement p-value;
# Wright (2000) rank (R1/R2) and sign (S1) statistics with simulated iid nulls;
# Kim (2006) wild-bootstrap p-values (Rademacher / normal weights).

"""
Variance-ratio (random-walk / martingale) tests.

Under the random-walk hypothesis the variance of `q`-period increments grows
linearly in `q`, so the variance ratio `VR(q) = Var(y_t − y_{t−q}) / (q·Var(y_t − y_{t−1}))`
equals one for every `q`. Deviations diagnose serial dependence in the increments:
`VR(q) > 1` ⇒ positive autocorrelation (trending), `VR(q) < 1` ⇒ mean reversion.

`y` is the **level** (log-price) series; the tests operate on its first differences
`x_t = y_t − y_{t−1}` (returns). With `N+1` level observations there are `N` returns.

References:
- Lo, A. W. and MacKinlay, A. C. (1988). "Stock Market Prices Do Not Follow Random
  Walks: Evidence from a Simple Specification Test." Review of Financial Studies 1(1): 41-66.
- Chow, K. V. and Denning, K. C. (1993). "A Simple Multiple Variance Ratio Test."
  Journal of Econometrics 58(3): 385-401.
- Wright, J. H. (2000). "Alternative Variance-Ratio Tests Using Ranks and Signs."
  Journal of Business & Economic Statistics 18(1): 1-9.
- Kim, J. H. (2006). "Wild Bootstrapping Variance Ratio Tests." Economics Letters 92(1): 38-43.
"""

# =============================================================================
# Core Lo–MacKinlay VR(q) from a return series
# =============================================================================

"""
Lo–MacKinlay statistics for one aggregation value `q`, computed from the return
vector `x` (length `N`). Returns `(VR, Z, Zstar)`:

- `σ̂²_a = (1/(N−1)) Σ_{t} (x_t − μ̂)²`               (1-period variance)
- overlapping `σ̂²_c = (1/m) Σ_{t=q}^{N} (Σ_{i=0}^{q−1} x_{t−i} − qμ̂)²` with the
  **unbiased** normalizer `m = q(N−q+1)(1 − q/N)` (using `Nq` instead biases VR down),
- `VR(q) = σ̂²_c / σ̂²_a`,
- homoskedastic `Z(q) = √N (VR−1) / √(2(2q−1)(q−1)/(3q))`,
- robust `Z*(q) = √N (VR−1) / √θ̂(q)` with
  `θ̂(q) = Σ_{j=1}^{q−1} (2(q−j)/q)² δ̂_j`,
  `δ̂_j = [Σ_{t=j+1}^{N} (x_t−μ̂)²(x_{t−j}−μ̂)²] / [Σ_t (x_t−μ̂)²]²`.
"""
function _vr_lomac(x::AbstractVector{T}, q::Int) where {T<:AbstractFloat}
    N = length(x)
    μ = sum(x) / N
    e = x .- μ
    ss = dot(e, e)                      # Σ (x_t − μ)²
    σ2a = ss / (N - 1)
    # Overlapping q-increment sum via cumulative sums: y_t − y_{t−q} = Σ_{i} x_{t−i+1}.
    c = cumsum(x)                       # c[t] = Σ_{i=1}^{t} x_i
    m = q * (N - q + 1) * (1 - q / N)
    s = zero(T)
    @inbounds for t in q:N
        prev = t - q == 0 ? zero(T) : c[t-q]
        w = (c[t] - prev) - q * μ
        s += w * w
    end
    σ2c = s / m
    vr = σ2c / σ2a
    # Homoskedastic standardization (Lo–MacKinlay 1988, eq. for the M1 statistic).
    homo_var = T(2 * (2q - 1) * (q - 1)) / T(3q)     # 2(2q−1)(q−1)/(3q)
    z = sqrt(T(N)) * (vr - 1) / sqrt(homo_var)
    # Heteroskedasticity-robust θ̂(q). δ̂_j is the heteroskedasticity-consistent
    # asymptotic variance of √N·ρ̂(j):
    #   δ̂_j = N · [Σ_{t=j+1}^{N} (x_t−μ)²(x_{t−j}−μ)²] / [Σ_t (x_t−μ)²]²
    # The numerator carries the factor N (= nq) and the denominator is the SAME
    # ss² for every j (both are the classic scaling traps); under iid homoskedastic
    # returns δ̂_j → 1, so θ̂(q) → 2(2q−1)(q−1)/(3q) and Z*(q) → Z(q).
    θ = zero(T)
    @inbounds for j in 1:(q-1)
        num = zero(T)
        for t in (j+1):N
            num += e[t]^2 * e[t-j]^2
        end
        δj = T(N) * num / (ss * ss)
        wj = T(2 * (q - j)) / T(q)                    # 2(q−j)/q
        θ += wj * wj * δj
    end
    zstar = sqrt(T(N)) * (vr - 1) / sqrt(θ)
    return vr, z, zstar
end

# =============================================================================
# Wright (2000) rank / sign statistics and their simulated iid nulls
# =============================================================================

"""Average (tied) ranks of `x`, 1-based; ties share their mean rank."""
function _tiedrank(x::AbstractVector)
    n = length(x)
    p = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    @inbounds while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for kk in i:j
            r[p[kk]] = avg
        end
        i = j + 1
    end
    return r
end

"""
Wright's standardized variance-ratio statistic on a (mean-zero) score vector `z`
(length `N`) at aggregation `k`. Uses the **simple** normalizer `Nk` (Wright 2000,
not the Lo–MacKinlay unbiased `m`):

`stat = [ (Nk)⁻¹ Σ_{t=k}^{N} (Σ_{i=0}^{k−1} z_{t−i})² / (N⁻¹ Σ z_t²) − 1 ] / √(2(2k−1)(k−1)/(3kN))`.
"""
function _wright_stat(z::AbstractVector{T}, k::Int, N::Int) where {T<:AbstractFloat}
    c = cumsum(z)
    num = zero(T)
    @inbounds for t in k:N
        prev = t - k == 0 ? zero(T) : c[t-k]
        w = c[t] - prev
        num += w * w
    end
    σ2c = num / (N * k)
    σ2a = dot(z, z) / N
    vr = σ2c / σ2a
    return (vr - 1) / sqrt(T(2 * (2k - 1) * (k - 1)) / T(3k * N))
end

# Module-level cache of the simulated iid nulls, keyed by (N, q, kind). Wright's
# rank/sign nulls depend only on the sample size, the aggregation, and the score
# type — never on the data — so they are computed once and reused. The cap bounds
# memory when a batch sweeps many odd sample sizes.
const _WRIGHT_NULL_CACHE = Dict{Tuple{Int,Int,Symbol},Vector{Float64}}()
const _WRIGHT_CACHE_CAP = 128
const _WRIGHT_NDRAWS = 10_000
const _WRIGHT_BASE_SEED = 0x57524947   # "WRIG"

"""
Sorted vector of `_WRIGHT_NDRAWS` draws of the Wright statistic under the iid null
for `(N, q, kind)`, `kind ∈ (:r1, :r2, :s1)`. `:r1` shuffles the fixed standardized
ranks, `:r2` shuffles the van-der-Waerden normal scores, `:s1` draws iid ±1 signs.
Deterministically seeded from `(N, q, kind)`; cached (size-capped) by `(N, q, kind)`.
"""
function _wright_null(N::Int, q::Int, kind::Symbol)
    key = (N, q, kind)
    haskey(_WRIGHT_NULL_CACHE, key) && return _WRIGHT_NULL_CACHE[key]
    length(_WRIGHT_NULL_CACHE) >= _WRIGHT_CACHE_CAP && empty!(_WRIGHT_NULL_CACHE)
    seed = (UInt64(_WRIGHT_BASE_SEED) ⊻ (UInt64(N) * 0x9E3779B1) ⊻
            (UInt64(q) << 17) ⊻ (UInt64(hash(kind)))) % typemax(UInt32)
    rng = Random.MersenneTwister(Int(seed))
    base = if kind === :r1
        Float64[(i - (N + 1) / 2) / sqrt((N - 1) * (N + 1) / 12) for i in 1:N]
    elseif kind === :r2
        Float64[quantile(Normal(), i / (N + 1)) for i in 1:N]
    else
        Float64[]                       # :s1 draws signs directly
    end
    draws = Vector{Float64}(undef, _WRIGHT_NDRAWS)
    z = Vector{Float64}(undef, N)
    @inbounds for b in 1:_WRIGHT_NDRAWS
        if kind === :s1
            for t in 1:N
                z[t] = rand(rng, Bool) ? 1.0 : -1.0
            end
        else
            Random.shuffle!(rng, copyto!(z, base))
        end
        draws[b] = _wright_stat(z, q, N)
    end
    sort!(draws)
    _WRIGHT_NULL_CACHE[key] = draws
    return draws
end

"""Two-sided simulated p-value: fraction of |null draws| ≥ |obs| (with a +1 guard)."""
function _wright_pvalue(draws::Vector{Float64}, obs::Real)
    a = abs(obs)
    cnt = count(d -> abs(d) >= a, draws)
    return (cnt + 1) / (length(draws) + 1)
end

# =============================================================================
# Kim (2006) wild bootstrap
# =============================================================================

"""
Kim (2006) wild-bootstrap p-values for the robust `Z*(q)` and the Chow–Denning
`max_q |Z*(q)|` statistic. Each replication draws external weights `η_t`
(Rademacher `±1` by default, or standard normal) and forms `x*_t = η_t (x_t − μ̂)`,
a martingale-difference resample under the null, then recomputes `Z*(q)` and the
joint max. Returns `(per_q_pvalues, cd_pvalue)`; each is the two-sided fraction of
bootstrap statistics at least as extreme as the observed one (+1 guard).
"""
function _kim_bootstrap(x::AbstractVector{T}, qvec::Vector{Int}, B::Int,
                        weights::Symbol, seed::Int) where {T<:AbstractFloat}
    N = length(x)
    μ = sum(x) / N
    e = x .- μ
    zstar_obs = T[_vr_lomac(x, q)[3] for q in qvec]
    cd_obs = maximum(abs, zstar_obs)
    nq = length(qvec)
    cnt_z = zeros(Int, nq)
    cnt_cd = 0
    rng = Random.MersenneTwister(seed)
    xstar = Vector{T}(undef, N)
    for _ in 1:B
        @inbounds for t in 1:N
            η = weights === :normal ? T(randn(rng)) : (rand(rng, Bool) ? one(T) : -one(T))
            xstar[t] = η * e[t]
        end
        cd_b = zero(T)
        for i in 1:nq
            zb = _vr_lomac(xstar, qvec[i])[3]
            azb = abs(zb)
            azb >= abs(zstar_obs[i]) && (cnt_z[i] += 1)
            azb > cd_b && (cd_b = azb)
        end
        cd_b >= cd_obs && (cnt_cd += 1)
    end
    zstar_boot_p = T[(cnt_z[i] + 1) / (B + 1) for i in 1:nq]
    cd_boot_p = T((cnt_cd + 1) / (B + 1))
    return zstar_boot_p, cd_boot_p
end

# =============================================================================
# variance_ratio_test
# =============================================================================

"""
    variance_ratio_test(y::AbstractVector; q=[2,4,8,16], method=:lomackinlay,
                        bootstrap::Int=0, robust::Bool=true,
                        boot_weights::Symbol=:rademacher, seed::Int=1234)
        -> VarianceRatioResult

Variance-ratio test of the random-walk (martingale) hypothesis for the **level**
series `y` (log-price convention). Works with first differences `x_t = y_t − y_{t−1}`.

- `H₀`: `y` is a random walk — `VR(q) = 1` for every `q`.
- `H₁`: increments are serially dependent — `VR(q) ≠ 1` for some `q`.

For each `q` the overlapping Lo–MacKinlay `VR(q)` is computed with the unbiased
normalizer `m = q(N−q+1)(1−q/N)` (`N` = number of returns), together with the
homoskedastic `Z(q)` and the heteroskedasticity-robust `Z*(q)` (both asymptotically
`N(0,1)`). The Chow–Denning joint statistic `CD = max_q |Z(q)|` (and its robust
counterpart `max_q |Z*(q)|`) tests all `q` simultaneously; its p-value comes from
the studentized-maximum-modulus complement `1 − (2Φ(CD) − 1)^m`, `m = length(q)`.

# Keyword Arguments
- `q`: aggregation values (each `2 ≤ q < N`). Default `[2,4,8,16]`.
- `method`: `:lomackinlay` (default) or `:wright` (additionally reports Wright's
  rank `R1`/`R2` and sign `S1` statistics with simulated iid-null p-values).
- `bootstrap`: number of Kim (2006) wild-bootstrap replications for `Z*(q)` and the
  Chow–Denning statistic (`0` = asymptotic only).
- `robust`: if `true` (default) the robust `Z*`/Chow–Denning branch is the reported
  primary; `false` selects the homoskedastic branch.
- `boot_weights`: `:rademacher` (default) or `:normal` wild-bootstrap weights.
- `seed`: RNG seed for the wild bootstrap.

# Example
```julia
y = cumsum(randn(500))                 # a random walk
r = variance_ratio_test(y; q=[2,4,8,16])
r.vr, r.cd_star_stat, r.cd_star_pvalue
```

# References
- Lo & MacKinlay (1988), Review of Financial Studies 1(1).
- Chow & Denning (1993), Journal of Econometrics 58(3).
- Wright (2000), JBES 18(1); Kim (2006), Economics Letters 92(1).
"""
function variance_ratio_test(y::AbstractVector;
                             q=[2, 4, 8, 16], method::Symbol=:lomackinlay,
                             bootstrap::Int=0, robust::Bool=true,
                             boot_weights::Symbol=:rademacher, seed::Int=1234)
    method in (:lomackinlay, :wright) ||
        throw(ArgumentError("method must be :lomackinlay or :wright, got :$method"))
    boot_weights in (:rademacher, :normal) ||
        throw(ArgumentError("boot_weights must be :rademacher or :normal, got :$boot_weights"))
    bootstrap >= 0 || throw(ArgumentError("bootstrap must be ≥ 0, got $bootstrap"))
    yv = float.(collect(y))
    T = eltype(yv)
    nlev = length(yv)
    nlev >= 4 || throw(ArgumentError("need at least 4 level observations, got $nlev"))
    qvec = sort(unique(Int.(collect(q))))
    all(qi -> qi >= 2, qvec) || throw(ArgumentError("every q must be ≥ 2"))
    N = nlev - 1                                    # number of returns
    maximum(qvec) < N ||
        throw(ArgumentError("every q must be < number of returns ($N); got q=$qvec"))

    x = diff(yv)                                    # returns, length N
    nq = length(qvec)

    vr = Vector{T}(undef, nq)
    z = Vector{T}(undef, nq)
    zstar = Vector{T}(undef, nq)
    for (i, qi) in enumerate(qvec)
        vr[i], z[i], zstar[i] = _vr_lomac(x, qi)
    end
    # Per-q asymptotic two-sided p-values.
    z_pvalue = T[2 * ccdf(Normal(), abs(zi)) for zi in z]
    zstar_pvalue = T[2 * ccdf(Normal(), abs(zi)) for zi in zstar]

    # Chow–Denning joint statistics + SMM-complement p-values (m = nq, ∞ df,
    # asymptotically independent standard normals): p = 1 − (2Φ(CD) − 1)^m.
    cd = maximum(abs, z)
    cd_star = maximum(abs, zstar)
    _smm_p(stat) = T(1 - (2 * cdf(Normal(), stat) - 1)^nq)
    cd_pvalue = _smm_p(cd)
    cd_star_pvalue = _smm_p(cd_star)

    # Wright (2000) rank / sign statistics (optional).
    do_wright = method === :wright
    R1 = T[]; R2 = T[]; S1 = T[]
    R1p = T[]; R2p = T[]; S1p = T[]
    if do_wright
        ranks = _tiedrank(x)
        r1 = T[(ranks[t] - (N + 1) / 2) / sqrt((N - 1) * (N + 1) / 12) for t in 1:N]
        r2 = T[quantile(Normal(), ranks[t] / (N + 1)) for t in 1:N]
        μx = sum(x) / N
        s = T[sign(x[t] - μx) for t in 1:N]
        R1 = Vector{T}(undef, nq); R2 = Vector{T}(undef, nq); S1 = Vector{T}(undef, nq)
        R1p = Vector{T}(undef, nq); R2p = Vector{T}(undef, nq); S1p = Vector{T}(undef, nq)
        for (i, qi) in enumerate(qvec)
            R1[i] = _wright_stat(r1, qi, N)
            R2[i] = _wright_stat(r2, qi, N)
            S1[i] = _wright_stat(s, qi, N)
            R1p[i] = T(_wright_pvalue(_wright_null(N, qi, :r1), R1[i]))
            R2p[i] = T(_wright_pvalue(_wright_null(N, qi, :r2), R2[i]))
            S1p[i] = T(_wright_pvalue(_wright_null(N, qi, :s1), S1[i]))
        end
    end

    # Kim (2006) wild bootstrap (optional).
    zstar_boot_p = T[]
    cd_boot_p = T(NaN)
    if bootstrap > 0
        zstar_boot_p, cd_boot_p = _kim_bootstrap(x, qvec, bootstrap, boot_weights, seed)
    end

    return VarianceRatioResult{T}(qvec, vr, z, zstar, z_pvalue, zstar_pvalue,
                                  cd, cd_pvalue, cd_star, cd_star_pvalue,
                                  method, robust, do_wright,
                                  R1, R2, S1, R1p, R2p, S1p,
                                  bootstrap, boot_weights, seed,
                                  zstar_boot_p, cd_boot_p, nlev)
end
