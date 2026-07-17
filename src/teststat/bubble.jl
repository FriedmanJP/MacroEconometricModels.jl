# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# EV-30 (#438): explosive / rational-bubble detection.
#
# Right-tailed sup-ADF family of Phillips, Wu & Yu (2011, SADF) and Phillips,
# Shi & Yu (2015, GSADF + BSADF date-stamping). Unlike the ADF/DF-GLS/PP tests
# elsewhere in this module (which reject a unit root for LARGE NEGATIVE
# statistics), the bubble tests are RIGHT-TAILED: they reject the unit-root null
# in favour of a mildly explosive alternative for LARGE POSITIVE statistics and
# use UPPER null quantiles as critical values. Getting the tail wrong is the
# classic bug, so every rejection here is `stat > cv`.
# =============================================================================

using LinearAlgebra, Statistics, Random

"""
Backward sup-ADF null-simulation cache, keyed by
`(kind, T, swindow0, adflag, cv_method, mc_reps, seed)`. Stores the raw
simulated `(sup_stats::Vector, bsadf_matrix::Matrix)` so that both the sup
critical values and the per-`r2` critical-value sequence can be recovered
without re-running the (expensive) double-sup null loop. Only the analytic
(`:asymptotic`) null is cached — the wild bootstrap depends on the sample's own
residuals and is recomputed every call.
"""
const _BUBBLE_CV_CACHE = Dict{Tuple{Symbol,Int,Int,Int,Symbol,Int,Int},
                              Tuple{Vector{Float64},Matrix{Float64}}}()
const _BUBBLE_CV_CACHE_CAP = 64

# -----------------------------------------------------------------------------
# ADF t-statistic on a level window [i1, i2] (1-indexed into the level series),
# with a constant and `p` augmenting lags. Returns the right-tailed t-statistic
# on the lagged level coefficient. Operates directly on the precomputed `dy`
# (first differences) to avoid reallocating sub-vectors per window; returns
# `-Inf` for degenerate/rank-deficient windows so they never win a sup.
# -----------------------------------------------------------------------------
@inline function _adf_window_tstat(y::AbstractVector{T}, dy::AbstractVector{T},
                                   i1::Int, i2::Int, p::Int) where {T<:AbstractFloat}
    n = (i2 - i1) - p            # number of regression rows
    n < p + 4 && return T(-Inf)  # need enough dof for a meaningful t-stat
    k = p + 2                    # constant + lagged level + p lag terms
    X = Matrix{T}(undef, n, k)
    Y = Vector{T}(undef, n)
    @inbounds for r in 1:n
        base = i1 + p + r - 1
        Y[r] = dy[base]          # Δy_t
        X[r, 1] = one(T)         # constant
        X[r, 2] = y[base]        # y_{t-1}
        for j in 1:p
            X[r, 2 + j] = dy[base - j]   # Δy_{t-j}
        end
    end
    XtX = X'X
    (isfinite(cond(XtX)) && cond(XtX) < 1e12) || return T(-Inf)
    XtXi = robust_inv(XtX)
    B = XtXi * (X'Y)
    resid = Y - X * B
    dof = n - k
    dof < 1 && return T(-Inf)
    sigma2 = sum(abs2, resid) / dof
    v = sigma2 * XtXi[2, 2]
    v > zero(T) || return T(-Inf)
    return B[2] / sqrt(v)        # t-stat on y_{t-1} (right-tailed)
end

# -----------------------------------------------------------------------------
# Backward sup-ADF sequence and the two sup statistics for one series.
#
# For each end index `i2` in `swindow0:T` (so window length ≥ swindow0):
#   BSADF(i2)  = sup_{i1 ∈ 1:(i2-swindow0+1)} ADF(i1, i2)     [double-sup]
#   ADF0(i2)   = ADF(1, i2)                                    [fixed start]
# Returns (bsadf_seq, adf0_seq) both indexed over r2 = swindow0:T.
#   GSADF = maximum(bsadf_seq);  SADF = maximum(adf0_seq).
# -----------------------------------------------------------------------------
function _bsadf_sequences(y::AbstractVector{T}, swindow0::Int, p::Int) where {T<:AbstractFloat}
    Tn = length(y)
    dy = diff(y)
    m = Tn - swindow0 + 1
    bsadf = Vector{T}(undef, m)
    adf0 = Vector{T}(undef, m)
    @inbounds for (pos, i2) in enumerate(swindow0:Tn)
        best = T(-Inf)
        for i1 in 1:(i2 - swindow0 + 1)
            s = _adf_window_tstat(y, dy, i1, i2, p)
            s > best && (best = s)
        end
        bsadf[pos] = best
        adf0[pos] = _adf_window_tstat(y, dy, 1, i2, p)
    end
    return bsadf, adf0
end

# -----------------------------------------------------------------------------
# Null critical-value simulation (analytic PSY driftless random walk) or the
# Phillips–Shi (2020) wild bootstrap. Returns (sup_stats, bsadf_matrix) where
# `bsadf_matrix` is mc_reps × m (m = T - swindow0 + 1). Per-draw seeded via the
# repo's threaded-bootstrap pattern (pre-generate seeds, one MersenneTwister per
# draw so results are thread-order independent).
#
# `kind` selects which sup the returned `sup_stats` records (`:gsadf` → row-max
# of BSADF, `:sadf` → max of the fixed-start ADF0 sequence); the BSADF matrix is
# always returned so the per-r2 CV sequence uses the correct backward-sup null.
# -----------------------------------------------------------------------------
function _simulate_bubble_null(kind::Symbol, Tn::Int, swindow0::Int, p::Int,
                               mc_reps::Int, seed::Int;
                               cv_method::Symbol=:asymptotic,
                               y_sample::Union{Nothing,Vector{Float64}}=nothing)
    m = Tn - swindow0 + 1
    sup_stats = Vector{Float64}(undef, mc_reps)
    bsadf_mat = Matrix{Float64}(undef, mc_reps, m)

    # Wild-bootstrap innovations (Phillips & Shi 2020): resample the sample's own
    # driftless first-difference residuals with N(0,1) multipliers to preserve
    # heteroskedasticity while imposing the unit-root null.
    local wb_resid::Vector{Float64}
    if cv_method == :wildboot
        y_sample === nothing && throw(ArgumentError("wildboot null requires the sample series"))
        dy = diff(y_sample)
        wb_resid = dy .- mean(dy)          # driftless null residuals
    end

    boot_rng = MersenneTwister(seed)
    draw_seeds = rand(boot_rng, UInt64, mc_reps)

    Threads.@threads for b in 1:mc_reps
        rng = MersenneTwister(draw_seeds[b])
        ystar = Vector{Float64}(undef, Tn)
        if cv_method == :wildboot
            ystar[1] = 0.0
            @inbounds for t in 2:Tn
                w = randn(rng)
                ystar[t] = ystar[t-1] + w * wb_resid[rand(rng, eachindex(wb_resid))]
            end
        else
            # PSY driftless random walk: y*_t = y*_{t-1} + ε_t, ε ~ N(0,1/T).
            ystar[1] = 0.0
            s = 1.0 / sqrt(Tn)
            @inbounds for t in 2:Tn
                ystar[t] = ystar[t-1] + s * randn(rng)
            end
        end
        bs, a0 = _bsadf_sequences(ystar, swindow0, p)
        @inbounds bsadf_mat[b, :] .= bs
        sup_stats[b] = kind == :sadf ? maximum(a0) : maximum(bs)
    end
    return sup_stats, bsadf_mat
end

# Fetch (from cache when analytic) the simulated null arrays.
function _bubble_null_arrays(kind::Symbol, Tn::Int, swindow0::Int, p::Int,
                             mc_reps::Int, seed::Int, cv_method::Symbol,
                             y_sample::Vector{Float64})
    if cv_method == :asymptotic
        key = (kind, Tn, swindow0, p, cv_method, mc_reps, seed)
        cached = get(_BUBBLE_CV_CACHE, key, nothing)
        cached !== nothing && return cached
        arrays = _simulate_bubble_null(kind, Tn, swindow0, p, mc_reps, seed;
                                       cv_method=:asymptotic)
        if length(_BUBBLE_CV_CACHE) >= _BUBBLE_CV_CACHE_CAP
            delete!(_BUBBLE_CV_CACHE, first(keys(_BUBBLE_CV_CACHE)))
        end
        _BUBBLE_CV_CACHE[key] = arrays
        return arrays
    else
        # Wild bootstrap depends on the sample residuals — never cached.
        return _simulate_bubble_null(kind, Tn, swindow0, p, mc_reps, seed;
                                     cv_method=:wildboot, y_sample=y_sample)
    end
end

# -----------------------------------------------------------------------------
# Date-stamping (PSY 2015): compare BSADF(r2) against its 95% CV *sequence*.
# Origination = first r2 crossing above CV; termination = first subsequent r2
# falling back below. Enforce the minimum-duration rule (episode spans at least
# ⌈log(T)⌉ observations). Returns (start, end) index pairs into the level series.
# -----------------------------------------------------------------------------
function _date_stamp(bsadf::Vector{T}, cv_seq::Vector{T}, r2_index::Vector{Int},
                     Tn::Int) where {T<:AbstractFloat}
    min_dur = max(1, ceil(Int, log(Tn)))
    episodes = Tuple{Int,Int}[]
    m = length(bsadf)
    k = 1
    while k <= m
        if bsadf[k] > cv_seq[k]
            start_idx = r2_index[k]
            kk = k + 1
            while kk <= m && bsadf[kk] > cv_seq[kk]
                kk += 1
            end
            end_idx = kk <= m ? r2_index[kk] - 1 : r2_index[m]
            if end_idx - start_idx + 1 >= min_dur
                push!(episodes, (start_idx, end_idx))
            end
            k = kk + 1
        else
            k += 1
        end
    end
    return episodes
end

# -----------------------------------------------------------------------------
# Shared core: `kind ∈ (:sadf, :gsadf)`.
# -----------------------------------------------------------------------------
function _bubble_core(y::AbstractVector{T}, kind::Symbol;
                      r0::Union{Symbol,Real}=:auto, adflag::Int=0,
                      mc_reps::Int=999, cv::Symbol=:asymptotic,
                      seed::Int=20240716) where {T<:AbstractFloat}
    adflag >= 0 || throw(ArgumentError("adflag must be ≥ 0"))
    cv ∈ (:asymptotic, :wildboot) ||
        throw(ArgumentError("cv must be :asymptotic or :wildboot"))
    Tn = length(y)
    Tn >= 20 || throw(ArgumentError("Need at least 20 observations, got $Tn"))

    r0f = r0 === :auto ? T(0.01) + T(1.8) / sqrt(T(Tn)) : T(r0)
    (zero(T) < r0f < one(T)) || throw(ArgumentError("r0 must lie in (0,1), got $r0f"))
    swindow0 = max(adflag + 5, floor(Int, r0f * Tn))
    swindow0 <= Tn || throw(ArgumentError("minimum window exceeds sample length"))

    # Sample BSADF / ADF0 sequences and sup statistics.
    bsadf_seq, adf0_seq = _bsadf_sequences(y, swindow0, adflag)
    r2_index = collect(swindow0:Tn)
    stat = kind == :sadf ? maximum(adf0_seq) : maximum(bsadf_seq)
    seq_for_stamp = kind == :sadf ? adf0_seq : bsadf_seq

    # Null simulation → sup CVs, per-r2 CV sequence, p-value.
    y64 = Vector{Float64}(collect(float.(y)))
    sup_stats, bsadf_mat = _bubble_null_arrays(kind, Tn, swindow0, adflag,
                                               mc_reps, seed, cv, y64)
    cvals = Dict{Int,T}(
        10 => T(quantile(sup_stats, 0.90)),
        5  => T(quantile(sup_stats, 0.95)),
        1  => T(quantile(sup_stats, 0.99)),
    )
    m = length(r2_index)
    cv_seq = Vector{T}(undef, m)
    @inbounds for j in 1:m
        cv_seq[j] = T(quantile(view(bsadf_mat, :, j), 0.95))
    end
    pval = T(count(>=(Float64(stat)), sup_stats) / mc_reps)

    episodes = _date_stamp(seq_for_stamp, cv_seq, r2_index, Tn)

    return BubbleResult{T}(kind, stat, pval, cvals, seq_for_stamp, cv_seq,
                           r2_index, episodes, r0f, adflag, cv, mc_reps, Tn)
end

"""
    sadf_test(y; r0=:auto, adflag=0, mc_reps=999, cv=:asymptotic, seed=20240716) -> BubbleResult

Supremum Augmented Dickey-Fuller (SADF) test for explosive / rational-bubble
behaviour (Phillips, Wu & Yu 2011).

Fits a fixed-start, forward-expanding sequence of right-tailed ADF regressions
`ADF₀^{r₂}` (a constant plus `adflag` augmenting lags) over end fractions
`r₂ ∈ [r₀, 1]` and reports `SADF = sup_{r₂} ADF₀^{r₂}`. The unit-root null is
rejected in favour of a mildly explosive root for **large** statistics, using
**upper** simulated critical values.

# Arguments
- `y`: time series (levels — e.g. a price or price-dividend ratio).
- `r0`: minimum window fraction; `:auto` ⇒ `0.01 + 1.8/√T` (the PSY rule).
- `adflag`: number of augmenting lags in each window ADF regression.
- `mc_reps`: null Monte-Carlo replications for the critical values.
- `cv`: `:asymptotic` (PSY driftless random walk) or `:wildboot`
  (Phillips-Shi 2020 wild bootstrap on the sample's ADF residuals).
- `seed`: RNG seed (per-draw seeded; reproducible and thread-order independent).

Returns a [`BubbleResult`](@ref) whose `bsadf` field holds the fixed-start
sequence used for date-stamping.

# References
- Phillips, P. C. B., Wu, Y., & Yu, J. (2011). Explosive behavior in the 1990s
  Nasdaq: When did exuberance escalate asset values? *International Economic
  Review*, 52(1), 201-226.
"""
function sadf_test(y::AbstractVector{T}; r0::Union{Symbol,Real}=:auto,
                   adflag::Int=0, mc_reps::Int=999, cv::Symbol=:asymptotic,
                   seed::Int=20240716) where {T<:AbstractFloat}
    _bubble_core(y, :sadf; r0=r0, adflag=adflag, mc_reps=mc_reps, cv=cv, seed=seed)
end
sadf_test(y::AbstractVector; kwargs...) = sadf_test(float.(collect(y)); kwargs...)

"""
    gsadf_test(y; r0=:auto, adflag=0, mc_reps=999, cv=:asymptotic, seed=20240716) -> BubbleResult

Generalized supremum ADF (GSADF) test with backward sup-ADF (BSADF)
date-stamping (Phillips, Shi & Yu 2015).

Floats **both** window endpoints: for each end fraction `r₂` it takes the
backward sup `BSADF(r₂) = sup_{r₁ ∈ [0, r₂−r₀]} ADF_{r₁}^{r₂}`, and reports
`GSADF = sup_{r₂} BSADF(r₂)`. The full `BSADF(r₂)` sequence is compared against
its 95% critical-value *sequence* to stamp bubble origination/termination dates
(with the PSY `log(T)` minimum-duration rule). Right-tailed, upper CVs.

# Arguments
See [`sadf_test`](@ref) — identical signature. `gsadf_test` is the recommended
real-time exuberance monitor because the double sup detects periodically
collapsing bubbles that the fixed-start SADF misses.

Returns a [`BubbleResult`](@ref) exposing the sup statistic, its CVs and
p-value, the `BSADF` sequence, the per-`r₂` CV sequence, and the stamped
`episodes` (index pairs into `y`).

# References
- Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for multiple bubbles:
  Historical episodes of exuberance and collapse in the S&P 500. *International
  Economic Review*, 56(4), 1043-1078.
- Phillips, P. C. B., & Shi, S. (2020). Real-time monitoring of asset markets:
  Bubbles and crises. In *Handbook of Statistics* 42, 61-80 (wild bootstrap).
"""
function gsadf_test(y::AbstractVector{T}; r0::Union{Symbol,Real}=:auto,
                    adflag::Int=0, mc_reps::Int=999, cv::Symbol=:asymptotic,
                    seed::Int=20240716) where {T<:AbstractFloat}
    _bubble_core(y, :gsadf; r0=r0, adflag=adflag, mc_reps=mc_reps, cv=cv, seed=seed)
end
gsadf_test(y::AbstractVector; kwargs...) = gsadf_test(float.(collect(y)); kwargs...)
