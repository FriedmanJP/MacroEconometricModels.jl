# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
HEGY seasonal unit-root tests — quarterly (Hylleberg-Engle-Granger-Yoo 1990)
and monthly (Beaulieu-Miron 1993). EV-29 (#437).
"""

using LinearAlgebra, Statistics

# =============================================================================
# Polynomial machinery
# -----------------------------------------------------------------------------
# The seasonal difference Δ_s = 1 − L^s factors into real/complex-conjugate
# blocks, one per spectral frequency ω_k = 2πk/s (k = 0, …, s/2):
#   • zero frequency ω₀ = 0        →  factor (1 − L)
#   • Nyquist       ω_{s/2} = π    →  factor (1 + L)          (s even)
#   • harmonic pair ω_k            →  factor (1 − 2cos(ω_k)·L + L²)
# The HEGY transform isolating frequency k applies the PRODUCT OF ALL OTHER
# factors to yₜ (so that only the root at ω_k survives). y₁ (zero) keeps the
# HEGY sign (+); y₂ (Nyquist) carries a leading (−) so its t-test is left-tailed
# like an ADF τ; the harmonic pairs are tested with a sign-invariant joint F, so
# their sign is immaterial. This reproduces the quarterly filters of HEGY (1990,
# eq. 3.6): y₁=(1+L)(1+L²), y₂=−(1−L)(1+L²), y₃=(1−L²), and the monthly filters
# of Beaulieu-Miron (1993).
# =============================================================================

# Convolve two coefficient vectors (polynomials in L, index 1 = L^0).
function _poly_conv(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    na, nb = length(a), length(b)
    c = zeros(T, na + nb - 1)
    @inbounds for i in 1:na, j in 1:nb
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

# Apply a lag polynomial `c` (index 1 = L^0) to y: z[t] = Σ_j c[j+1]·y[t-j].
# Rows 1:deg are left as NaN (insufficient history).
function _poly_filter(c::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    m = length(c) - 1
    n = length(y)
    z = fill(T(NaN), n)
    @inbounds for t in (m+1):n
        s = zero(T)
        for j in 0:m
            s += c[j+1] * y[t-j]
        end
        z[t] = s
    end
    return z
end

# The seasonal-frequency factors of Δ_s and the HEGY transform polynomials.
# Returns (freqs, transforms, is_pair) where `transforms[k]` is the lag
# polynomial isolating frequency k, ordered: zero, Nyquist, then harmonic pairs
# in ascending frequency.
function _hegy_transforms(s::Int, ::Type{T}) where {T<:AbstractFloat}
    # Real/complex factors of Δ_s, one per distinct spectral frequency.
    factors = Vector{Vector{T}}()
    freqs = T[]
    push!(factors, T[1, -1]); push!(freqs, zero(T))          # ω = 0     (1 − L)
    push!(factors, T[1, 1]);  push!(freqs, T(pi))            # ω = π     (1 + L)
    for k in 1:(s ÷ 2 - 1)
        ω = T(2 * pi * k / s)
        push!(factors, T[1, -2 * cos(ω), 1])                # 1 − 2cosω·L + L²
        push!(freqs, ω)
    end

    # Transform k = ∏_{j≠k} factor_j, with HEGY signs on the two real roots.
    nf = length(factors)
    transforms = Vector{Vector{T}}(undef, nf)
    is_pair = falses(nf)
    for k in 1:nf
        poly = T[1]
        for j in 1:nf
            j == k && continue
            poly = _poly_conv(poly, factors[j])
        end
        if k == 1
            transforms[k] = poly                 # y₁ (+), zero frequency
        elseif k == 2
            transforms[k] = -poly                # y₂ (−), Nyquist
        else
            transforms[k] = poly                 # harmonic pair (sign-free F)
            is_pair[k] = true
        end
    end
    return freqs, transforms, is_pair
end

# Deterministic design columns for the auxiliary regression rows `t_rows`.
# Seasonal dummies span the constant, so no separate intercept when present.
function _hegy_deterministics(deterministic::Symbol, t_rows::AbstractVector{Int},
                              s::Int, ::Type{T}) where {T<:AbstractFloat}
    n = length(t_rows)
    cols = Vector{Vector{T}}()
    has_seas = deterministic in (:const_seas, :const_trend_seas)
    has_trend = deterministic in (:const_trend, :const_trend_seas)
    if has_seas
        # s seasonal dummies (they contain the constant in their span).
        for q in 1:s
            push!(cols, T[((t - 1) % s) + 1 == q ? one(T) : zero(T) for t in t_rows])
        end
    elseif deterministic != :none
        push!(cols, ones(T, n))                  # intercept
    end
    if has_trend
        push!(cols, T[T(t) for t in t_rows])     # linear trend
    end
    isempty(cols) ? Matrix{T}(undef, n, 0) : reduce(hcat, cols)
end

# Joint Wald/q F-statistic for coefficient indices `idx` in an OLS fit with
# coefficient covariance `covb` (= σ̂²·(X'X)⁻¹).
function _wald_F(beta::AbstractVector{T}, covb::AbstractMatrix{T},
                idx::AbstractVector{Int}) where {T<:AbstractFloat}
    b = beta[idx]
    V = covb[idx, idx]
    q = length(idx)
    return (b' * (V \ b)) / q
end

"""
    hegy_test(y; frequency=4, deterministic=:const_trend_seas, lags=:auto) -> HEGYResult

HEGY seasonal unit-root test — quarterly (`frequency=4`, Hylleberg-Engle-Granger-
Yoo 1990) or monthly (`frequency=12`, Beaulieu-Miron 1993).

Regresses the seasonal difference `Δ_s yₜ` on the HEGY transform regressors
(`y₁,ₜ₋₁`, `y₂,ₜ₋₁`, and two lags of each harmonic-pair transform), the chosen
deterministics, and `p` augmenting lags of `Δ_s y`. Reports:

- `t(π₁)` — zero-frequency unit-root statistic (left-tailed; reject ⇒ no
  zero-frequency root).
- `t(π₂)` — Nyquist (π) statistic (left-tailed).
- a joint `F` for each complex-conjugate harmonic pair (right-tailed; reject ⇒
  no unit root at that seasonal frequency), plus the joint `F` over all seasonal
  frequencies and over all frequencies.

# Arguments
- `y`: seasonal time series (already ordered, no missing seasons).
- `frequency`: `4` (quarterly) or `12` (monthly). Other values throw.
- `deterministic`: `:none`, `:const`, `:const_seas`, `:const_trend`, or
  `:const_trend_seas` (default).
- `lags`: number of augmenting lags of `Δ_s y`, or `:auto` for AIC selection up
  to `⌊12(T/100)^{1/4}⌋`.

Critical values are the published HEGY (1990) / Beaulieu-Miron (1993) tables
(see `critical_values.jl`; Díaz-Emparanza 2014 response surfaces are a SHOULD-
have refinement not implemented here).

# References
- Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990). Seasonal
  integration and cointegration. Journal of Econometrics, 44(1-2), 215-238.
- Beaulieu, J. J., & Miron, J. A. (1993). Seasonal unit roots in aggregate U.S.
  data. Journal of Econometrics, 55(1-2), 305-328.
"""
function hegy_test(y::AbstractVector{T};
                   frequency::Int=4,
                   deterministic::Symbol=:const_trend_seas,
                   lags::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}
    frequency ∈ (4, 12) ||
        throw(ArgumentError("frequency must be 4 (quarterly) or 12 (monthly), got $frequency"))
    deterministic ∈ (:none, :const, :const_seas, :const_trend, :const_trend_seas) ||
        throw(ArgumentError("deterministic must be :none, :const, :const_seas, " *
                            ":const_trend, or :const_trend_seas"))
    s = frequency
    n = length(y)
    n < 3 * s + 5 &&
        throw(ArgumentError("Need at least $(3s + 5) observations for frequency $s, got $n"))

    freqs, transforms, is_pair = _hegy_transforms(s, T)
    # Seasonal difference Δ_s = 1 − L^s (dependent variable).
    ds_coef = zeros(T, s + 1); ds_coef[1] = one(T); ds_coef[end] = -one(T)
    z_dep = _poly_filter(ds_coef, y)                    # valid for t ≥ s+1
    z_tr = [_poly_filter(tp, y) for tp in transforms]   # valid for t ≥ deg+1 ≤ s

    max_p = floor(Int, 12 * (n / 100)^0.25)
    max_p = min(max_p, n - 3 * s - 2)                   # keep the regression feasible
    max_p = max(max_p, 0)

    # Build the (fixed-sample) design at a given number of augmenting lags `p`.
    # `t_start` is common when `common_start` is supplied (for comparable AIC).
    function _fit(p::Int, t_start::Int)
        t_rows = collect(t_start:n)
        nrow = length(t_rows)
        D = _hegy_deterministics(deterministic, t_rows, s, T)
        # π regressors: y₁,ₜ₋₁, y₂,ₜ₋₁, then (y_k,ₜ₋₁, y_k,ₜ₋₂) per pair.
        pi_cols = Vector{Vector{T}}()
        pi_is_pair = Bool[]      # marks the two columns of each harmonic pair
        pair_col_ranges = Vector{UnitRange{Int}}()
        col = 0
        # zero frequency
        push!(pi_cols, T[z_tr[1][t-1] for t in t_rows]); col += 1; push!(pi_is_pair, false)
        zero_idx = col
        # Nyquist
        push!(pi_cols, T[z_tr[2][t-1] for t in t_rows]); col += 1; push!(pi_is_pair, false)
        nyq_idx = col
        # harmonic pairs
        for k in 3:length(transforms)
            push!(pi_cols, T[z_tr[k][t-1] for t in t_rows]); col += 1; push!(pi_is_pair, true)
            push!(pi_cols, T[z_tr[k][t-2] for t in t_rows]); col += 1; push!(pi_is_pair, true)
            push!(pair_col_ranges, (col-1):col)
        end
        Xpi = reduce(hcat, pi_cols)
        # augmenting lags of Δ_s y
        Xaug = p > 0 ? reduce(hcat, [T[z_dep[t-i] for t in t_rows] for i in 1:p]) :
                       Matrix{T}(undef, nrow, 0)
        X = hcat(D, Xpi, Xaug)
        Y = T[z_dep[t] for t in t_rows]
        npi = size(Xpi, 2)
        pi_offset = size(D, 2)                          # π columns start after D
        return (; X, Y, nrow, pi_offset, npi, zero_idx, nyq_idx, pair_col_ranges)
    end

    # AIC lag selection on the common (max_p) sample; then re-fit at the chosen p.
    if lags isa Symbol
        (lags == :auto || lags == :aic || lags == :bic) ||
            throw(ArgumentError("lags must be an Int, :auto, :aic, or :bic"))
        common_start = s + max_p + 1
        best_ic = T(Inf); best_p = 0
        for p in 0:max_p
            f = _fit(p, common_start)
            beta = f.X \ f.Y
            resid = f.Y - f.X * beta
            ssr = sum(resid .^ 2)
            k = size(f.X, 2)
            ic = lags == :bic ? f.nrow * log(ssr / f.nrow) + log(f.nrow) * k :
                                f.nrow * log(ssr / f.nrow) + 2 * k
            if ic < best_ic
                best_ic = ic; best_p = p
            end
        end
        p_final = best_p
    else
        p_final = lags
        (0 <= p_final <= max_p) ||
            throw(ArgumentError("lags=$p_final out of range 0:$max_p for this sample"))
    end

    f = _fit(p_final, s + p_final + 1)
    X, Y = f.X, f.Y
    beta = X \ Y
    resid = Y - X * beta
    k = size(X, 2)
    dof = f.nrow - k
    dof <= 0 && throw(ArgumentError("Insufficient degrees of freedom; reduce lags"))
    sigma2 = sum(resid .^ 2) / dof
    XtX = X'X
    covb = sigma2 * inv(XtX)
    se = sqrt.(max.(diag(covb), zero(T)))

    off = f.pi_offset
    pi_idx = (off + 1):(off + f.npi)
    pi_coefs = beta[pi_idx]

    t_zero = beta[off + f.zero_idx] / se[off + f.zero_idx]
    t_nyquist = beta[off + f.nyq_idx] / se[off + f.nyq_idx]

    pair_freqs = T[]
    pair_F = T[]
    for (kk, rng) in enumerate(f.pair_col_ranges)
        idx = collect((off .+ rng))
        push!(pair_F, _wald_F(beta, covb, idx))
        push!(pair_freqs, freqs[2 + kk])                # pairs start at index 3
    end

    # Joint F over all seasonal frequencies (Nyquist + pairs) and all frequencies.
    seasonal_idx = Int[off + f.nyq_idx]
    for rng in f.pair_col_ranges; append!(seasonal_idx, collect(off .+ rng)); end
    all_idx = vcat(off + f.zero_idx, seasonal_idx)
    F_seasonal = _wald_F(beta, covb, seasonal_idx)
    F_all = _wald_F(beta, covb, all_idx)

    tz_cv, tn_cv, pf_cv = _hegy_critical_values(frequency, deterministic, T)

    return HEGYResult(frequency, deterministic, p_final, pi_coefs,
                      t_zero, t_nyquist, tz_cv, tn_cv,
                      pair_freqs, pair_F, pf_cv, F_seasonal, F_all, f.nrow)
end

hegy_test(y::AbstractVector; kwargs...) = hegy_test(Float64.(y); kwargs...)
