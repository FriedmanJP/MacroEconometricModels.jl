# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Empirical-distribution-function (EDF) goodness-of-fit battery (EV-26, #434).

`edf_test(y; dist, test, params)` runs the Kolmogorov–Smirnov / Lilliefors /
Cramér–von Mises (W²) / Anderson–Darling (A²) / Watson (U²) statistics against a
specified or ML-estimated continuous distribution, mirroring EViews' "Empirical
distribution tests".

For sorted probability-integral transforms `z₍ᵢ₎ = F(y₍ᵢ₎; θ̂)` the statistics are

    KS      D⁺ = maxᵢ (i/n − z₍ᵢ₎),  D⁻ = maxᵢ (z₍ᵢ₎ − (i−1)/n),  D = max(D⁺, D⁻)
    CvM     W² = 1/(12n) + Σᵢ (z₍ᵢ₎ − (2i−1)/(2n))²
    AD      A² = −n − (1/n) Σᵢ (2i−1)[ln z₍ᵢ₎ + ln(1 − z₍ₙ₊₁₋ᵢ₎)]
    Watson  U² = W² − n(z̄ − 0.5)²

**P-value routes** depend on whether θ is specified or estimated:

- `:specified` (Case 0, distribution-free): KS uses the Marsaglia–Tsang–Wang
  (2003) exact CDF for `n ≤ 100` and the asymptotic Kolmogorov series otherwise;
  AD uses the Marsaglia–Marsaglia (2004) asymptotic ADinf CDF; CvM/Watson use the
  standard asymptotic upper-tail tables (Anderson–Darling 1952; Stephens 1970).
- `:estimate` — the statistics are no longer distribution-free. For the NORMAL
  family (Case 3, both μ,σ estimated) we apply the Stephens (1974) modified
  statistics and D'Agostino & Stephens (1986) closed-form p-values for A² and W²,
  and the Dallal–Wilkinson (1986) analytic p-value for the Lilliefors (estimated-
  normal KS) statistic. Estimated-parameter null tables are only hard-coded where
  published; other families return `pvalue = NaN` with a note rather than a wrong
  number.

**Convention (normal, estimated).** The Stephens/Lilliefors null tables are
calibrated on the UNBIASED σ̂ = √(Σ(y−ȳ)²/(n−1)); the normal-estimate route
therefore standardizes with the sample s.d. (`corrected=true`), matching R
`nortest`, `scipy.stats.anderson` and `statsmodels.lilliefors`. Other families use
`Distributions.fit_mle` (with an `Optim.optimize` MLE fallback for Gumbel/Logistic/
Chisq, which lack a `fit_mle` method in the pinned Distributions.jl).

References:
- Anderson, T. W. & Darling, D. A. (1952). "Asymptotic theory of certain
  goodness-of-fit criteria based on stochastic processes." Ann. Math. Statist.
- Anderson, T. W. & Darling, D. A. (1954). "A test of goodness of fit." JASA.
- Lilliefors, H. W. (1967). "On the KS test for normality with mean and variance
  unknown." JASA 62(318): 399–402.
- Stephens, M. A. (1974). "EDF statistics for goodness of fit and some
  comparisons." JASA 69(347): 730–737.
- D'Agostino, R. B. & Stephens, M. A. (1986). Goodness-of-Fit Techniques, ch. 4.
- Dallal, G. E. & Wilkinson, L. (1986). "An analytic approximation to the
  distribution of Lilliefors's test statistic for normality." Amer. Statist.
- Marsaglia, G., Tsang, W. W. & Wang, J. (2003). "Evaluating Kolmogorov's
  distribution." J. Statistical Software 8(18).
- Marsaglia, G. & Marsaglia, J. (2004). "Evaluating the Anderson-Darling
  distribution." J. Statistical Software 9(2).
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Supported families
# =============================================================================

const _EDF_DISTS  = (:normal, :exponential, :logistic, :gumbel, :gamma, :weibull, :chisq)
const _EDF_TESTS  = (:ks, :lilliefors, :cvm, :ad, :watson)

const _EDF_DIST_LABEL = Dict{Symbol,String}(
    :normal => "Normal", :exponential => "Exponential", :logistic => "Logistic",
    :gumbel => "Gumbel", :gamma => "Gamma", :weibull => "Weibull", :chisq => "Chi-squared")
const _EDF_TEST_LABEL = Dict{Symbol,String}(
    :ks => "Kolmogorov–Smirnov (D)", :lilliefors => "Lilliefors (D, est. normal)",
    :cvm => "Cramér–von Mises (W²)", :ad => "Anderson–Darling (A²)",
    :watson => "Watson (U²)")

# =============================================================================
# Critical-value tables
# =============================================================================

# --- Case 0 (fully specified), distribution-free asymptotic upper-tail points ---
# KS: quantiles of √n·D (Kolmogorov limiting law); returned scaled by 1/√n.
const _EDF_KS_C0     = Dict(1 => 1.6276, 5 => 1.3581, 10 => 1.2238)
# CvM W² and AD A² and Watson U² asymptotic upper-tail points (Anderson–Darling
# 1952; Stephens 1970; D'Agostino & Stephens 1986, Table 4.2).
const _EDF_CVM_C0    = Dict(1 => 0.743, 5 => 0.461, 10 => 0.347)
const _EDF_AD_C0     = Dict(1 => 3.857, 5 => 2.492, 10 => 1.933)
const _EDF_WATSON_C0 = Dict(1 => 0.267, 5 => 0.187, 10 => 0.152)

# --- Case 3 (normal, μ & σ estimated): MODIFIED-statistic asymptotic points ---
# Stephens (1974, Table 1) / D'Agostino & Stephens (1986, Table 4.7).
const _EDF_AD_N3     = Dict(1 => 1.092, 5 => 0.787, 10 => 0.656)   # A²*(1+.75/n+2.25/n²)
const _EDF_CVM_N3    = Dict(1 => 0.178, 5 => 0.126, 10 => 0.104)   # W²*(1+0.5/n)
const _EDF_WATSON_N3 = Dict(1 => 0.163, 5 => 0.117, 10 => 0.096)   # U²*(1+0.5/n)
# Lilliefors (1967) estimated-normal KS asymptotic points: coefficient / √n.
const _EDF_LILLIE_N  = Dict(1 => 1.035, 5 => 0.895, 10 => 0.819)

# =============================================================================
# Parameter fitting
# =============================================================================

"""Build a `Distributions` object for `dist` from a parameter tuple/vector `θ`."""
function _edf_build_dist(dist::Symbol, theta)
    t = collect(float.(theta))
    if dist == :normal
        length(t) == 2 || throw(ArgumentError(":normal needs theta=(μ,σ)"))
        return Normal(t[1], t[2])
    elseif dist == :exponential
        length(t) == 1 || throw(ArgumentError(":exponential needs theta=(scale,)"))
        return Exponential(t[1])
    elseif dist == :logistic
        length(t) == 2 || throw(ArgumentError(":logistic needs theta=(μ,s)"))
        return Logistic(t[1], t[2])
    elseif dist == :gumbel
        length(t) == 2 || throw(ArgumentError(":gumbel needs theta=(μ,θ)"))
        return Gumbel(t[1], t[2])
    elseif dist == :gamma
        length(t) == 2 || throw(ArgumentError(":gamma needs theta=(shape,scale)"))
        return Gamma(t[1], t[2])
    elseif dist == :weibull
        length(t) == 2 || throw(ArgumentError(":weibull needs theta=(shape,scale)"))
        return Weibull(t[1], t[2])
    elseif dist == :chisq
        length(t) == 1 || throw(ArgumentError(":chisq needs theta=(dof,)"))
        return Chisq(t[1])
    else
        throw(ArgumentError("unsupported dist :$dist (must be one of $(_EDF_DISTS))"))
    end
end

"""
Generic 1-D MLE via `Optim.optimize` for families without a `fit_mle` method.
`build(θ)` maps an UNCONSTRAINED parameter vector to a distribution; returns the
constrained parameter vector actually used (via `params`).
"""
function _edf_mle_optim(y::AbstractVector{T}, x0::Vector{T}, build) where {T<:AbstractFloat}
    nll(x) = begin
        d = build(x)
        s = zero(T)
        @inbounds for v in y
            s -= T(logpdf(d, v))
        end
        isfinite(s) ? s : T(1e18)
    end
    res = Optim.optimize(nll, x0, Optim.NelderMead(),
                         Optim.Options(g_tol = 1e-10, iterations = 2000))
    xopt = Optim.minimizer(res)
    d = build(xopt)
    return d, T[float(p) for p in params(d)]
end

"""
Fit `dist` to `y` (estimated-parameter route). Returns `(distribution, θ̂::Vector)`.
Normal uses (ȳ, sample s.d.) to match Stephens/Lilliefors table calibration;
positive-support families require `y > 0`.
"""
function _edf_fit(dist::Symbol, y::AbstractVector{T}) where {T<:AbstractFloat}
    if dist == :normal
        mu = mean(y); s = std(y; corrected = true)   # unbiased σ̂ (Case-3 convention)
        return Normal(mu, s), T[mu, s]
    elseif dist == :exponential
        all(>(zero(T)), y) || throw(ArgumentError(":exponential requires y > 0"))
        d = fit_mle(Exponential, y)
        return d, T[float(scale(d))]
    elseif dist == :gamma
        all(>(zero(T)), y) || throw(ArgumentError(":gamma requires y > 0"))
        d = fit_mle(Gamma, y)
        return d, T[float(shape(d)), float(scale(d))]
    elseif dist == :weibull
        all(>(zero(T)), y) || throw(ArgumentError(":weibull requires y > 0"))
        d = fit_mle(Weibull, y)
        return d, T[float(shape(d)), float(scale(d))]
    elseif dist == :logistic
        # Logistic(μ, s): moment start s₀ = σ·√3/π.
        s0 = std(y; corrected = true) * sqrt(T(3)) / T(π)
        x0 = T[median(y), log(max(s0, eps(T)))]
        return _edf_mle_optim(y, x0, x -> Logistic(x[1], exp(x[2])))
    elseif dist == :gumbel
        # Gumbel(μ, θ): θ₀ = σ·√6/π, μ₀ = ȳ − γ·θ₀ (γ = Euler–Mascheroni).
        th0 = std(y; corrected = true) * sqrt(T(6)) / T(π)
        mu0 = mean(y) - T(0.5772156649015329) * th0
        x0 = T[mu0, log(max(th0, eps(T)))]
        return _edf_mle_optim(y, x0, x -> Gumbel(x[1], exp(x[2])))
    elseif dist == :chisq
        all(>(zero(T)), y) || throw(ArgumentError(":chisq requires y > 0"))
        x0 = T[log(max(mean(y), eps(T)))]     # E[χ²_ν] = ν
        return _edf_mle_optim(y, x0, x -> Chisq(exp(x[1])))
    else
        throw(ArgumentError("unsupported dist :$dist (must be one of $(_EDF_DISTS))"))
    end
end

# =============================================================================
# EDF statistics from sorted PIT values
# =============================================================================

"""
Compute `(D, Dplus, Dminus, W2, A2, U2)` from sorted PIT values `z` (ascending).
The AD sum clamps `z` to `[eps(T), 1-eps(T)]` so tied/boundary PITs cannot send
`ln z` or `ln(1−z)` to `∓Inf`.
"""
function _edf_statistics(z::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(z)
    fn = T(n)
    dplus = typemin(T); dminus = typemin(T)
    w2 = one(T) / (T(12) * fn)
    a2acc = zero(T)
    lo = eps(T); hi = one(T) - eps(T)
    @inbounds for i in 1:n
        zi = z[i]
        dp = T(i) / fn - zi
        dm = zi - T(i - 1) / fn
        dp > dplus && (dplus = dp)
        dm > dminus && (dminus = dm)
        # CvM term
        c = zi - (T(2i - 1)) / (T(2) * fn)
        w2 += c * c
        # AD term: (2i-1)[ln z₍ᵢ₎ + ln(1 − z₍ₙ₊₁₋ᵢ₎)]
        zi_c = clamp(zi, lo, hi)
        zj_c = clamp(z[n + 1 - i], lo, hi)
        a2acc += T(2i - 1) * (log(zi_c) + log(one(T) - zj_c))
    end
    a2 = -fn - a2acc / fn
    zbar = sum(z) / fn
    u2 = w2 - fn * (zbar - T(0.5))^2
    d = max(dplus, dminus)
    return (d, dplus, dminus, w2, a2, u2)
end

# =============================================================================
# P-value machinery
# =============================================================================

"""
Marsaglia–Tsang–Wang (2003) EXACT CDF `P(Dₙ < d)` of the one-sample KS statistic
under a fully specified continuous null. Valid and used for `n ≤ 100`.
"""
function _edf_ks_cdf_exact(n::Int, d::T) where {T<:AbstractFloat}
    (d <= 0) && return zero(T)
    (d >= 1) && return one(T)
    nd = n * d
    k = ceil(Int, nd)
    m = 2k - 1
    h = k - nd                          # h ∈ [0,1)
    H = Matrix{Float64}(undef, m, m)
    @inbounds for i in 1:m, j in 1:m
        H[i, j] = (i - j + 1 >= 0) ? 1.0 : 0.0
    end
    @inbounds for i in 1:m
        H[i, 1]  -= h^i
        H[m, i]  -= h^(m - i + 1)
    end
    H[m, 1] += (2h - 1 > 0 ? (2h - 1)^m : 0.0)
    @inbounds for i in 1:m, j in 1:m
        if i - j + 1 > 0
            for g in 1:(i - j + 1)
                H[i, j] /= g
            end
        end
    end
    # Q = Hⁿ with exponent bookkeeping to avoid overflow.
    Q = H^n
    s = Q[k, k]
    eQ = 0
    @inbounds for i in 1:n
        s *= i / n
        if s < 1e-140
            s *= 1e140
            eQ -= 140
        end
    end
    return T(s * 10.0^eQ)
end

"""Asymptotic Kolmogorov survival `Q(λ) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²λ²}`."""
function _edf_kolmogorov_sf(lambda::T) where {T<:AbstractFloat}
    lambda <= 0 && return one(T)
    s = zero(T)
    @inbounds for k in 1:200
        term = T((-1)^(k - 1)) * exp(T(-2) * T(k)^2 * lambda^2)
        s += term
        abs(term) < 1e-14 && break
    end
    return clamp(T(2) * s, zero(T), one(T))
end

"""KS specified-null p-value: exact (MTW) for `n ≤ 100`, else asymptotic with the
Stephens finite-sample correction `λ = (√n + 0.12 + 0.11/√n)·D`."""
function _edf_ks_pvalue(d::T, n::Int) where {T<:AbstractFloat}
    if n <= 100
        return clamp(one(T) - _edf_ks_cdf_exact(n, d), zero(T), one(T))
    else
        lambda = (sqrt(T(n)) + T(0.12) + T(0.11) / sqrt(T(n))) * d
        return _edf_kolmogorov_sf(lambda)
    end
end

"""Marsaglia–Marsaglia (2004) asymptotic Anderson–Darling CDF `P(A²∞ ≤ z)`."""
function _edf_adinf_cdf(z::T) where {T<:AbstractFloat}
    z <= 0 && return zero(T)
    if z < 2
        return exp(-T(1.2337141) / z) / sqrt(z) *
            (T(2.00012) + (T(0.247105) - (T(0.0649821) - (T(0.0347962) -
             (T(0.0116720) - T(0.00168691) * z) * z) * z) * z) * z)
    else
        return exp(-exp(T(1.0776) - (T(2.30695) - (T(0.43424) - (T(0.082433) -
             (T(0.008056) - T(0.0003146) * z) * z) * z) * z) * z))
    end
end

"""AD specified-null asymptotic p-value."""
_edf_ad_pvalue(a2::T) where {T<:AbstractFloat} = clamp(one(T) - _edf_adinf_cdf(a2), zero(T), one(T))

"""
Dallal–Wilkinson (1986) analytic p-value for the Lilliefors (estimated-normal KS)
statistic `D` at sample size `n`. Calibrated for `p < 0.10`; extrapolates smoothly
above that (the value is then approximate — reported as such by the case label).
"""
function _edf_lilliefors_pvalue(d::T, n::Int) where {T<:AbstractFloat}
    dd = d; nn = n
    if nn > 100
        dd *= T((nn / 100)^0.49)
        nn = 100
    end
    fnn = T(nn)
    p = exp(T(-7.01256) * dd^2 * (fnn + T(2.78019)) +
            T(2.99587) * dd * sqrt(fnn + T(2.78019)) - T(0.122119) +
            T(0.974598) / sqrt(fnn) + T(1.67997) / fnn)
    return clamp(p, zero(T), one(T))
end

"""D'Agostino & Stephens (1986) closed-form p-value for the modified A²* (normal,
both parameters estimated)."""
function _edf_ad_pvalue_normal_est(a2s::T) where {T<:AbstractFloat}
    p = if a2s < T(0.2)
        one(T) - exp(T(-13.436) + T(101.14) * a2s - T(223.73) * a2s^2)
    elseif a2s < T(0.34)
        one(T) - exp(T(-8.318) + T(42.796) * a2s - T(59.938) * a2s^2)
    elseif a2s < T(0.6)
        exp(T(0.9177) - T(4.279) * a2s - T(1.38) * a2s^2)
    else
        exp(T(1.2937) - T(5.709) * a2s + T(0.0186) * a2s^2)
    end
    return clamp(p, zero(T), one(T))
end

"""D'Agostino & Stephens (1986) closed-form p-value for the modified W²* (normal,
both parameters estimated)."""
function _edf_cvm_pvalue_normal_est(w2s::T) where {T<:AbstractFloat}
    p = if w2s < T(0.0275)
        one(T) - exp(T(-13.953) + T(775.5) * w2s - T(12542.61) * w2s^2)
    elseif w2s < T(0.051)
        one(T) - exp(T(-5.903) + T(179.546) * w2s - T(1515.29) * w2s^2)
    elseif w2s < T(0.092)
        exp(T(0.886) - T(31.62) * w2s + T(10.897) * w2s^2)
    else
        exp(T(1.111) - T(34.242) * w2s + T(12.832) * w2s^2)
    end
    return clamp(p, zero(T), one(T))
end

"""Piecewise-linear upper-tail p-value from a `{1,5,10} => cv` table (house
convention, cf. `kpss_pvalue`); large statistic ⇒ small p."""
function _edf_pval_interp(stat::T, cv::Dict) where {T<:AbstractFloat}
    c1 = T(cv[1]); c5 = T(cv[5]); c10 = T(cv[10])
    if stat >= c1
        return T(0.01)
    elseif stat >= c5
        return T(0.01) + T(0.04) * (c1 - stat) / (c1 - c5)
    elseif stat >= c10
        return T(0.05) + T(0.05) * (c5 - stat) / (c5 - c10)
    else
        return clamp(T(0.10) + T(0.40) * (c10 - stat) / c10, zero(T), one(T))
    end
end

# =============================================================================
# Resolve statistic + p-value + critical values + case label
# =============================================================================

"""Scale a `coefficient/√n` KS-type table into level critical values."""
_edf_scale_cv(tbl::Dict, n::Int, ::Type{T}) where {T} =
    Dict{Int,T}(k => T(v) / sqrt(T(n)) for (k, v) in tbl)
_edf_copy_cv(tbl::Dict, ::Type{T}) where {T} = Dict{Int,T}(k => T(v) for (k, v) in tbl)

function _edf_resolve(test::Symbol, dist::Symbol, params::Symbol,
                      stats::NTuple{6,T}, n::Int) where {T<:AbstractFloat}
    d, _, _, w2, a2, u2 = stats
    empty_cv = Dict{Int,T}()

    if params == :specified
        # Case 0 — distribution-free asymptotics.
        if test == :ks
            return (d, d, _edf_ks_pvalue(d, n), _edf_scale_cv(_EDF_KS_C0, n, T),
                    "Case 0 (fully specified) — KS exact/asymptotic")
        elseif test == :cvm
            cv = _edf_copy_cv(_EDF_CVM_C0, T)
            return (w2, w2, _edf_pval_interp(w2, cv), cv,
                    "Case 0 (fully specified) — CvM asymptotic table")
        elseif test == :ad
            return (a2, a2, _edf_ad_pvalue(a2), _edf_copy_cv(_EDF_AD_C0, T),
                    "Case 0 (fully specified) — AD asymptotic (ADinf)")
        elseif test == :watson
            cv = _edf_copy_cv(_EDF_WATSON_C0, T)
            return (u2, u2, _edf_pval_interp(u2, cv), cv,
                    "Case 0 (fully specified) — Watson asymptotic table")
        else # :lilliefors is inherently an estimated-parameter test
            throw(ArgumentError("test=:lilliefors is only defined for params=:estimate"))
        end
    end

    # params == :estimate ---------------------------------------------------
    if dist == :normal
        if test == :ks || test == :lilliefors
            cv = _edf_scale_cv(_EDF_LILLIE_N, n, T)
            return (d, d, _edf_lilliefors_pvalue(d, n), cv,
                    "Case 3 (μ,σ estimated) — Lilliefors / Dallal–Wilkinson")
        elseif test == :ad
            a2s = a2 * (one(T) + T(0.75) / n + T(2.25) / n^2)
            return (a2s, a2, _edf_ad_pvalue_normal_est(a2s), _edf_copy_cv(_EDF_AD_N3, T),
                    "Case 3 (μ,σ estimated) — Stephens A²* / D'Agostino–Stephens")
        elseif test == :cvm
            w2s = w2 * (one(T) + T(0.5) / n)
            return (w2s, w2, _edf_cvm_pvalue_normal_est(w2s), _edf_copy_cv(_EDF_CVM_N3, T),
                    "Case 3 (μ,σ estimated) — Stephens W²* / D'Agostino–Stephens")
        elseif test == :watson
            u2s = u2 * (one(T) + T(0.5) / n)
            cv = _edf_copy_cv(_EDF_WATSON_N3, T)
            return (u2s, u2, _edf_pval_interp(u2s, cv), cv,
                    "Case 3 (μ,σ estimated) — Stephens U²* (table)")
        end
    end

    # Estimated parameters, non-normal family: no published null table.
    raw = test == :ks || test == :lilliefors ? d :
          test == :cvm ? w2 : test == :ad ? a2 : u2
    return (raw, raw, T(NaN), empty_cv,
            "$(_EDF_DIST_LABEL[dist]) parameters estimated — no published null " *
            "table; p-value not available")
end

# =============================================================================
# Public API
# =============================================================================

"""
    edf_test(y; dist=:normal, test=:ad, params=:estimate, theta=nothing) -> EDFTestResult

Empirical-distribution-function goodness-of-fit test.

Tests `H₀`: `y` is drawn from `dist` (with parameters either estimated from the
data or fully specified) against `H₁`: `y` is not.

# Arguments
- `y`: data vector (`AbstractVector`, converted to floats internally).

# Keyword arguments
- `dist`: null family — one of `:normal`, `:exponential`, `:logistic`, `:gumbel`,
  `:gamma`, `:weibull`, `:chisq`.
- `test`: EDF statistic — `:ks`, `:lilliefors` (estimated-normal KS), `:cvm`,
  `:ad` (default), `:watson`.
- `params`: `:estimate` (default, ML-fit θ) or `:specified` (supply `theta`).
- `theta`: parameter tuple/vector when `params=:specified`
  (e.g. `theta=(0.0, 1.0)` for `dist=:normal`).

# Returns
`EDFTestResult{T}` — statistic, p-value, critical values and case label; renders
in the ADF style via `report`/`show`.

# Notes
- Estimated-parameter p-values are only tabulated for the NORMAL family (Case 3,
  Stephens 1974 / Dallal–Wilkinson 1986). Other families return `pvalue = NaN`
  under `params=:estimate` and report the reason in the case label.
- Constant / zero-variance input raises an `ArgumentError`.

# Example
```julia
using Random
y = randn(MersenneTwister(1), 200)
edf_test(y; dist=:normal, test=:ad, params=:estimate)
edf_test(y; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0))
```

# References
- Stephens, M. A. (1974). *EDF statistics for goodness of fit*. JASA 69: 730–737.
- Anderson & Darling (1954); Lilliefors (1967); Dallal & Wilkinson (1986);
  Marsaglia, Tsang & Wang (2003).
"""
function edf_test(y::AbstractVector{T};
                  dist::Symbol = :normal,
                  test::Symbol = :ad,
                  params::Symbol = :estimate,
                  theta = nothing) where {T<:AbstractFloat}
    dist ∈ _EDF_DISTS || throw(ArgumentError("dist must be one of $(_EDF_DISTS); got :$dist"))
    test ∈ _EDF_TESTS || throw(ArgumentError("test must be one of $(_EDF_TESTS); got :$test"))
    params ∈ (:estimate, :specified) ||
        throw(ArgumentError("params must be :estimate or :specified; got :$params"))

    n = length(y)
    n >= 5 || throw(ArgumentError("EDF test needs at least 5 observations (got n=$n)"))
    (maximum(y) - minimum(y) <= eps(T) * max(one(T), abs(mean(y)))) &&
        throw(ArgumentError("input is (near-)constant / zero variance — EDF test undefined"))

    # :lilliefors is an estimated-normal KS test by construction.
    if test == :lilliefors
        dist == :normal ||
            throw(ArgumentError("test=:lilliefors is only defined for dist=:normal"))
        params == :estimate ||
            throw(ArgumentError("test=:lilliefors requires params=:estimate"))
    end

    # Distribution + parameters.
    local D0, theta_vec
    if params == :specified
        theta === nothing &&
            throw(ArgumentError("params=:specified requires a `theta` tuple (e.g. theta=(0.0,1.0))"))
        D0 = _edf_build_dist(dist, theta)
        theta_vec = T[float(p) for p in params_of(D0)]
    else
        D0, theta_vec = _edf_fit(dist, y)
    end

    # PIT and sort.
    z = sort!(T[T(cdf(D0, v)) for v in y])
    stats = _edf_statistics(z)
    statistic, raw_statistic, pval, cv, case = _edf_resolve(test, dist, params, stats, n)

    return EDFTestResult{T}(test, dist, params, statistic, raw_statistic, T(pval),
                            n, theta_vec, cv, case)
end

# Float fallback: accept integer/other real vectors.
edf_test(y::AbstractVector; kwargs...) = edf_test(float.(collect(y)); kwargs...)

# Small shim so the specified branch can read parameters uniformly.
params_of(d::Distribution) = params(d)
