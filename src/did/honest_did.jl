# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Rambachan & Roth (2023) honest DiD sensitivity analysis.

Constructs confidence sets for post-treatment causal effects that remain valid
under bounded violations of parallel trends. The event-study coefficients are
modeled as `betahat ~ N(tau + delta, Sigma)` with `tau_pre = 0`, and the trend
violation `delta` is restricted to one of two sets:

- `Delta^RM(Mbar)` (relative magnitudes): post-treatment first differences of
  `delta` are no larger than `Mbar` times the largest observed pre-treatment
  first difference. `Mbar` is dimensionless.
- `Delta^SD(M)` (second differences / smoothness): consecutive slope changes of
  `delta` are bounded by `M`, in outcome units per period squared. Inference
  uses the Armstrong-Kolesár (2018) fixed-length confidence interval (FLCI)
  built from an affine estimator whose worst-case bias over `Delta^SD(M)` is
  traded off against its variance.

# References
- Rambachan, A. & Roth, J. (2023). A More Credible Approach to Parallel Trends.
  *Review of Economic Studies* 90(5), 2555-2591.
- Armstrong, T. & Kolesár, M. (2018). Optimal Inference in a Class of Regression
  Models. *Econometrica* 86(2), 655-683.
"""

using Distributions, LinearAlgebra
import NLopt

# =============================================================================
# Small helpers
# =============================================================================

"""Basis vector e_i of length n."""
function _basis_vector(i::Int, n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    v = zeros(T, n)
    v[i] = one(T)
    v
end

"""
    _fold_normal_cv(b, alpha) -> cv

(1-alpha) quantile of |N(b,1)| — the Armstrong-Kolesár FLCI critical value for
worst-case-bias-to-sd ratio `b`. Equals `quantile(Normal(), 1-alpha/2)` at b=0.
"""
function _fold_normal_cv(b::T, alpha::T) where {T<:AbstractFloat}
    babs = abs(b)
    if babs < T(1e-10)
        return T(quantile(Normal(), 1 - alpha / 2))
    elseif babs > T(20)
        # P(Z + b < 0) is negligible: quantile of |Z+b| = b + z_{1-alpha} to
        # machine precision (also avoids Rmath pnchisq non-convergence)
        return babs + T(quantile(Normal(), 1 - alpha))
    end
    T(sqrt(quantile(NoncentralChisq(1, Float64(babs)^2), 1 - Float64(alpha))))
end

# w -> l_pre map: l_pre = WtoL * w with WtoL unit diagonal and -1 subdiagonal
# (w_k = cumulative sum of pre-period weights l_{-p..-p+k-1} in RR's FLCI).
function _w_to_l_matrix(p::Int, ::Type{T}) where {T<:AbstractFloat}
    W = Matrix{T}(I, p, p)
    for c in 1:(p-1)
        W[c+1, c] = -one(T)
    end
    W
end

# =============================================================================
# Delta^SD(M) — fixed-length confidence interval (FLCI)
# =============================================================================

# Variance of the affine estimator theta_hat = l_pre'beta_pre + l_vec'beta_post
# as a quadratic in the w-parametrization: var(w) = w'Qw w + qw'w + c0.
function _flci_variance_terms(sigma::Matrix{T}, p::Int, q::Int,
                              l_vec::Vector{T}) where {T<:AbstractFloat}
    Sig_pre = sigma[1:p, 1:p]
    Sig_pp = sigma[1:p, (p+1):(p+q)]
    Sig_post = sigma[(p+1):(p+q), (p+1):(p+q)]
    WtoL = _w_to_l_matrix(p, T)
    Qw = WtoL' * Sig_pre * WtoL
    qw = 2 .* (WtoL' * (Sig_pp * l_vec))
    c0 = dot(l_vec, Sig_post * l_vec)
    (Qw, qw, c0, WtoL)
end

_flci_variance(w::AbstractVector{T}, Qw, qw, c0) where {T} =
    dot(w, Qw * w) + dot(qw, w) + c0

# Minimum-variance w subject to sum(w) = K (equality-constrained QP, closed form).
function _flci_min_variance_w(Qw::Matrix{T}, qw::Vector{T}, K::T, p::Int) where {T<:AbstractFloat}
    KKT = [2 .* Qw ones(T, p); ones(T, 1, p) zero(T)]
    rhs = vcat(-qw, K)
    sol = Matrix{T}(robust_inv(KKT; silent=true)) * rhs
    sol[1:p]
end

# Constant part of the worst-case bias over Delta^SD(1) for target l_vec
# (RR / HonestDiD .createObjectiveObjectForBias).
function _flci_bias_constant(l_vec::Vector{T}, q::Int) where {T<:AbstractFloat}
    con = zero(T)
    for s in 1:q
        acc = zero(T)
        for i in 1:s
            acc += T(i) * l_vec[q-s+i]
        end
        con += abs(acc)
    end
    con - dot(T.(1:q), l_vec)
end

# Worst-case bias (M = 1 scale, before adding the constant) of the affine
# estimator subject to sd <= h: SLSQP over x = [U; w] with U_k >= |cumsum(w)_k|.
function _flci_worst_case_bias(h::T, Qw::Matrix{T}, qw::Vector{T}, c0::T,
                               K::T, p::Int, w0::Vector{T}) where {T<:AbstractFloat}
    n = 2p
    opt = NLopt.Opt(:LD_SLSQP, n)

    NLopt.min_objective!(opt, (x, g) -> begin
        if length(g) > 0
            for j in 1:p; g[j] = 1.0; end
            for j in (p+1):n; g[j] = 0.0; end
        end
        s = 0.0
        for j in 1:p; s += x[j]; end
        s
    end)

    # |cumsum(w)_k| <= U_k, cast as two linear inequalities per k
    for k in 1:p, sgn in (1.0, -1.0)
        let k = k, sgn = sgn
            NLopt.inequality_constraint!(opt, (x, g) -> begin
                if length(g) > 0
                    fill!(g, 0.0)
                    g[k] = -1.0
                    for j in 1:k; g[p+j] = sgn; end
                end
                cs = 0.0
                for j in 1:k; cs += x[p+j]; end
                -x[k] + sgn * cs
            end, 1e-10)
        end
    end

    # sum(w) = K (removes the unbounded linear-trend bias direction)
    NLopt.equality_constraint!(opt, (x, g) -> begin
        if length(g) > 0
            fill!(g, 0.0)
            for j in (p+1):n; g[j] = 1.0; end
        end
        s = 0.0
        for j in (p+1):n; s += x[j]; end
        s - Float64(K)
    end, 1e-10)

    # variance constraint: var(w) <= h^2
    Qw64 = Float64.(Qw); qw64 = Float64.(qw); c064 = Float64(c0); h64 = Float64(h)
    NLopt.inequality_constraint!(opt, (x, g) -> begin
        w = x[(p+1):n]
        Qwv = Qw64 * w
        if length(g) > 0
            for j in 1:p; g[j] = 0.0; end
            for j in 1:p; g[p+j] = 2.0 * Qwv[j] + qw64[j]; end
        end
        dot(w, Qwv) + dot(qw64, w) + c064 - h64^2
    end, 1e-10)

    NLopt.xtol_rel!(opt, 1e-10)
    NLopt.maxeval!(opt, 4000)

    x0 = vcat(abs.(cumsum(Float64.(w0))) .+ 1e-8, Float64.(w0))
    (fval, xopt, ret) = NLopt.optimize(opt, x0)
    (T(fval), T.(xopt[(p+1):n]), ret)
end

"""
    _flci_delta_sd(betahat, sigma, p, q; M, l_vec, alpha) -> NamedTuple

Armstrong-Kolesár fixed-length CI for `theta = l_vec'tau_post` under
`Delta^SD(M)` (port of HonestDiD `findOptimalFLCI`). Minimizes the half-length
`cv_alpha(M·bias(h)/h)·h` over the sd level `h` of affine estimators
`theta_hat = l_pre'beta_pre + l_vec'beta_post` whose weights satisfy the
slope-cancellation constraint. Returns the CI, half-length, and optimal weights.
"""
function _flci_delta_sd(betahat::Vector{T}, sigma::Matrix{T}, p::Int, q::Int;
                        M::T=zero(T), l_vec::Vector{T}=_basis_vector(1, q, T),
                        alpha::T=T(0.05)) where {T<:AbstractFloat}
    Qw, qw, c0, WtoL = _flci_variance_terms(sigma, p, q, l_vec)
    K = dot(T.(1:q), l_vec)
    bias_con = _flci_bias_constant(l_vec, q)

    w_min = _flci_min_variance_w(Qw, qw, K, p)
    h_min = sqrt(max(_flci_variance(w_min, Qw, qw, c0), zero(T)))
    w_h0 = vcat(zeros(T, p - 1), K)
    h0 = sqrt(max(_flci_variance(w_h0, Qw, qw, c0), zero(T)))

    halflen_at = function (h::T)
        bias_w, w_opt, _ = _flci_worst_case_bias(h, Qw, qw, c0, K, p, w_min)
        bias = M * (bias_w + bias_con)
        hl = _fold_normal_cv(h > zero(T) ? bias / h : zero(T), alpha) * h
        (hl, w_opt)
    end

    local hstar::T, w_star::Vector{T}, hl_star::T
    if h0 - h_min < T(1e-12) * max(h0, one(T)) || p == 1
        hstar = h_min
        hl_star, w_star = halflen_at(hstar)
    else
        # Golden-section search on the convex half-length over [h_min, h0]
        gr = T((sqrt(5) - 1) / 2)
        a, b = h_min, h0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, _ = halflen_at(c)
        fd, _ = halflen_at(d)
        for _ in 1:60
            if b - a < T(1e-8) * max(h0, one(T))
                break
            end
            if fc < fd
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc, _ = halflen_at(c)
            else
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd, _ = halflen_at(d)
            end
        end
        hstar = (a + b) / 2
        hl_star, w_star = halflen_at(hstar)
        # Guard against endpoint minima
        hl_min, w_min_opt = halflen_at(h_min)
        hl_h0, w_h0_opt = halflen_at(h0)
        if hl_min <= hl_star && hl_min <= hl_h0
            hstar, hl_star, w_star = h_min, hl_min, w_min_opt
        elseif hl_h0 < hl_star
            hstar, hl_star, w_star = h0, hl_h0, w_h0_opt
        end
    end

    l_pre = WtoL * w_star
    theta = dot(l_pre, betahat[1:p]) + dot(l_vec, betahat[(p+1):(p+q)])

    (ci_lower = theta - hl_star, ci_upper = theta + hl_star,
     halflength = hl_star, theta = theta,
     optimal_l = vcat(l_pre, l_vec))
end

# =============================================================================
# Delta^RM(Mbar) — identified set and delta-method robust CI
# =============================================================================

# Pre-treatment first differences of [beta_pre; 0] (the trailing 0 is the
# normalized reference period): p differences, the last being 0 - beta_pre[p].
function _deltarm_pre_diffs(beta_pre::Vector{T}) where {T<:AbstractFloat}
    p = length(beta_pre)
    d = Vector{T}(undef, p)
    for r in 1:(p-1)
        d[r] = beta_pre[r+1] - beta_pre[r]
    end
    d[p] = -beta_pre[p]
    d
end

"""
    _deltarm_identified_set(betahat, p, q; Mbar, l_vec) -> (lb, ub)

Identified set for `theta = l_vec'tau_post` under `Delta^RM(Mbar)`, matching the
union-of-polyhedra linear programs of HonestDiD `.compute_IDset_DeltaRM` in
closed form: with `c = max_s |Δdelta_s^pre|` (delta_pre = betahat_pre under
tau_pre = 0), post first differences are bounded by `Mbar·c`, so
`max/min l'delta_post = ± Mbar·c·G` with `G = Σ_i |Σ_{j>=i} l_j|`.
"""
function _deltarm_identified_set(betahat::Vector{T}, p::Int, q::Int;
                                 Mbar::T=one(T),
                                 l_vec::Vector{T}=_basis_vector(1, q, T)) where {T<:AbstractFloat}
    d = _deltarm_pre_diffs(betahat[1:p])
    c = maximum(abs, d)
    G = sum(abs(sum(@view l_vec[i:q])) for i in 1:q)
    theta = dot(l_vec, betahat[(p+1):(p+q)])
    (theta - Mbar * c * G, theta + Mbar * c * G)
end

"""
    _deltarm_robust_ci(betahat, sigma, p, q; Mbar, l_vec, alpha) -> NamedTuple

Robust CI for `theta` under `Delta^RM(Mbar)`: the identified-set bounds widened
by delta-method standard errors of each bound function (the bound is
`l'beta_post ± Mbar·G·c(beta_pre)` with `c` the max absolute pre-period first
difference — piecewise smooth, differentiated at the argmax).

This is a documented interim for the ARP (Andrews-Roth-Pakes) conditional
moment-inequality confidence set used by the R package's C-LF method: it is
pre-period-dependent and scale-covariant with `Mbar` dimensionless, but its
coverage is pointwise rather than uniform.
"""
function _deltarm_robust_ci(betahat::Vector{T}, sigma::Matrix{T}, p::Int, q::Int;
                            Mbar::T=one(T),
                            l_vec::Vector{T}=_basis_vector(1, q, T),
                            alpha::T=T(0.05)) where {T<:AbstractFloat}
    d = _deltarm_pre_diffs(betahat[1:p])
    rstar = argmax(abs.(d))
    sgn = d[rstar] >= zero(T) ? one(T) : -one(T)
    c = abs(d[rstar])
    G = sum(abs(sum(@view l_vec[i:q])) for i in 1:q)
    theta = dot(l_vec, betahat[(p+1):(p+q)])

    # Gradient of c(beta_pre) at the argmax first difference
    gc = zeros(T, p + q)
    if rstar < p
        gc[rstar] = -sgn
        gc[rstar+1] = sgn
    else
        gc[p] = -sgn
    end

    # Gradient of the upper/lower bound functions
    g_theta = vcat(zeros(T, p), l_vec)
    g_ub = g_theta .+ (Mbar * G) .* gc
    g_lb = g_theta .- (Mbar * G) .* gc

    se_ub = sqrt(max(dot(g_ub, sigma * g_ub), zero(T)))
    se_lb = sqrt(max(dot(g_lb, sigma * g_lb), zero(T)))
    z = T(quantile(Normal(), 1 - alpha / 2))

    (ci_lower = theta - Mbar * c * G - z * se_lb,
     ci_upper = theta + Mbar * c * G + z * se_ub,
     id_lb = theta - Mbar * c * G, id_ub = theta + Mbar * c * G, theta = theta)
end

# =============================================================================
# Breakdown values
# =============================================================================

# Smallest bound value at which the robust CI covers 0 (bisection; the CI is
# nested-increasing in the bound for both restrictions in practice).
function _honest_breakdown(ci_at::Function, ::Type{T}) where {T<:AbstractFloat}
    covers = function (m::T)
        lo, hi = ci_at(m)
        lo <= zero(T) <= hi
    end
    covers(zero(T)) && return zero(T)
    m_hi = one(T)
    n_expand = 0
    while !covers(m_hi) && n_expand < 60
        m_hi *= 2
        n_expand += 1
    end
    covers(m_hi) || return T(Inf)
    m_lo = zero(T)
    for _ in 1:48
        mid = (m_lo + m_hi) / 2
        if covers(mid)
            m_hi = mid
        else
            m_lo = mid
        end
    end
    (m_lo + m_hi) / 2
end

# =============================================================================
# Core API — (betahat, sigma)
# =============================================================================

"""
    honest_did(betahat, sigma; num_pre, num_post, restriction=:rm,
               Mbar=1.0, M=0.0, l_vec=nothing, conf_level=0.95) -> NamedTuple

Rambachan-Roth (2023) robust confidence set for `theta = l_vec'tau_post` from
event-study coefficients `betahat = [beta_pre; beta_post]` (reference period
omitted) with joint covariance `sigma`.

# Arguments
- `betahat` — `num_pre + num_post` event-study coefficients, pre first
- `sigma` — joint covariance of `betahat`
- `restriction` — `:rm` (relative magnitudes `Delta^RM(Mbar)`) or `:sd`
  (second differences `Delta^SD(M)`)
- `Mbar` — relative-magnitudes bound (dimensionless; used when `restriction=:rm`)
- `M` — smoothness bound in outcome units (used when `restriction=:sd`)
- `l_vec` — weights on the post-period effects (default `e_1`)
- `conf_level` — confidence level

# Returns
NamedTuple with `ci_lower`, `ci_upper`, `theta` (point estimate of the target),
`original_ci_lower/upper` (conventional CI ignoring trend violations),
`breakdown` (smallest bound at which the robust CI covers 0), `restriction`,
and `method` (`:flci` for `:sd`, `:delta_id` for `:rm`).

For `:rm` the CI is the closed-form identified set (matching the R package's
LP union exactly) widened by delta-method standard errors — a documented
interim for the ARP conditional confidence set. For `:sd` it is the
Armstrong-Kolesár FLCI, matching the R package's `findOptimalFLCI`.

# References
- Rambachan, A. & Roth, J. (2023). *Review of Economic Studies* 90(5), 2555-2591.
- Armstrong, T. & Kolesár, M. (2018). *Econometrica* 86(2), 655-683.
"""
function honest_did(betahat::AbstractVector{<:Real}, sigma::AbstractMatrix{<:Real};
                    num_pre::Int, num_post::Int,
                    restriction::Symbol=:rm,
                    Mbar::Real=1.0, M::Real=0.0,
                    l_vec::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    conf_level::Real=0.95)
    T = float(promote_type(eltype(betahat), eltype(sigma)))
    p, q = num_pre, num_post
    length(betahat) == p + q ||
        throw(ArgumentError("betahat must have num_pre + num_post = $(p+q) elements"))
    size(sigma) == (p + q, p + q) ||
        throw(ArgumentError("sigma must be $(p+q)x$(p+q)"))
    p >= 1 || throw(ArgumentError("need at least one pre-treatment coefficient"))
    q >= 1 || throw(ArgumentError("need at least one post-treatment coefficient"))
    restriction in (:rm, :sd) ||
        throw(ArgumentError("restriction must be :rm or :sd; got :$restriction"))

    b = Vector{T}(betahat)
    S = Matrix{T}(sigma)
    lv = l_vec === nothing ? _basis_vector(1, q, T) : Vector{T}(l_vec)
    length(lv) == q || throw(ArgumentError("l_vec must have num_post = $q elements"))
    alpha = one(T) - T(conf_level)

    theta = dot(lv, b[(p+1):(p+q)])
    sd_conv = sqrt(max(dot(lv, S[(p+1):(p+q), (p+1):(p+q)] * lv), zero(T)))
    z = T(quantile(Normal(), 1 - alpha / 2))
    orig_lo = theta - z * sd_conv
    orig_hi = theta + z * sd_conv

    if restriction == :rm
        r = _deltarm_robust_ci(b, S, p, q; Mbar=T(Mbar), l_vec=lv, alpha=alpha)
        bd = _honest_breakdown(T) do m
            rr = _deltarm_robust_ci(b, S, p, q; Mbar=m, l_vec=lv, alpha=alpha)
            (rr.ci_lower, rr.ci_upper)
        end
        return (ci_lower = r.ci_lower, ci_upper = r.ci_upper, theta = theta,
                original_ci_lower = orig_lo, original_ci_upper = orig_hi,
                breakdown = bd, restriction = :rm, Mbar = T(Mbar), M = T(M),
                method = :delta_id)
    else
        r = _flci_delta_sd(b, S, p, q; M=T(M), l_vec=lv, alpha=alpha)
        bd = _honest_breakdown(T) do m
            rr = _flci_delta_sd(b, S, p, q; M=m, l_vec=lv, alpha=alpha)
            (rr.ci_lower, rr.ci_upper)
        end
        return (ci_lower = r.ci_lower, ci_upper = r.ci_upper, theta = r.theta,
                original_ci_lower = orig_lo, original_ci_upper = orig_hi,
                breakdown = bd, restriction = :sd, Mbar = T(Mbar), M = T(M),
                method = :flci)
    end
end

# =============================================================================
# Dispatch — DIDResult / EventStudyLP
# =============================================================================

# Assemble (betahat, sigma, pre/post indices) from an event-study layout.
function _honest_assemble(att::Vector{T}, se::Vector{T}, event_times::Vector{Int},
                          reference_period::Int,
                          att_vcov::Union{Matrix{T},Nothing}) where {T<:AbstractFloat}
    valid = findall(i -> isfinite(att[i]) && event_times[i] != reference_period,
                    eachindex(att))
    pre_idx = [i for i in valid if event_times[i] < reference_period]
    post_idx = [i for i in valid if event_times[i] > reference_period]
    isempty(pre_idx) &&
        throw(ArgumentError("honest_did requires at least one pre-treatment coefficient " *
                            "(event time before the reference period $reference_period)"))
    isempty(post_idx) &&
        throw(ArgumentError("No post-treatment periods found. Cannot compute honest DiD."))

    ord = vcat(pre_idx, post_idx)
    betahat = att[ord]
    if att_vcov === nothing
        @warn "no joint event-study covariance available; using a diagonal Σ from the " *
              "per-period standard errors — cross-period covariance is ignored" maxlog = 1
        sigma = Matrix{T}(Diagonal(se[ord] .^ 2))
    else
        sigma = att_vcov[ord, ord]
    end
    (betahat, sigma, length(pre_idx), length(post_idx), event_times[post_idx],
     att[post_idx], se[post_idx])
end

function _honest_did_eventstudy(att::Vector{T}, se::Vector{T}, event_times::Vector{Int},
                                reference_period::Int,
                                att_vcov::Union{Matrix{T},Nothing};
                                restriction::Symbol, Mbar::T, M::T,
                                conf_level::T) where {T<:AbstractFloat}
    betahat, sigma, p, q, post_times, post_att, post_se =
        _honest_assemble(att, se, event_times, reference_period, att_vcov)

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    robust_lo = Vector{T}(undef, q)
    robust_hi = Vector{T}(undef, q)
    orig_lo = Vector{T}(undef, q)
    orig_hi = Vector{T}(undef, q)
    breakdown = T(Inf)

    for k in 1:q
        res = honest_did(betahat, sigma; num_pre=p, num_post=q,
                         restriction=restriction, Mbar=Mbar, M=M,
                         l_vec=_basis_vector(k, q, T), conf_level=conf_level)
        robust_lo[k] = res.ci_lower
        robust_hi[k] = res.ci_upper
        orig_lo[k] = res.original_ci_lower
        orig_hi[k] = res.original_ci_upper
        breakdown = min(breakdown, res.breakdown)
    end

    HonestDiDResult{T}(
        Mbar, robust_lo, robust_hi, orig_lo, orig_hi,
        breakdown, post_times, post_att, conf_level,
        restriction, M, restriction == :sd ? :flci : :delta_id
    )
end

"""
    honest_did(result::DIDResult; restriction=:rm, Mbar=1.0, M=0.0, conf_level=0.95)

Rambachan-Roth (2023) sensitivity analysis for a DiD event-study result. Runs
the robust confidence set for every post-treatment period (target `l_vec = e_k`)
and reports the per-period robust CIs plus the breakdown value (smallest bound
at which any post-period robust CI covers zero).

Uses the joint event-study covariance `result.att_vcov` when available;
otherwise falls back to a diagonal covariance built from the per-period
standard errors (with a warning).

# Example
```julia
did = estimate_did(pd, :y, :treat)
h = honest_did(did; restriction=:rm, Mbar=1.0)
h.breakdown_value  # smallest Mbar that overturns significance
```
"""
function honest_did(result::DIDResult{T}; restriction::Symbol=:rm,
                    Mbar::Real=1.0, M::Real=0.0,
                    conf_level::Real=0.95) where {T<:AbstractFloat}
    _honest_did_eventstudy(result.att, result.se, result.event_times,
                           result.reference_period, result.att_vcov;
                           restriction=restriction, Mbar=T(Mbar), M=T(M),
                           conf_level=T(conf_level))
end

"""
    honest_did(result::EventStudyLP; restriction=:rm, Mbar=1.0, M=0.0, conf_level=0.95)

Rambachan-Roth (2023) sensitivity analysis for an event-study LP result. The
cross-horizon covariance is not stored for LP event studies, so a diagonal
covariance from the per-horizon standard errors is used (with a warning).
"""
function honest_did(result::EventStudyLP{T}; restriction::Symbol=:rm,
                    Mbar::Real=1.0, M::Real=0.0,
                    conf_level::Real=0.95) where {T<:AbstractFloat}
    _honest_did_eventstudy(result.coefficients, result.se, result.event_times,
                           result.reference_period, nothing;
                           restriction=restriction, Mbar=T(Mbar), M=T(M),
                           conf_level=T(conf_level))
end
