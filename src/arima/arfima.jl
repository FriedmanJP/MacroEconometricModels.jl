# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
ARFIMA(p,d,q) — fractionally integrated ARMA (long memory).

Provides:
- Fractional-differencing filter helpers `_frac_diff_weights` / `_frac_diff`
  ((1−L)^d weights π₀=1, π_k = π_{k−1}(k−1−d)/k) with a direct O(T·K) recursion
  and an FFT convolution path (Jensen & Nielsen 2014, O(T log T)).
  **EV-14 (FIGARCH) reuses `_frac_diff_weights`** — keep its name/signature stable.
- `estimate_arfima(y, p, q; method=:css | :mle)` estimating d ∈ (−0.5, 0.5)
  jointly with the ARMA(p,q) parameters. CSS filters y by (1−L)^d and fits the
  ARMA conditional sum of squares; exact ML uses the Durbin–Levinson recursion over
  the Sowell (1992) / Hosking (1981) ARFIMA autocovariances (O(T²) path).
- Semiparametric d estimators `gph_test` (Geweke & Porter-Hudak 1983 log-periodogram
  regression) and `local_whittle` (Robinson 1995).

d is parameterized by a logit ξ ↦ d = −0.5 + 1/(1+e^{−ξ}) so the optimizer stays
strictly inside (−0.5, 0.5) (boundary d breaks the ACF and the CSS objective).
"""

using LinearAlgebra, Statistics, Distributions
using Distributions: loggamma
import Optim
import FFTW

# =============================================================================
# Fractional-differencing filter  (shared — EV-14 reuses `_frac_diff_weights`)
# =============================================================================

"""
    _frac_diff_weights(d, K) -> Vector

Coefficients [π₀, π₁, …, π_K] of the fractional-difference filter (1−L)^d, from
the recursion π₀ = 1, π_k = π_{k−1}·(k−1−d)/k. For `d == 0` returns `[1, 0, …, 0]`.

Reused by EV-14 (FIGARCH) — keep the name and signature stable.
"""
function _frac_diff_weights(d::T, K::Int) where {T<:AbstractFloat}
    K < 0 && throw(ArgumentError("K must be non-negative, got $K"))
    w = Vector{T}(undef, K + 1)
    w[1] = one(T)
    @inbounds for k in 1:K
        w[k+1] = w[k] * (T(k - 1) - d) / T(k)
    end
    w
end
_frac_diff_weights(d::Real, K::Int) = _frac_diff_weights(float(d), K)

"""
    _frac_diff(y, d; method=:auto) -> Vector

Apply the (1−L)^d filter to `y`, i.e. wₜ = Σ_{k=0}^{t−1} π_k y_{t−k} (pre-sample
values treated as zero). Returns a vector the same length as `y`.

`method`:
- `:direct` — O(T·K) recursion (default for short series / FIGARCH truncation).
- `:fft` — O(T log T) linear convolution via FFTW (Jensen & Nielsen 2014); use for
  the exact-ML long-series path.
- `:auto` — `:fft` when `length(y) > 512`, else `:direct`.

`_frac_diff(y, 0)` returns `y` unchanged (up to float roundoff).
"""
function _frac_diff(y::AbstractVector{T}, d::Real; method::Symbol=:auto) where {T<:AbstractFloat}
    n = length(y)
    n == 0 && return T[]
    dd = T(d)
    w = _frac_diff_weights(dd, n - 1)
    use_fft = method == :fft || (method == :auto && n > 512)
    if use_fft
        return _frac_diff_fft(Vector{T}(y), w, n)
    elseif method == :direct || method == :auto
        out = zeros(T, n)
        @inbounds for t in 1:n
            s = zero(T)
            for k in 0:t-1
                s += w[k+1] * y[t-k]
            end
            out[t] = s
        end
        return out
    else
        throw(ArgumentError("Unknown method: $method. Use :direct, :fft, or :auto."))
    end
end
_frac_diff(y::AbstractVector, d::Real; kw...) = _frac_diff(Float64.(y), d; kw...)

"""FFT linear convolution of filter weights `w` with `y`, keeping the first `n` terms."""
function _frac_diff_fft(y::Vector{T}, w::Vector{T}, n::Int) where {T<:AbstractFloat}
    L = nextpow(2, 2n - 1)
    yp = zeros(Float64, L); yp[1:n] .= y
    wp = zeros(Float64, L); wp[1:n] .= w
    conv = FFTW.irfft(FFTW.rfft(yp) .* FFTW.rfft(wp), L)
    T.(conv[1:n])
end

# =============================================================================
# ARFIMA autocovariances (Sowell 1992 / Hosking 1981)
# =============================================================================

"""
    _arfima0_autocov(d, sigma2, maxlag) -> Vector

Autocovariances γ(0),…,γ(maxlag) of the pure fractional process ARFIMA(0,d,0)
(Hosking 1981, closed form):

    γ(0) = σ² Γ(1−2d) / Γ(1−d)²,   γ(k) = γ(k−1)·(k−1+d)/(k−d).

The lag-1 autocorrelation is ρ(1) = d/(1−d).
"""
function _arfima0_autocov(d::T, sigma2::T, maxlag::Int) where {T<:AbstractFloat}
    g = zeros(T, maxlag + 1)
    if d == zero(T)
        g[1] = sigma2
        return g
    end
    # Γ(1−2d)/Γ(1−d)² via loggamma (stable); requires d < 0.5.
    g[1] = sigma2 * exp(loggamma(one(T) - 2d) - 2 * loggamma(one(T) - d))
    @inbounds for k in 1:maxlag
        g[k+1] = g[k] * (T(k - 1) + d) / (T(k) - d)
    end
    g
end

"""
    _arfima_autocov(d, phi, theta, sigma2, maxlag; trunc=200) -> Vector

Autocovariances of ARFIMA(p,d,q). The short-memory ARMA operator is applied to the
fractional noise via its MA(∞) ψ-weights:

    γ_x(h) = Σ_{i,j≥0} ψ_i ψ_j γ_u(h+i−j),

with γ_u the pure-fractional autocovariance and ψ truncated at `trunc`. For p=q=0
this reduces exactly to `_arfima0_autocov`.
"""
function _arfima_autocov(d::T, phi::Vector{T}, theta::Vector{T}, sigma2::T,
                         maxlag::Int; trunc::Int=200) where {T<:AbstractFloat}
    (isempty(phi) && isempty(theta)) && return _arfima0_autocov(d, sigma2, maxlag)
    L = min(trunc, maxlag + 50)
    psi = vcat(one(T), _compute_psi_weights(phi, theta, L))   # ψ₀=1, …, ψ_L
    gu = _arfima0_autocov(d, sigma2, maxlag + L)              # γ_u up to maxlag+L
    guf(k::Int) = gu[abs(k)+1]
    g = zeros(T, maxlag + 1)
    @inbounds for h in 0:maxlag
        s = zero(T)
        for i in 0:L, j in 0:L
            s += psi[i+1] * psi[j+1] * guf(h + i - j)
        end
        g[h+1] = s
    end
    g
end

# =============================================================================
# Exact Gaussian log-likelihood via Durbin–Levinson
# =============================================================================

"""
    _dl_concentrated_loglik(x, r) -> (loglik, sigma2, innovations)

Exact Gaussian log-likelihood of a mean-zero stationary series `x` whose
autocovariance (up to a scalar σ²) is `r` (r[1]=γ₀,…). The Durbin–Levinson
recursion produces the one-step prediction errors and variances; σ² is
concentrated out. Returns the concentrated log-likelihood, σ̂², and the raw
one-step innovations e_t = x_t − x̂_t.
"""
function _dl_concentrated_loglik(x::Vector{T}, r::Vector{T}) where {T<:AbstractFloat}
    n = length(x)
    phi = zeros(T, n)
    phiprev = zeros(T, n)
    e = zeros(T, n)
    v = r[1]
    (v <= zero(T) || !isfinite(v)) && return (T(-Inf), T(NaN), e)
    e[1] = x[1]
    logdet = log(v)
    ssr = e[1]^2 / v
    @inbounds for k in 1:n-1
        acc = r[k+1]
        for j in 1:k-1
            acc -= phiprev[j] * r[k-j+1]
        end
        phikk = acc / v
        phi[k] = phikk
        for j in 1:k-1
            phi[j] = phiprev[j] - phikk * phiprev[k-j]
        end
        v = v * (one(T) - phikk^2)
        (v <= zero(T) || !isfinite(v)) && return (T(-Inf), T(NaN), e)
        xhat = zero(T)
        for j in 1:k
            xhat += phi[j] * x[k+1-j]
        end
        e[k+1] = x[k+1] - xhat
        logdet += log(v)
        ssr += e[k+1]^2 / v
        for j in 1:k
            phiprev[j] = phi[j]
        end
    end
    sigma2 = ssr / n
    loglik = -T(0.5) * (n * log(T(2π)) + n * log(sigma2) + logdet + n)
    (loglik, sigma2, e)
end

# =============================================================================
# ARFIMA estimation
# =============================================================================

# logit maps ξ (unconstrained) ↦ d ∈ (−0.5, 0.5)
_arfima_d(xi::T) where {T} = -T(0.5) + one(T) / (one(T) + exp(-xi))
_arfima_xi(d::T) where {T} = log((d + T(0.5)) / (T(0.5) - d))
# dd/dξ = (d+0.5)(0.5−d) for the delta method
_arfima_ddxi(d::T) where {T} = (d + T(0.5)) * (T(0.5) - d)

"""
    _arfima_components(d, c, phi, theta, y, method; trunc) -> (loglik, sigma2, residuals, fitted)

Evaluate the ARFIMA log-likelihood and one-step innovations for a parameter set.
- `:css` — filter y by (1−L)^d, fit ARMA(p,q) conditional sum of squares (`c` is the
  ARMA intercept on the filtered series); conditional Gaussian log-likelihood.
- `:mle` — exact Gaussian log-likelihood on (y−c) via Durbin–Levinson over the
  ARFIMA autocovariances (`c` is the process mean μ).
"""
function _arfima_components(d::T, c::T, phi::Vector{T}, theta::Vector{T},
                           y::Vector{T}, method::Symbol; trunc::Int=200) where {T<:AbstractFloat}
    n = length(y)
    if method == :css
        w = _frac_diff(y, d; method=:direct)
        resid = _compute_arma_residuals(w, c, phi, theta)
        mmax = max(length(phi), length(theta))
        vr = @view resid[mmax+1:end]
        n_eff = length(vr)
        sigma2 = sum(abs2, vr) / n_eff
        loglik = -T(0.5) * n_eff * (log(T(2π)) + log(sigma2) + one(T))
        fitted = y .- resid
        return (loglik, sigma2, resid, fitted)
    elseif method == :mle
        x = y .- c
        r = _arfima_autocov(d, phi, theta, one(T), n - 1; trunc=trunc)
        loglik, sigma2, e = _dl_concentrated_loglik(x, r)
        fitted = y .- e
        return (loglik, sigma2, e, fitted)
    else
        throw(ArgumentError("Unknown method: $method. Use :css or :mle."))
    end
end

"""Negative (penalized) log-likelihood over the unconstrained vector [ξ, c, φ…, θ…]."""
function _arfima_negloglik(pv::Vector{T}, y::Vector{T}, p::Int, q::Int,
                           method::Symbol; trunc::Int=200) where {T<:AbstractFloat}
    xi = pv[1]
    d = _arfima_d(xi)
    c = pv[2]
    phi = p > 0 ? pv[3:2+p] : T[]
    theta = q > 0 ? pv[3+p:2+p+q] : T[]
    penalty = T(1e10)
    (!_is_stationary(phi) || !_is_invertible(theta)) && return penalty
    loglik, _, _, _ = _arfima_components(d, c, phi, theta, y, method; trunc=trunc)
    (isnan(loglik) || isinf(loglik)) && return penalty
    -loglik
end

"""
    estimate_arfima(y, p, q; method=:css, d0=nothing, trunc=200, max_iter=500) -> ARFIMAModel

Estimate an ARFIMA(p,d,q) model with fractional integration order d ∈ (−0.5, 0.5).

# Arguments
- `y`: Time series vector.
- `p`, `q`: AR and MA orders.
- `method`: `:css` (conditional sum of squares, default) or `:mle` (exact Gaussian
  ML via Durbin–Levinson; O(T²)).
- `d0`: Optional starting value for d (default: a GPH pre-estimate).
- `trunc`: ψ-weight truncation for the exact-ML autocovariances (ignored by `:css`).
- `max_iter`: Maximum optimizer iterations.

# Returns
`ARFIMAModel` with the fractional order `d`, its standard error `d_se`, the ARMA
coefficients, and standard diagnostics.

# Example
```julia
y = load_example(:nile).data[:, 1]
m = estimate_arfima(y, 0, 0; method=:css)
println(m.d)          # ≈ long-memory parameter
```
"""
function estimate_arfima(y::AbstractVector{T}, p::Int, q::Int;
                         method::Symbol=:css, d0::Union{Nothing,Real}=nothing,
                         trunc::Int=200, max_iter::Int=500) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_arima_inputs(y, p, 0, q)
    (method == :css || method == :mle) ||
        throw(ArgumentError("Unknown method: $method. Use :css or :mle."))
    y_vec = Vector{T}(y)
    n = length(y_vec)

    # Starting values
    if d0 === nothing
        dstart = try
            clamp(T(gph_test(y_vec).d), T(-0.45), T(0.45))
        catch
            T(0.1)
        end
    else
        dstart = clamp(T(d0), T(-0.49), T(0.49))
    end
    xi0 = _arfima_xi(dstart)
    c0 = mean(y_vec)
    phi0 = _yule_walker(y_vec, p)
    theta0 = _innovations_algorithm(y_vec, q)
    pv0 = vcat(xi0, c0, phi0, theta0)

    obj = pv -> _arfima_negloglik(pv, y_vec, p, q, method; trunc=trunc)

    # LBFGS (numerical gradient) per spec; NelderMead fallback for robustness.
    result = Optim.optimize(obj, pv0, Optim.LBFGS(),
        Optim.Options(iterations=max_iter, g_tol=T(1e-8), show_trace=false))
    converged = Optim.converged(result)
    if !converged || Optim.minimum(result) >= T(1e9)
        result2 = Optim.optimize(obj, pv0, Optim.NelderMead(),
            Optim.Options(iterations=max_iter, show_trace=false))
        if Optim.minimum(result2) < Optim.minimum(result)
            result = result2
            converged = Optim.converged(result2)
        end
    end

    pv = Optim.minimizer(result)
    iterations = Optim.iterations(result)
    d = _arfima_d(pv[1])
    c = pv[2]
    phi = p > 0 ? pv[3:2+p] : T[]
    theta = q > 0 ? pv[3+p:2+p+q] : T[]

    loglik, sigma2, residuals, fitted =
        _arfima_components(d, c, phi, theta, y_vec, method; trunc=trunc)

    # d standard error: numerical Hessian of the negative log-likelihood, delta-method
    # back-transform through the logit (dd/dξ = (d+0.5)(0.5−d)).
    d_se = _arfima_dse(pv, y_vec, p, q, method, d, trunc)

    k = p + q + 3                       # c/μ + d + φ + θ + σ²
    N_ic = method == :css ? (n - max(p, q)) : n
    aic, bic = _compute_aic_bic(loglik, k, N_ic)

    ARFIMAModel(y_vec, p, d, q, c, phi, theta, sigma2, d_se, residuals, fitted,
                loglik, aic, bic, method, converged, iterations)
end

estimate_arfima(y::AbstractVector, p::Int, q::Int; kwargs...) =
    estimate_arfima(Float64.(y), p, q; kwargs...)

"""d standard error via the numerical Hessian of the negloglik + logit delta method."""
function _arfima_dse(pv::Vector{T}, y::Vector{T}, p::Int, q::Int, method::Symbol,
                     d::T, trunc::Int) where {T<:AbstractFloat}
    obj = p2 -> _arfima_negloglik(p2, y, p, q, method; trunc=trunc)
    try
        H = _numerical_hessian(obj, pv)
        C = robust_inv(H)
        se_xi = sqrt(max(C[1, 1], zero(T)))
        abs(_arfima_ddxi(d)) * se_xi
    catch
        T(NaN)
    end
end

"""
    _arfima_stderror(m) -> Vector

Standard errors aligned to `coef(m) = [c, d, φ…, θ…]`, from the numerical Hessian of
the negative log-likelihood in the unconstrained (ξ, c, φ, θ) space (ξ→d via the
logit delta method).
"""
function _arfima_stderror(m::ARFIMAModel)
    T = eltype(m.y)
    p, q = m.p, m.q
    pv = vcat(_arfima_xi(m.d), m.c, m.phi, m.theta)
    obj = p2 -> _arfima_negloglik(p2, m.y, p, q, m.method)
    se = try
        H = _numerical_hessian(obj, pv)
        C = robust_inv(H)
        sqrt.(max.(diag(C), zero(T)))
    catch
        return fill(T(NaN), 2 + p + q)
    end
    se_d = abs(_arfima_ddxi(m.d)) * se[1]
    vcat(se[2], se_d, se[3:2+p], se[3+p:2+p+q])
end

StatsAPI.stderror(m::ARFIMAModel) = _arfima_stderror(m)

# =============================================================================
# GPH log-periodogram regression (Geweke & Porter-Hudak 1983)
# =============================================================================

"""Periodogram I(λ_j) at Fourier frequencies λ_j = 2πj/n, j = 1,…,⌊n/2⌋."""
function _periodogram(x::Vector{T}) where {T<:AbstractFloat}
    n = length(x)
    X = FFTW.fft(complex.(x))
    M = div(n, 2)
    lam = T[2π * j / n for j in 1:M]
    I = T[abs2(X[j+1]) / (2π * n) for j in 1:M]    # constant cancels in d estimation
    (lam, I)
end

"""
    gph_test(y; m=:default, trim=0) -> GPHResult

Geweke & Porter-Hudak (1983) log-periodogram estimator of the fractional integration
order d. Regresses log I(λ_j) on the regressor uⱼ = −log(4 sin²(λ_j/2)) over the first
`m` Fourier frequencies (default m = ⌊√T⌋), optionally trimming the first `trim`
frequencies. d̂ is the slope; asymptotic SE = π/√(6 Σ(uⱼ−ū)²).

Tests H₀: d = 0 (no long memory) with a two-sided normal z.
"""
function gph_test(y::AbstractVector{T}; m::Union{Symbol,Int}=:default, trim::Int=0) where {T<:AbstractFloat}
    yv = Vector{T}(y)
    n = length(yv)
    n < 8 && throw(ArgumentError("Series too short for GPH (n=$n)."))
    x = yv .- mean(yv)
    mm = m === :default ? floor(Int, sqrt(n)) : Int(m)
    (mm isa Int && mm >= 2) || throw(ArgumentError("m must be an integer ≥ 2, got $m"))
    lam, I = _periodogram(x)
    lo = trim + 1
    hi = trim + mm
    hi > length(I) && (hi = length(I))
    lo >= hi && throw(ArgumentError("Too few frequencies after trimming (m=$mm, trim=$trim)."))
    λ = @view lam[lo:hi]
    Ij = @view I[lo:hi]
    u = -log.(4 .* sin.(λ ./ 2) .^ 2)
    yr = log.(Ij)
    ubar = mean(u)
    Sxx = sum(abs2, u .- ubar)
    dhat = sum((u .- ubar) .* (yr .- mean(yr))) / Sxx
    se = T(π) / sqrt(6 * Sxx)
    tstat = dhat / se
    pval = 2 * (1 - cdf(Normal(), abs(tstat)))
    GPHResult{T}(dhat, se, tstat, T(pval), length(u), n, trim)
end

gph_test(y::AbstractVector; kwargs...) = gph_test(Float64.(y); kwargs...)

# =============================================================================
# Local Whittle estimator (Robinson 1995)
# =============================================================================

"""
    local_whittle(y; m=:default) -> LocalWhittleResult

Robinson (1995) semiparametric (local Whittle) estimator of d. Minimizes

    R(d) = log( m⁻¹ Σ_{j=1}^m λ_j^{2d} I(λ_j) ) − 2d·m⁻¹ Σ_{j=1}^m log λ_j

over the first `m` Fourier frequencies (default m = ⌊√T⌋) via a bounded 1-D search.
Asymptotic SE = 1/(2√m). Tests H₀: d = 0 with a two-sided normal z.
"""
function local_whittle(y::AbstractVector{T}; m::Union{Symbol,Int}=:default) where {T<:AbstractFloat}
    yv = Vector{T}(y)
    n = length(yv)
    n < 8 && throw(ArgumentError("Series too short for local Whittle (n=$n)."))
    x = yv .- mean(yv)
    mm = m === :default ? floor(Int, sqrt(n)) : Int(m)
    mm >= 2 || throw(ArgumentError("m must be ≥ 2, got $mm"))
    lam, I = _periodogram(x)
    mm > length(I) && (mm = length(I))
    λ = lam[1:mm]
    Ij = I[1:mm]
    loglam = log.(λ)
    mlogl = mean(loglam)
    R = d -> log(mean(λ .^ (2d) .* Ij)) - 2 * d * mlogl
    res = Optim.optimize(R, -0.49, 0.99)
    dhat = T(Optim.minimizer(res))
    se = one(T) / (2 * sqrt(T(mm)))
    tstat = dhat / se
    pval = 2 * (1 - cdf(Normal(), abs(tstat)))
    LocalWhittleResult{T}(dhat, se, tstat, T(pval), mm, n, T(Optim.minimum(res)))
end

local_whittle(y::AbstractVector; kwargs...) = local_whittle(Float64.(y); kwargs...)

# =============================================================================
# StatsAPI fit interface
# =============================================================================

StatsAPI.fit(::Type{ARFIMAModel}, y::AbstractVector, p::Int, q::Int; kwargs...) =
    estimate_arfima(y, p, q; kwargs...)
