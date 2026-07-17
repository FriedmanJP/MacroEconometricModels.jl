# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Robust (outlier-resistant) linear regression: Huber/bisquare M-estimation via IRLS and
Yohai MM-estimation (fast-S subsampling scale seed + high-efficiency bisquare M-step).
Mirrors R `MASS::rlm(method="M"/"MM")`. Reuses `robust_inv` and `estimate_reg`.
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# ρ / ψ / weight functions (Huber & Tukey bisquare)
# =============================================================================
#
# Convention: `u = r/ŝ` is the scaled residual. The IRLS weight is `w(u) = ψ(u)/u`.
# For Huber:    ψ(u) = clamp(u,-k,k),  ρ(u) = u²/2 (|u|≤k) else k|u|-k²/2.
# For bisquare: ψ(u) = u(1-(u/c)²)² (|u|≤c) else 0,
#               ρ(u) = (c²/6)(1-(1-(u/c)²)³) (|u|≤c) else c²/6.

@inline function _rw_huber(u::T, k::T) where {T<:AbstractFloat}
    au = abs(u)
    au <= k ? one(T) : k / au                    # weight ψ/u = min(1, k/|u|)
end
@inline _psi_huber(u::T, k::T) where {T<:AbstractFloat} = clamp(u, -k, k)
@inline _psip_huber(u::T, k::T) where {T<:AbstractFloat} = abs(u) <= k ? one(T) : zero(T)
@inline function _rho_huber(u::T, k::T) where {T<:AbstractFloat}
    au = abs(u)
    au <= k ? au^2 / 2 : k * au - k^2 / 2
end

@inline function _rw_bisq(u::T, c::T) where {T<:AbstractFloat}
    au = abs(u)
    au <= c ? (one(T) - (u / c)^2)^2 : zero(T)   # weight ψ/u = (1-(u/c)²)²
end
@inline function _psi_bisq(u::T, c::T) where {T<:AbstractFloat}
    abs(u) <= c ? u * (one(T) - (u / c)^2)^2 : zero(T)
end
@inline function _psip_bisq(u::T, c::T) where {T<:AbstractFloat}
    a = (u / c)^2
    abs(u) <= c ? (one(T) - a) * (one(T) - 5a) : zero(T)
end
@inline function _rho_bisq(u::T, c::T) where {T<:AbstractFloat}
    a = (u / c)^2
    abs(u) <= c ? (c^2 / 6) * (one(T) - (one(T) - a)^3) : c^2 / 6
end

# Normalized-MAD scale (÷0.6745 ⇒ Fisher-consistent at the normal), taken about zero
# exactly as `MASS::rlm` (`median(abs(resid))/0.6745`).
@inline function _mad_scale(r::AbstractVector{T}) where {T<:AbstractFloat}
    s = median(abs.(r)) / T(0.6745)
    s > zero(T) ? s : (m = mean(abs.(r)); m > zero(T) ? m : eps(T))
end

# Weighted least squares β = (X'WX)⁻¹X'Wy via the √W-transformed normal equations.
function _wls_beta(X::AbstractMatrix{T}, y::AbstractVector{T},
                   w::AbstractVector{T}) where {T<:AbstractFloat}
    sw = sqrt.(w)
    Xw = X .* sw
    yw = y .* sw
    robust_inv(Symmetric(Xw' * Xw)) * (Xw' * yw)
end

# S-estimator M-scale: solve (1/n)Σρ((rᵢ)/s) = b for the bisquare ρ with tuning `c`.
# Monotone fixed-point s ← s·√(mean(ρ(r/s))/b) (Salibian-Barrera & Yohai 2006).
function _m_scale(r::AbstractVector{T}, c::T, b::T;
                  tol::T=T(1e-9), maxit::Int=200) where {T<:AbstractFloat}
    s = _mad_scale(r)
    s <= zero(T) && return eps(T)
    for _ in 1:maxit
        m = mean(_rho_bisq(ri / s, c) for ri in r)
        s_new = s * sqrt(m / b)
        (isfinite(s_new) && s_new > zero(T)) || break
        abs(s_new - s) <= tol * s && (s = s_new; break)
        s = s_new
    end
    s
end

# =============================================================================
# S-estimator constant b (50%-breakdown bisquare, tuning c0 = 1.548)
# =============================================================================
#
# `b = E_Φ[ρ(Z; c0)]` chosen so the S-estimator is consistent at the normal AND has
# 50% breakdown (b/ρ(∞) = 0.5). Value matches `lqs`/`rlm` (numerically integrated
# once; ρ(∞) = c0²/6 = 0.399384, b/ρ(∞) = 0.49991). See the `Rscript` provenance in
# `test/reg/test_robust.jl`.
const _S_C0 = 1.548
const _S_B  = 0.1996572094

# =============================================================================
# Fast-S subsampling (initial high-breakdown S-estimate for MM)
# =============================================================================
#
# Salibian-Barrera & Yohai (2006): draw random p-subsets, refine each with a couple of
# concentration (I-)steps, keep the lowest-scale candidates, then iterate them to
# convergence. Returns (β̂_S, ŝ_S). Seeded `rng` ⇒ reproducible subsamples.
function _fast_s(X::AbstractMatrix{T}, y::AbstractVector{T};
                 rng::Random.AbstractRNG,
                 n_resample::Int=500, k_step::Int=2, n_best::Int=5,
                 refine_maxit::Int=200, refine_tol::T=T(1e-8)) where {T<:AbstractFloat}
    n, p = size(X)
    c0 = T(_S_C0); b = T(_S_B)

    best_scales = fill(T(Inf), n_best)
    best_betas  = [zeros(T, p) for _ in 1:n_best]
    best_scale = T(Inf); best_beta = zeros(T, p)

    scratch = collect(1:n)
    # Random p-subsets and concentration steps routinely hit rank-deficient weighted
    # designs (some elemental sets / heavily-trimmed reweightings are collinear); the
    # pseudo-inverse fallback handles them and such candidates simply lose on scale.
    # Suppress the expected near-singular `@warn` spam from `robust_inv`.
    _suppress_warnings() do
    for _ in 1:n_resample
        # random p-subset via partial Fisher-Yates on `scratch`
        @inbounds for i in 1:p
            j = i + (rand(rng, 0:(n - i)))
            scratch[i], scratch[j] = scratch[j], scratch[i]
        end
        idx = @view scratch[1:p]
        Xs = X[idx, :]; ys = y[idx]
        beta = try
            Xs \ ys
        catch
            continue
        end
        all(isfinite, beta) || continue

        # concentration (I-)steps
        for _s in 1:k_step
            r = y .- X * beta
            s = _m_scale(r, c0, b)
            u = r ./ s
            w = _rw_bisq.(u, c0)
            beta = _wls_beta(X, y, w)
            all(isfinite, beta) || break
        end
        all(isfinite, beta) || continue
        s = _m_scale(y .- X * beta, c0, b)

        # keep if it improves the worst retained candidate
        wmax, wpos = findmax(best_scales)
        if s < wmax
            best_scales[wpos] = s
            best_betas[wpos] = copy(beta)
        end
    end

    # fully refine the best candidates, track the global minimum scale
    for cand in 1:n_best
        isfinite(best_scales[cand]) || continue
        beta = best_betas[cand]
        for _ in 1:refine_maxit
            r = y .- X * beta
            s = _m_scale(r, c0, b)
            u = r ./ s
            w = _rw_bisq.(u, c0)
            beta_new = _wls_beta(X, y, w)
            all(isfinite, beta_new) || break
            delta = norm(beta_new - beta) / max(norm(beta), eps(T))
            beta = beta_new
            delta <= refine_tol && break
        end
        s = _m_scale(y .- X * beta, c0, b)
        if s < best_scale
            best_scale = s
            best_beta = beta
        end
    end
    end  # _suppress_warnings
    best_beta, best_scale
end

# =============================================================================
# Huber "Proposal 2" joint scale update
# =============================================================================
#
# s solves (1/(n-p))Σ ψ_H(rᵢ/s)² = κ, κ = E_Φ[ψ_H(Z)²] = E[min(Z²,k²)].
# Fixed point s² ← s²·(1/((n-p)κ))Σ ψ_H(rᵢ/s)².
function _kappa_huber(k::T) where {T<:AbstractFloat}
    Φk = cdf(Normal(), k); φk = pdf(Normal(), k)
    (2Φk - one(T)) - 2k * φk + k^2 * (2 * (one(T) - Φk))
end
function _prop2_scale(r::AbstractVector{T}, s0::T, k::T, n::Int, p::Int;
                      tol::T=T(1e-9), maxit::Int=200) where {T<:AbstractFloat}
    κ = _kappa_huber(k)
    s = s0 > zero(T) ? s0 : _mad_scale(r)
    denom = T(n - p) * κ
    for _ in 1:maxit
        acc = zero(T)
        for ri in r
            ψ = _psi_huber(ri / s, k)
            acc += ψ^2
        end
        s_new = s * sqrt(acc / denom)
        (isfinite(s_new) && s_new > zero(T)) || break
        abs(s_new - s) <= tol * s && (s = s_new; break)
        s = s_new
    end
    s
end

# =============================================================================
# estimate_robust — public entry point
# =============================================================================

"""
    estimate_robust(y, X; psi=:huber, method=:m, k=nothing, maxiter=50, tol=1e-6,
                    scale_update=:mad, rng=Random.default_rng(),
                    n_resample=500, varnames=nothing) -> RobustRegModel{T}

Fit an outlier-resistant linear regression by M- or MM-estimation.

`X` should include an intercept column. Mirrors R `MASS::rlm`.

# Arguments
- `y::AbstractVector`, `X::AbstractMatrix` — response and regressors (`n × k`).
- `psi::Symbol` — influence function: `:huber` (default) or `:bisquare` (Tukey biweight).
  Forced to `:bisquare` when `method = :mm`.
- `method::Symbol` — `:m` (M-estimation, default) or `:mm` (Yohai MM-estimation).
- `k::Union{Nothing,Real}` — ψ tuning constant. Defaults to `1.345` (Huber, 95% Gaussian
  efficiency) or `4.685` (bisquare, ≈95% efficiency). Do not change silently.
- `maxiter::Int`, `tol::Real` — IRLS iteration cap and residual-convergence tolerance.
- `scale_update::Symbol` — `:mad` (default; normalized MAD each iteration) or `:proposal2`
  (Huber Proposal-2 joint scale updating). M-estimation only.
- `rng::Random.AbstractRNG` — RNG seeding the MM fast-S subsampling (reproducible).
- `n_resample::Int` — number of fast-S random p-subsets (MM only).
- `varnames::Union{Nothing,Vector{String}}` — coefficient names.

# Estimation
M-estimation runs IRLS: at each step form scaled residuals `uᵢ = rᵢ/ŝ` (ŝ = normalized
MAD), weights `wᵢ = ψ(uᵢ)/uᵢ`, and update `β = (X'WX)⁻¹X'Wy`. MM-estimation seeds a
high-breakdown S-estimate of scale (bisquare, 50% breakdown, via fast-S subsampling) then
fixes that scale and re-estimates `β` by a high-efficiency bisquare M-step; `m.scale`
returns the S-scale.

Covariance is the Huber–Ronchetti sandwich
`V = ŝ²·[(1/n)Σψ(uᵢ)²] / [(1/n)Σψ'(uᵢ)]² · (X'X)⁻¹`.

# Returns
`RobustRegModel{T}`. The `weights` field carries the final ψ-weights — points with
`wᵢ ≈ 0` are downweighted outliers.

# Examples
```julia
using MacroEconometricModels
d = load_example(:stackloss)
y = d.data[:, 4]
X = hcat(ones(21), d.data[:, 1:3])
m = estimate_robust(y, X; psi=:huber, method=:m)
report(m)
```

# References
- Huber, P. J. (1964). *Annals of Mathematical Statistics* 35(1), 73-101.
- Yohai, V. J. (1987). *Annals of Statistics* 15(2), 642-656.
- Huber, P. J. & Ronchetti, E. M. (2009). *Robust Statistics*. 2nd ed. Wiley.
"""
function estimate_robust(y::AbstractVector{T}, X::AbstractMatrix{T};
                         psi::Symbol=:huber,
                         method::Symbol=:m,
                         k::Union{Nothing,Real}=nothing,
                         maxiter::Int=50,
                         tol::Real=1e-6,
                         scale_update::Symbol=:mad,
                         rng::Random.AbstractRNG=Random.default_rng(),
                         n_resample::Int=500,
                         varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    p = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > p || throw(ArgumentError("Need n > p (n=$n, p=$p)"))
    method in (:m, :mm) || throw(ArgumentError("method must be :m or :mm; got :$method"))
    psi in (:huber, :bisquare) ||
        throw(ArgumentError("psi must be :huber or :bisquare; got :$psi"))
    scale_update in (:mad, :proposal2) ||
        throw(ArgumentError("scale_update must be :mad or :proposal2; got :$scale_update"))

    Xm = Matrix{T}(X); yv = Vector{T}(y)
    vn = something(varnames, [all(==(one(T)), @view Xm[:, j]) ? "(Intercept)" : "x$j"
                              for j in 1:p])
    length(vn) == p || throw(ArgumentError("varnames must have length $p"))

    tolT = T(tol)

    if method == :mm
        # MM: forced bisquare, high-efficiency tuning; S-scale held fixed in the M-step.
        psi = :bisquare
        c = k === nothing ? T(4.685) : T(k)
        beta, s_scale = _fast_s(Xm, yv; rng=rng, n_resample=n_resample)
        beta = Vector{T}(beta)
        converged = false
        iters = 0
        r = yv .- Xm * beta
        for it in 1:maxiter
            iters = it
            u = r ./ s_scale
            w = _rw_bisq.(u, c)
            beta_new = _wls_beta(Xm, yv, w)
            r_new = yv .- Xm * beta_new
            delta = sqrt(sum(abs2, r_new .- r) / max(T(1e-20), sum(abs2, r)))
            beta = beta_new; r = r_new
            if delta <= tolT
                converged = true
                break
            end
        end
        scale = s_scale
        tuning = c
        u = r ./ scale
        weights = _rw_bisq.(u, c)
        psip = _psip_bisq.(u, c)
        psival = _psi_bisq.(u, c)
        rho_used = uu -> _rho_bisq(uu, c)
    else
        # M-estimation via IRLS, LS start (reuse estimate_reg), MAD (or Proposal-2) scale.
        beta = Vector{T}(estimate_reg(yv, Xm; cov_type=:ols).beta)
        c = if k !== nothing
            T(k)
        else
            psi == :huber ? T(1.345) : T(4.685)
        end
        rwfun = psi == :huber ? (uu -> _rw_huber(uu, c)) : (uu -> _rw_bisq(uu, c))
        converged = false
        iters = 0
        r = yv .- Xm * beta
        s = _mad_scale(r)
        for it in 1:maxiter
            iters = it
            s = scale_update == :proposal2 ? _prop2_scale(r, s, c, n, p) : _mad_scale(r)
            u = r ./ s
            w = rwfun.(u)
            beta_new = _wls_beta(Xm, yv, w)
            r_new = yv .- Xm * beta_new
            delta = sqrt(sum(abs2, r_new .- r) / max(T(1e-20), sum(abs2, r)))
            beta = beta_new; r = r_new
            if delta <= tolT
                converged = true
                break
            end
        end
        scale = scale_update == :proposal2 ? _prop2_scale(r, s, c, n, p) : _mad_scale(r)
        tuning = c
        u = r ./ scale
        if psi == :huber
            weights = _rw_huber.(u, c)
            psip = _psip_huber.(u, c)
            psival = _psi_huber.(u, c)
            rho_used = uu -> _rho_huber(uu, c)
        else
            weights = _rw_bisq.(u, c)
            psip = _psip_bisq.(u, c)
            psival = _psi_bisq.(u, c)
            rho_used = uu -> _rho_bisq(uu, c)
        end
    end

    fitted = Xm * beta
    resid = yv .- fitted

    # Huber–Ronchetti sandwich: V = ŝ²·[mean(ψ²)] / [mean(ψ')]² · (X'X)⁻¹.
    mean_psi2 = mean(abs2, psival)
    mean_psip = mean(psip)
    XtXinv = robust_inv(Symmetric(Xm' * Xm))
    factor = mean_psip > eps(T) ? scale^2 * mean_psi2 / mean_psip^2 : T(NaN)
    vcov_mat = Matrix{T}(factor .* XtXinv)

    # Robust R² = 1 − Σρ(rᵢ/ŝ)/Σρ((yᵢ−median y)/ŝ).
    ymed = median(yv)
    num = sum(rho_used(ri / scale) for ri in resid)
    den = sum(rho_used((yi - ymed) / scale) for yi in yv)
    robust_r2 = den > zero(T) ? one(T) - num / den : zero(T)

    RobustRegModel{T}(yv, Xm, Vector{T}(beta), vcov_mat, T(scale), Vector{T}(weights),
                      Vector{T}(resid), Vector{T}(fitted), psi, method, T(tuning),
                      T(robust_r2), vn, converged, iters)
end

# Convenience: estimate directly from a CrossSectionData response/regressor selection.
"""
    estimate_robust(d::CrossSectionData, y::Union{Symbol,String,Int},
                    xs::Vector; add_intercept=true, kwargs...) -> RobustRegModel

Fit robust regression from a [`CrossSectionData`](@ref) container, selecting the response
column `y` and regressor columns `xs` by name or index. An intercept column is prepended
when `add_intercept=true`.
"""
function estimate_robust(d::CrossSectionData{T}, y::Union{Symbol,String,Int},
                         xs::AbstractVector; add_intercept::Bool=true,
                         kwargs...) where {T<:AbstractFloat}
    _colidx(v::Int) = v
    _colidx(v) = (i = findfirst(==(String(v)), d.varnames);
                  i === nothing ? throw(ArgumentError("column $v not found")) : i)
    yi = _colidx(y)
    xis = Int[_colidx(v) for v in xs]
    yv = Vector{T}(d.data[:, yi])
    Xcore = Matrix{T}(d.data[:, xis])
    xnames = String[d.varnames[j] for j in xis]
    if add_intercept
        Xmat = hcat(ones(T, size(Xcore, 1)), Xcore)
        vn = vcat("(Intercept)", xnames)
    else
        Xmat = Xcore
        vn = xnames
    end
    estimate_robust(yv, Xmat; varnames=vn, kwargs...)
end
