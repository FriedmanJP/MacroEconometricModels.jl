# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Long-Run Variance Toolkit
# =============================================================================
#
# Public API on top of the HAC internals in `core/covariance.jl`:
#
#   lrvar(U; kernel, bandwidth, prewhiten, demean)   -> Ω  (two-sided LRV, scalar or matrix)
#   lrcov(U; ...)                                     -> Ω  (two-sided long-run covariance)
#   lrcov_oneside(U; ...)                             -> Λ  (one-sided, Σ_{j≥0} Γ_j)
#   varhac(U; ic, max_lag, demean)                    -> Ω  (Den Haan–Levin 1997, parametric)
#
# For mean-zero (after `demean`) U (T×k) with Γ_j = T⁻¹ Σ_t U_t U_{t−j}' and kernel k(·):
#   two-sided  Ω = Γ₀ + Σ_{j≥1} k(j/b) (Γ_j + Γ_j')
#   one-sided  Λ = Γ₀ + Σ_{j≥1} k(j/b)  Γ_j          (no transpose — the FMOLS/CCR object)
# so Ω = Λ + Λ' − Γ₀ by construction.
#
# The kernel-weighted autocovariance accumulation reuses `_hac_meat` (the same code path
# `newey_west` uses; `lrcov` divides its un-normalized meat by T). Bandwidth selectors
# `:andrews` (Andrews 1991 plug-in, `optimal_bandwidth_nw`) and `:nw94` (Newey–West 1994,
# `optimal_bandwidth_nw94`) and prewhitening (`_prewhiten_moments`, Andrews–Monahan 1992
# with a 0.97 modulus cap) are the internals from issues #151/#152/#153.
#
# References:
#   Andrews (1991, Econometrica 59:817–858) — plug-in bandwidth
#   Newey & West (1994, Rev. Econ. Stud. 61:631–653) — automatic bandwidth
#   Andrews & Monahan (1992, Econometrica 60:953–966) — prewhitening
#   Den Haan & Levin (1997) — VARHAC

"""
    _lrv_kernel(kernel::Symbol) -> Symbol

Normalize a public kernel symbol to the internal `kernel_weight` name. Accepts the public
short alias `:qs` for the quadratic-spectral kernel in addition to the internal names.
"""
function _lrv_kernel(kernel::Symbol)
    kernel === :qs && return :quadratic_spectral
    kernel ∈ (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning) && return kernel
    throw(ArgumentError("kernel must be :bartlett, :parzen, :qs (:quadratic_spectral), or :tukey_hanning; got :$kernel"))
end

"""
    _resolve_bandwidth(Ud, kernel_internal, bandwidth; prewhiten=false) -> Int

Resolve the `bandwidth` argument of the long-run-variance API to an integer truncation lag.

- a real `bandwidth` is used as-is (rounded to the nearest non-negative integer truncation lag);
- `:andrews` → Andrews (1991) AR(1) plug-in (`optimal_bandwidth_nw`);
- `:nw94`    → Newey–West (1994) automatic bandwidth (`optimal_bandwidth_nw94`).
"""
function _resolve_bandwidth(Ud::AbstractMatrix{T}, kernel_internal::Symbol, bandwidth;
                            prewhiten::Bool=false) where {T<:AbstractFloat}
    if bandwidth isa Symbol
        if bandwidth === :andrews
            return optimal_bandwidth_nw(Ud; kernel=kernel_internal)
        elseif bandwidth === :nw94
            return optimal_bandwidth_nw94(Ud; kernel=kernel_internal, prewhiten=prewhiten)
        else
            throw(ArgumentError("bandwidth Symbol must be :andrews or :nw94; got :$bandwidth"))
        end
    else
        bandwidth < 0 && throw(ArgumentError("bandwidth must be ≥ 0; got $bandwidth"))
        return round(Int, bandwidth)
    end
end

"""
    _lrv_accumulate(Ud, bw, kernel_internal) -> (Ω, Λ, Γ0)

Compute the two-sided long-run covariance `Ω`, one-sided `Λ = Σ_{j≥0} Γ_j`, and lag-0 `Γ0`
from the (already demeaned) data `Ud` (T×k), sharing the `_hac_meat` accumulation with
`newey_west`. All three use `Γ_j = T⁻¹ Σ_t U_t U_{t−j}'` (division by `T`, matching
`long_run_variance`/`long_run_covariance`).
"""
function _lrv_accumulate(Ud::AbstractMatrix{T}, bw::Int, kernel_internal::Symbol) where {T<:AbstractFloat}
    n, k = size(Ud)
    Γ0 = (Ud' * Ud) / n
    # Two-sided Ω via the shared HAC meat (Γ₀ + Σ w(Γ_j + Γ_j')), normalized by T.
    Ω = _hac_meat(Ud, bw, kernel_internal) / n
    # One-sided Λ = Γ₀ + Σ w Γ_j (no transpose).
    Λ = Matrix{T}(Γ0)
    jmax = kernel_internal == :quadratic_spectral ? (n - 1) : bw
    @inbounds for j in 1:jmax
        w = kernel_weight(j, bw, kernel_internal, T)
        w == 0 && continue
        Γj = (@view(Ud[(j+1):n, :])' * @view(Ud[1:(n-j), :])) / n  # k×k
        Λ .+= w * Γj
    end
    return Ω, Λ, Matrix{T}(Γ0)
end

# --- prewhitening recolor ----------------------------------------------------
# Andrews–Monahan (1992): fit U_t = A U_{t-1} + e_t (0.97 modulus cap via
# `_prewhiten_moments`), estimate the residual LRV, recolor with D = (I − Â)⁻¹.
#   two-sided:  Ω = D Ω̂_e D'
#   one-sided:  under (approximately) white residuals the exact-VAR(1) recolor is
#               Λ = D Γ̂₀^{U,raw} + D (Λ̂_e − Γ̂₀^e) D'  (reduces to (I−A)⁻¹Γ₀ when e is white).

"""
    _lrv_prewhitened(Ud, bw, kernel_internal) -> (Ω, Λ, ok)

Prewhitened long-run covariance. Returns `(Ω, Λ, true)` on success or `(_, _, false)` when the
fitted moment VAR(1) is non-stable (spectral radius ≥ 0.97), so the caller falls back to the
raw-kernel estimate. `Λ` here is the one-sided recolor (see note above).
"""
function _lrv_prewhitened(Ud::AbstractMatrix{T}, bw::Int, kernel_internal::Symbol) where {T<:AbstractFloat}
    k = size(Ud, 2)
    Ehat, A = _prewhiten_moments(Ud)
    Ehat === nothing && return (zeros(T, k, k), zeros(T, k, k), false)
    Ω_e, Λ_e, Γ0_e = _lrv_accumulate(Ehat, bw, kernel_internal)
    D = robust_inv(Matrix{T}(I, k, k) - A)
    Ω = D * Ω_e * D'
    Ω = Matrix{T}((Ω + Ω') / 2)
    Γ0_raw = (Ud' * Ud) / size(Ud, 1)
    Λ = D * Matrix{T}(Γ0_raw) + D * (Λ_e - Γ0_e) * D'
    return (Ω, Λ, true)
end

# =============================================================================
# Public API
# =============================================================================

"""
    lrvar(U::AbstractVecOrMat; kernel=:bartlett, bandwidth=:andrews,
          prewhiten=false, demean=true)

Kernel (HAC) estimate of the two-sided **long-run variance/covariance** Ω of a mean-zero
(after `demean`) series. Returns a scalar for a vector `U` and a `k×k` matrix for a `T×k`
matrix `U`.

    Ω = Γ₀ + Σ_{j≥1} k(j/b) (Γ_j + Γ_j'),   Γ_j = T⁻¹ Σ_t U_t U_{t−j}'

# Arguments
- `kernel`: `:bartlett` (Newey–West), `:parzen`, `:qs` (quadratic-spectral), `:tukey_hanning`.
- `bandwidth`: `:andrews` (Andrews 1991 plug-in), `:nw94` (Newey–West 1994), or a real
  truncation lag used as-is.
- `prewhiten`: Andrews–Monahan (1992) VAR(1) prewhitening with recoloring and a 0.97 modulus
  cap (falls back to no prewhitening, with a warning, when the moment VAR(1) is near unit-root).
- `demean`: subtract the sample mean first (default `true`).

# References
Andrews (1991); Newey & West (1994); Andrews & Monahan (1992).
"""
function lrvar(U::AbstractMatrix; kernel::Symbol=:bartlett, bandwidth=:andrews,
               prewhiten::Bool=false, demean::Bool=true)
    return lrcov(U; kernel=kernel, bandwidth=bandwidth, prewhiten=prewhiten, demean=demean)
end

function lrvar(u::AbstractVector; kernel::Symbol=:bartlett, bandwidth=:andrews,
               prewhiten::Bool=false, demean::Bool=true)
    Ω = lrcov(reshape(collect(u), :, 1); kernel=kernel, bandwidth=bandwidth,
              prewhiten=prewhiten, demean=demean)
    return max(Ω[1, 1], zero(eltype(Ω)))
end

"""
    lrcov(U::AbstractMatrix; kernel=:bartlett, bandwidth=:andrews,
          prewhiten=false, demean=true) -> Matrix

Two-sided long-run covariance Ω (a `k×k` matrix), PSD-projected. Matrix analog of `lrvar`;
identical to `long_run_covariance` on the non-prewhitened path. See [`lrvar`](@ref).
"""
function lrcov(U::AbstractMatrix; kernel::Symbol=:bartlett, bandwidth=:andrews,
               prewhiten::Bool=false, demean::Bool=true)
    ki = _lrv_kernel(kernel)
    Uf = float.(U)
    T = eltype(Uf)
    Ud = demean ? Uf .- mean(Uf, dims=1) : Uf
    n = size(Ud, 1)
    n < 2 && return Matrix{T}(cov(Ud))
    bw = _resolve_bandwidth(Ud, ki, bandwidth; prewhiten=prewhiten)
    if prewhiten
        Ω, _, ok = _lrv_prewhitened(Ud, bw, ki)
        ok || (@warn "Andrews–Monahan prewhitening skipped: moment VAR(1) is non-stable (near unit root); falling back to no prewhitening." maxlog=1)
        ok && return _psd_project(Ω)
    end
    Ω, _, _ = _lrv_accumulate(Ud, bw, ki)
    return _psd_project(Ω)
end

"""
    lrcov_oneside(U::AbstractMatrix; kernel=:bartlett, bandwidth=:andrews,
                  prewhiten=false, demean=true) -> Matrix

One-sided long-run covariance Λ = Σ_{j≥0} Γ_j (a `k×k`, generally **non-symmetric** matrix).
This is the object fully-modified OLS / CCR and panel cointegration consume; getting the
transpose wrong (using `Γ_j + Γ_j'`) silently breaks FMOLS. Satisfies Ω = Λ + Λ' − Γ₀ with
the [`lrcov`](@ref) two-sided estimate on the non-prewhitened path.

See [`lrvar`](@ref) for keyword semantics.
"""
function lrcov_oneside(U::AbstractMatrix; kernel::Symbol=:bartlett, bandwidth=:andrews,
                       prewhiten::Bool=false, demean::Bool=true)
    ki = _lrv_kernel(kernel)
    Uf = float.(U)
    T = eltype(Uf)
    Ud = demean ? Uf .- mean(Uf, dims=1) : Uf
    n = size(Ud, 1)
    n < 2 && return Matrix{T}(cov(Ud))
    bw = _resolve_bandwidth(Ud, ki, bandwidth; prewhiten=prewhiten)
    if prewhiten
        _, Λ, ok = _lrv_prewhitened(Ud, bw, ki)
        ok || (@warn "Andrews–Monahan prewhitening skipped: moment VAR(1) is non-stable (near unit root); falling back to no prewhitening." maxlog=1)
        ok && return Λ
    end
    _, Λ, _ = _lrv_accumulate(Ud, bw, ki)
    return Λ
end

# =============================================================================
# VARHAC (Den Haan–Levin 1997)
# =============================================================================

"""
    varhac(U::AbstractVecOrMat; ic=:aic, max_lag=:auto, demean=true) -> Matrix

Parametric **VARHAC** long-run covariance (Den Haan & Levin 1997). Fits a reduced-form VAR of
order `p` selected by information criterion (up to `⌊T^{1/3}⌋` lags) and reads the
zero-frequency spectral density off the fitted filter:

    Ω = B̂(1)⁻¹ Σ̂ B̂(1)⁻ᵀ,   B̂(1) = I − Σ_{l=1}^p Â_l

where `Â_l` are the VAR coefficient matrices and `Σ̂` the residual covariance. No bandwidth or
kernel — a parametric alternative to kernel HAC.

# Arguments
- `ic`: order-selection criterion, `:aic` or `:bic`.
- `max_lag`: `:auto` = `⌊T^{1/3}⌋`, or a fixed integer maximum order (0 ⇒ white-noise Γ₀).
- `demean`: subtract the sample mean first (default `true`).

A common VAR order is selected across equations (the coefficient matrices `Â_l` are full),
which recovers the exact zero-frequency spectral density of a true VAR(p).

# References
Den Haan, W.J. and Levin, A. (1997), "A Practitioner's Guide to Robust Covariance Matrix
Estimation".
"""
function varhac(U::AbstractMatrix; ic::Symbol=:aic, max_lag=:auto, demean::Bool=true)
    ic ∈ (:aic, :bic) || throw(ArgumentError("ic must be :aic or :bic; got :$ic"))
    Uf = float.(U)
    T = eltype(Uf)
    Ud = demean ? Uf .- mean(Uf, dims=1) : Uf
    n, k = size(Ud)
    n < 3 && return Matrix{T}(cov(Ud))

    pmax = max_lag === :auto ? floor(Int, n^(one(T) / 3)) : Int(max_lag)
    pmax = max(pmax, 0)
    # Need n - pmax > k*pmax observations for a well-posed OLS at the largest order.
    while pmax > 0 && (n - pmax) ≤ k * pmax + 1
        pmax -= 1
    end

    # Select the order on a COMMON sample (drop the first pmax presample rows for every order).
    best_p = 0
    best_ic = T(Inf)
    for p in 0:pmax
        Σ = _var_ols_sigma(Ud, p, pmax)
        Σ === nothing && continue
        ld = _safe_logdet(Σ)
        isfinite(ld) || continue
        n_eff = n - pmax
        nparam = k * k * p
        penalty = ic === :aic ? T(2) : T(log(n_eff))
        crit = ld + penalty * nparam / n_eff
        if crit < best_ic
            best_ic = crit
            best_p = p
        end
    end

    # Refit at the selected order (full sample n - best_p) and recolor.
    A_list, Σ = _var_ols_fit(Ud, best_p)
    B1 = Matrix{T}(I, k, k)
    for Al in A_list
        B1 .-= Al
    end
    B1inv = robust_inv(B1)
    Ω = B1inv * Σ * B1inv'
    return _psd_project(Ω)
end

function varhac(u::AbstractVector; ic::Symbol=:aic, max_lag=:auto, demean::Bool=true)
    Ω = varhac(reshape(collect(u), :, 1); ic=ic, max_lag=max_lag, demean=demean)
    return max(Ω[1, 1], zero(eltype(Ω)))
end

"""
    _var_ols_design(Ud, p, presample) -> (Y, X)

Build the VAR(p) OLS response `Y` and stacked-lag regressor `X` on the common sample that
drops the first `presample` rows. `X` row `t` is `[U_{t-1}'  …  U_{t-p}']`; empty (0 columns)
for `p == 0`.
"""
function _var_ols_design(Ud::AbstractMatrix{T}, p::Int, presample::Int) where {T<:AbstractFloat}
    n, k = size(Ud)
    rows = (presample + 1):n
    Y = Matrix{T}(Ud[rows, :])
    n_eff = length(rows)
    X = Matrix{T}(undef, n_eff, k * p)
    @inbounds for l in 1:p
        X[:, ((l-1)*k+1):(l*k)] .= @view Ud[rows .- l, :]
    end
    return Y, X
end

# Residual covariance at order p on the common sample (presample=pmax); nothing if ill-posed.
function _var_ols_sigma(Ud::AbstractMatrix{T}, p::Int, pmax::Int) where {T<:AbstractFloat}
    Y, X = _var_ols_design(Ud, p, pmax)
    n_eff = size(Y, 1)
    n_eff ≤ size(X, 2) && return nothing
    resid = p == 0 ? Y : Y - X * (robust_inv(X' * X) * (X' * Y))
    return Matrix{T}((resid' * resid) / n_eff)
end

# Fit VAR(p) on its own maximal sample (presample=p); returns (A_list, Sigma).
function _var_ols_fit(Ud::AbstractMatrix{T}, p::Int) where {T<:AbstractFloat}
    n, k = size(Ud)
    if p == 0
        Σ = Matrix{T}((Ud' * Ud) / n)
        return (Vector{Matrix{T}}(), Σ)
    end
    Y, X = _var_ols_design(Ud, p, p)
    n_eff = size(Y, 1)
    B = robust_inv(X' * X) * (X' * Y)  # (k*p) × k, block l = A_l'
    resid = Y - X * B
    Σ = Matrix{T}((resid' * resid) / n_eff)
    A_list = [Matrix{T}(B[((l-1)*k+1):(l*k), :]') for l in 1:p]
    return (A_list, Σ)
end

function _safe_logdet(Σ::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = safe_cholesky(Matrix{T}((Σ + Σ') / 2); silent=true)
    d = diag(L)
    any(x -> !(x > zero(T)), d) && return T(Inf)
    return 2 * sum(log, d)
end
