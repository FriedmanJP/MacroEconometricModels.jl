# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Shared helpers + top-level dispatcher for cointegrating regression
# =============================================================================

"""
    _cointreg_deter(T_int, trend, ::Type{T}) -> (D, names)

Build the `T_int×d` deterministic-regressor block for `trend ∈ (:none, :const, :linear)`
and its column names.
"""
function _cointreg_deter(n::Int, trend::Symbol, ::Type{T}) where {T<:AbstractFloat}
    if trend === :none
        return zeros(T, n, 0), String[]
    elseif trend === :const
        return ones(T, n, 1), ["const"]
    elseif trend === :linear
        D = Matrix{T}(undef, n, 2)
        @inbounds for t in 1:n
            D[t, 1] = one(T)
            D[t, 2] = T(t)
        end
        return D, ["const", "trend"]
    else
        throw(ArgumentError("trend must be :none, :const, or :linear; got :$trend"))
    end
end

"""
    _cointreg_xnames(k) -> Vector{String}

Names for the `k` stochastic regressors (`"x"` if `k==1`, else `"x1"…"xk"`).
"""
_cointreg_xnames(k::Int) = k == 1 ? ["x"] : ["x$(i)" for i in 1:k]

"""
    _cointreg_lrv(U; kernel, bandwidth) -> (Ω, Λ, Σ)

Long-run covariance pieces of the stacked `(u, Δx)` process `U` (no demeaning), reusing the
EV-12 toolkit: two-sided `Ω = lrcov(U)`, one-sided `Λ = lrcov_oneside(U) = Σ_{j≥0} Γ_j`, and
contemporaneous `Σ = Γ₀ = UᵀU/n`.
"""
function _cointreg_lrv(U::AbstractMatrix{T}; kernel::Symbol, bandwidth) where {T<:AbstractFloat}
    Ω = lrcov(U; kernel=kernel, bandwidth=bandwidth, demean=false)
    Λ = lrcov_oneside(U; kernel=kernel, bandwidth=bandwidth, demean=false)
    Σ = Matrix{T}((U' * U) ./ size(U, 1))
    return Matrix{T}(Ω), Matrix{T}(Λ), Σ
end

"""
    _resolved_bw(U, kernel, bandwidth) -> Float64

The integer truncation lag `lrcov`/`lrcov_oneside` resolve `bandwidth` to (for display /
storage). Mirrors `_resolve_bandwidth`.
"""
function _resolved_bw(U::AbstractMatrix{T}, kernel::Symbol, bandwidth) where {T<:AbstractFloat}
    ki = _lrv_kernel(kernel)
    return T(_resolve_bandwidth(float.(U), ki, bandwidth))
end

"""
    estimate_cointreg(y, X; method=:fmols, trend=:const, kernel=:bartlett,
                      bandwidth=:andrews, leads=:auto, lags=:auto, ic=:aic,
                      dols_se=:lrv) -> CointRegModel{T}

Estimate a single cointegrating vector for `y ~ D + X` where `X` is an `I(1)` regressor
matrix, correcting the OLS-on-levels regression for regressor endogeneity and error serial
correlation. Returns a [`CointRegModel`](@ref).

# Arguments
- `y::AbstractVector` — dependent variable (levels).
- `X::AbstractVecOrMat` — `I(1)` stochastic regressor(s); a vector is treated as one column.

# Keywords
- `method`: `:fmols` (Phillips–Hansen fully-modified OLS, default), `:ccr` (Park canonical
  cointegrating regression), or `:dols` (Saikkonen / Stock–Watson dynamic OLS).
- `trend`: deterministics appended to `X` — `:none`, `:const` (default), or `:linear`.
- `kernel`: HAC kernel for the long-run covariance (`:bartlett`, `:parzen`, `:qs`,
  `:tukey_hanning`); forwarded to the EV-12 `lrcov`/`lrcov_oneside` toolkit.
- `bandwidth`: `:andrews` (default), `:nw94`, or a fixed real truncation lag.
- `leads`, `lags`: DOLS only — number of leads/lags of `Δx` (`:auto` selects by `ic`).
- `ic`: DOLS lead/lag selection criterion, `:aic` (default) or `:bic`.
- `dols_se`: DOLS standard errors — `:lrv` (long-run-variance corrected OLS, default) or
  `:robust` (Newey–West HAC sandwich).

# Examples
```julia
y, x = ...                                   # cointegrated series
m = estimate_cointreg(y, x; method=:fmols)   # Phillips–Hansen FMOLS
report(m)
coef(m)                                      # [const, slope]
```
"""
function estimate_cointreg(y::AbstractVector, X::AbstractVecOrMat;
                           method::Symbol=:fmols, trend::Symbol=:const,
                           kernel::Symbol=:bartlett, bandwidth=:andrews,
                           leads=:auto, lags=:auto, ic::Symbol=:aic,
                           dols_se::Symbol=:lrv)
    method ∈ (:fmols, :ccr, :dols) ||
        throw(ArgumentError("method must be :fmols, :ccr, or :dols; got :$method"))
    T = float(eltype(y))
    yv = collect(T, y)
    Xm = X isa AbstractVector ? reshape(collect(T, X), :, 1) : Matrix{T}(X)
    size(Xm, 1) == length(yv) ||
        throw(DimensionMismatch("length(y)=$(length(yv)) must equal size(X,1)=$(size(Xm,1))"))
    n = length(yv)
    n > 5 || throw(ArgumentError("need at least 6 observations; got $n"))

    if method === :fmols
        return _estimate_fmols(yv, Xm, trend, kernel, bandwidth)
    elseif method === :ccr
        return _estimate_ccr(yv, Xm, trend, kernel, bandwidth)
    else
        return _estimate_dols(yv, Xm, trend, kernel, bandwidth, leads, lags, ic, dols_se)
    end
end

# =============================================================================
# FMOLS — Phillips & Hansen (1990)
# =============================================================================

function _estimate_fmols(y::Vector{T}, X::Matrix{T}, trend::Symbol,
                         kernel::Symbol, bandwidth) where {T<:AbstractFloat}
    n, k = size(X)
    D, dnames = _cointreg_deter(n, trend, T)
    d = size(D, 2)
    Z = hcat(D, X)                                   # T×(d+k) design in levels

    # Stage 1: OLS on levels.
    ZtZ = Z' * Z
    theta_ols = robust_inv(ZtZ) * (Z' * y)
    u_ols = y .- Z * theta_ols                       # length T

    # Stacked (u, Δx) process, aligned to t = 2:T.
    u_lag = u_ols[2:n]                               # drop first residual
    Xdelta = diff(X; dims=1)                         # (T-1)×k
    U = hcat(u_lag, Xdelta)                          # (T-1)×(1+k)

    Ω, Λ, Σ = _cointreg_lrv(U; kernel=kernel, bandwidth=bandwidth)
    bw = _resolved_bw(U, kernel, bandwidth)
    Δcr = Matrix{T}(Λ')                              # cointReg's one-sided Δ = Σ w Γ_j'

    # Partition into u (index 1) and v = Δx (indices 2:k+1).
    vidx = 2:(k + 1)
    Ω_uu = Ω[1, 1]
    Ω_uv = reshape(Ω[1, vidx], 1, k)
    Ω_vu = reshape(Ω[vidx, 1], k, 1)
    Ω_vv = Ω[vidx, vidx]
    Δ_vu = reshape(Δcr[vidx, 1], k, 1)
    Δ_vv = Δcr[vidx, vidx]

    Ωvv_inv = robust_inv(Ω_vv)
    Ωvv_inv_vu = Ωvv_inv * Ω_vu                      # k×1
    ω_uv = Ω_uu - (Ω_uv*Ωvv_inv_vu)[1, 1]            # conditional LR variance
    Δ_vuplus = Δ_vu .- Δ_vv * Ωvv_inv_vu             # k×1

    # Endogeneity-corrected regressand (aligned to t = 2:T).
    y_plus = y[2:n] .- vec(Xdelta * Ωvv_inv_vu)      # (T-1)
    Zfm = Z[2:n, :]
    Zfm2s = robust_inv(Symmetric(Zfm' * Zfm))
    # Serial-correlation correction only on the stochastic block: [0_d ; T·Δ⁺_vu].
    corr = vcat(zeros(T, d), T(n) .* vec(Δ_vuplus))  # (d+k)
    numerat = Zfm' * y_plus .- corr
    theta_fm = vec(Matrix{T}(Zfm2s) * numerat)       # (d+k)

    vcov = Matrix{T}(ω_uv .* Zfm2s)
    fitted = Z * theta_fm
    resid = y .- fitted
    varnames = vcat(dnames, _cointreg_xnames(k))

    return CointRegModel{T}(y, X, :fmols, trend, kernel, bw, theta_fm, vcov,
                            resid, fitted, varnames, n, 0, 0, Ω, Λ, Σ, ω_uv, d, k)
end
