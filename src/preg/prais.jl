# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Prais-Winsten AR(1) FGLS transformation for panel (TSCS) data  [EV-25, #433].

Companion to the Beck-Katz PCSE in `covariance.jl`: quasi-differences `(y, X)`
per unit to purge first-order serial correlation. The first observation of each
unit carries the `√(1 - ρ̂²)` weight (dropping it silently degrades the estimator
to Cochrane-Orcutt and changes every standard error).
"""

using LinearAlgebra, Statistics

# =============================================================================
# AR(1) coefficient from within-unit residual autocorrelation
# =============================================================================

"""
    _prais_winsten_rho(resid, groups, time_ids, unique_groups, ar1) -> Vector{T}

Estimate the AR(1) coefficient ρ̂ from within-unit residual autocorrelation,
using only consecutive (`t == t_prev + 1`) period pairs:

    ρ̂ = Σ_i Σ_{t≥2} ê_it ê_{i,t-1} / Σ_i Σ_{t≥2} ê²_{i,t-1}.

Returns a length-`N` vector (one entry per entry of `unique_groups`). For
`ar1=:common` all entries are the single pooled ρ̂; for `ar1=:panel_specific`
each entry is that unit's own ρ̂_i. Estimates are clamped to `(-0.999, 0.999)`.
"""
function _prais_winsten_rho(resid::AbstractVector{T}, groups::AbstractVector{Int},
                            time_ids::AbstractVector{Int},
                            unique_groups::AbstractVector{Int},
                            ar1::Symbol) where {T<:AbstractFloat}
    N = length(unique_groups)
    gmap = _group_index_map(groups)

    _clamp(r) = clamp(r, T(-0.999), T(0.999))

    if ar1 == :common
        num = zero(T)
        den = zero(T)
        for g in unique_groups
            idx = gmap[g]
            perm = sortperm(@view time_ids[idx])
            is = idx[perm]
            ts = time_ids[idx][perm]
            for j in 2:length(is)
                if ts[j] == ts[j-1] + 1
                    num += resid[is[j]] * resid[is[j-1]]
                    den += resid[is[j-1]]^2
                end
            end
        end
        rho = den > zero(T) ? _clamp(num / den) : zero(T)
        return fill(rho, N)
    elseif ar1 == :panel_specific
        rhos = zeros(T, N)
        for (gi, g) in enumerate(unique_groups)
            idx = gmap[g]
            perm = sortperm(@view time_ids[idx])
            is = idx[perm]
            ts = time_ids[idx][perm]
            num = zero(T)
            den = zero(T)
            for j in 2:length(is)
                if ts[j] == ts[j-1] + 1
                    num += resid[is[j]] * resid[is[j-1]]
                    den += resid[is[j-1]]^2
                end
            end
            rhos[gi] = den > zero(T) ? _clamp(num / den) : zero(T)
        end
        return rhos
    else
        throw(ArgumentError("ar1 must be :none, :common, or :panel_specific; got :$ar1"))
    end
end

# =============================================================================
# Prais-Winsten quasi-difference transform
# =============================================================================

"""
    _prais_winsten_transform(y, X, groups, time_ids, resid; ar1=:common)
        -> (y_pw, X_pw, rho)

Prais-Winsten quasi-difference of `(y, X)` per unit, given first-pass residuals
`resid` (used to estimate ρ̂). Within each unit, in time order:

- **first observation** (and the first of any post-gap segment): scaled by
  `√(1 - ρ̂²)` — the Prais-Winsten weight that keeps the first observation in the
  GLS sample; omitting it is Cochrane-Orcutt.
- **consecutive observations** (`t == t_prev + 1`): quasi-differenced
  `x_it - ρ̂ x_{i,t-1}` (dependent and every regressor column).

`ar1=:common` uses one pooled ρ̂ (returned as a scalar); `ar1=:panel_specific`
uses per-unit ρ̂_i (returned as a `Vector{T}` aligned with `sort(unique(groups))`).
Point estimates from a subsequent regression on `(y_pw, X_pw)` are the FGLS
estimates. Order in the pipeline: **PW transform → estimate FE/OLS → PCSE**.
"""
function _prais_winsten_transform(y::AbstractVector{T}, X::AbstractMatrix{T},
                                  groups::AbstractVector{Int},
                                  time_ids::AbstractVector{Int},
                                  resid::AbstractVector{T};
                                  ar1::Symbol=:common) where {T<:AbstractFloat}
    unique_groups = sort(unique(groups))
    rho_vec = _prais_winsten_rho(resid, groups, time_ids, unique_groups, ar1)
    rho_pos = Dict(g => rho_vec[i] for (i, g) in enumerate(unique_groups))

    n, k = size(X)
    y_pw = Vector{T}(copy(y))
    X_pw = Matrix{T}(copy(X))
    gmap = _group_index_map(groups)

    for g in unique_groups
        idx = gmap[g]
        perm = sortperm(@view time_ids[idx])
        is = idx[perm]
        ts = time_ids[idx][perm]
        r = rho_pos[g]
        w = sqrt(max(one(T) - r^2, zero(T)))

        # First observation of the unit: Prais-Winsten √(1-ρ²) weight.
        y_pw[is[1]] = w * y[is[1]]
        @inbounds for p in 1:k
            X_pw[is[1], p] = w * X[is[1], p]
        end

        for j in 2:length(is)
            if ts[j] == ts[j-1] + 1
                # Consecutive: quasi-difference.
                y_pw[is[j]] = y[is[j]] - r * y[is[j-1]]
                @inbounds for p in 1:k
                    X_pw[is[j], p] = X[is[j], p] - r * X[is[j-1], p]
                end
            else
                # Time gap: treat as the first observation of a new segment.
                y_pw[is[j]] = w * y[is[j]]
                @inbounds for p in 1:k
                    X_pw[is[j], p] = w * X[is[j], p]
                end
            end
        end
    end

    rho_out = ar1 == :common ? rho_vec[1] : rho_vec
    return y_pw, X_pw, rho_out
end

# =============================================================================
# First-pass (untransformed) residuals for the ρ̂ estimate
# =============================================================================

"""
    _panel_first_pass_resid(y, X, groups, unique_groups) -> Vector{T}

Within-transformed OLS residuals of `y` on `X` (entity-demeaned, no intercept),
returned in the original observation order. These consistent first-pass residuals
feed the Prais-Winsten ρ̂ estimate.
"""
function _panel_first_pass_resid(y::AbstractVector{T}, X::AbstractMatrix{T},
                                 groups::AbstractVector{Int},
                                 unique_groups::AbstractVector{Int}) where {T<:AbstractFloat}
    y_dm, _ = _within_demean(y, groups, unique_groups)
    X_dm, _ = _within_demean_matrix(X, groups, unique_groups)
    XtXinv = robust_inv(X_dm' * X_dm)
    beta = XtXinv * (X_dm' * y_dm)
    return y_dm .- X_dm * beta
end
