# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Breitung (2000) first-generation panel unit root test (unit-root null).

Unlike LLC, Breitung's pooled `λ` statistic uses a variance-ratio construction
that is *bias-free by design*, so it needs no finite-sample moment table — it is
asymptotically N(0,1) under the null with very negative values rejecting. The
trend case uses the forward-orthogonal-deviations / detrending transformation of
Breitung (2000).

The public name is `breitung_panel_test` (result `BreitungPanelResult`) to avoid
collision with the unrelated Breitung-Eickmeier factor break test
([`factor_break_test`](@ref)) in `src/teststat/factor_break.jl`.

References:
- Breitung, J. (2000). The local power of some unit root tests for panel data.
  In B. Baltagi (Ed.), Advances in Econometrics, Vol. 15, 161-178. JAI Press.
"""

# =============================================================================
# Breitung test
# =============================================================================

"""
    breitung_panel_test(X::AbstractMatrix{T}; deterministic=:constant, lags=0,
                        cs_demean=false) -> BreitungPanelResult{T}

Breitung (2000) panel unit root test. `X` is `T×N`; a `PanelData` method is also
provided. Follows the Stata `xtunitroot breitung` Methods and Formulas.

For `deterministic ∈ (:none, :constant)` the statistic is
`λ = [Σᵢ Σₜ yℓᵢₜ Δyᵢₜ / σᵢ²] / √(Σᵢ Σₜ (yℓᵢₜ)² / σᵢ²)`, where `yℓᵢₜ = y_{i,t-1}`
(`:none`) or `y_{i,t-1} − y_{i,p+1}` (`:constant`) and `σᵢ² = (T−p−2)⁻¹ Σ (Δyᵢₜ)²`.
For `deterministic = :trend` the differences are forward-orthogonalized (Helmert)
and the levels detrended before pooling. `λ ~ N(0,1)`; very negative values reject
H0 (panel unit root).

# Keyword Arguments
- `deterministic`: `:none`, `:constant` (default), or `:trend`
- `lags`: prewhitening lag order `p` (default `0`; AR(1) assumption). With `p>0`,
  `Δy` and the level are pre-filtered on `Δy_{t-1},…,Δy_{t-p}` to whiten serial
  correlation.
- `cs_demean`: subtract the cross-sectional mean at each `t` (crude CSD mitigation;
  prefer [`pesaran_cips_test`](@ref) for genuine CSD)

# Example
```julia
X = cumsum(randn(60, 20); dims=1)
result = breitung_panel_test(X; deterministic=:constant)
result.pvalue          # large ⇒ fail to reject the panel unit root
```

# References
- Breitung (2000). Advances in Econometrics, 15, 161-178.
"""
function breitung_panel_test(X::AbstractMatrix{T};
                             deterministic::Symbol=:constant,
                             lags::Int=0,
                             cs_demean::Bool=false) where {T<:AbstractFloat}
    deterministic in (:none, :constant, :trend) || throw(ArgumentError(
        "deterministic must be :none, :constant, or :trend, got :$deterministic"))
    lags < 0 && throw(ArgumentError("lags must be ≥ 0, got $lags"))
    Xw = cs_demean ? _cs_demean(X) : X
    T_obs, N = size(Xw)
    p = lags
    T_obs - p - 2 < 2 && throw(ArgumentError(
        "Time dimension T=$T_obs too small for Breitung with p=$p lags"))
    N < 2 && throw(ArgumentError("Breitung needs at least N=2 units, got N=$N"))

    num = zero(T)      # Σᵢ Σ yℓ·Δy/σ²
    den = zero(T)      # Σᵢ Σ (yℓ)²/σ²
    used = 0
    for i in 1:N
        y = @view Xw[:, i]
        contrib = deterministic == :trend ?
            _breitung_trend_unit(y, p) : _breitung_notrend_unit(y, p, deterministic)
        contrib === nothing && continue
        n, d = contrib
        num += n
        den += d
        used += 1
    end
    used < 2 && throw(ArgumentError("Breitung: fewer than 2 usable panel units"))
    den <= zero(T) && throw(ArgumentError("Breitung: degenerate denominator"))

    lambda = num / sqrt(den)
    pval = T(cdf(Normal(), lambda))         # left-tailed unit-root null

    BreitungPanelResult{T}(lambda, pval, p, deterministic, T_obs, N)
end

breitung_panel_test(X::AbstractMatrix; kwargs...) = breitung_panel_test(Float64.(X); kwargs...)
breitung_panel_test(pd::PanelData; kwargs...) = breitung_panel_test(_panel_to_matrix(pd); kwargs...)

# --- Breitung without trend (:none / :constant) -----------------------------
# Returns (numerator_i, denominator_i) or `nothing` if the unit is unusable.
function _breitung_notrend_unit(y::AbstractVector{T}, p::Int, deterministic::Symbol) where {T<:AbstractFloat}
    Tn = length(y)
    ns = Tn - p - 1                                   # rows t = p+2 .. Tn
    ns < 2 && return nothing
    dy_eff = Vector{T}(undef, ns)                     # Δy_it
    ylev = Vector{T}(undef, ns)                       # yℓ_it
    base = deterministic == :constant ? y[p+1] : zero(T)
    for s in 1:ns
        t = p + 1 + s
        dy_eff[s] = y[t] - y[t-1]
        ylev[s] = y[t-1] - base
    end
    if p > 0
        # Prewhiten Δy and the level on Δy_{t-1},…,Δy_{t-p} (no intercept).
        Z = Matrix{T}(undef, ns, p)
        for s in 1:ns
            t = p + 1 + s
            for j in 1:p
                Z[s, j] = y[t-j] - y[t-j-1]
            end
        end
        ZtZ_inv = robust_inv(Z'Z)
        dy_eff = dy_eff - Z * (ZtZ_inv * (Z'dy_eff))
        ylev = ylev - Z * (ZtZ_inv * (Z'ylev))
    end
    sig2 = dot(dy_eff, dy_eff) / (Tn - p - 2)
    sig2 <= zero(T) && return nothing
    num = dot(ylev, dy_eff) / sig2
    d = dot(ylev, ylev) / sig2
    (num, d)
end

# --- Breitung with trend (forward orthogonal deviations) ---------------------
function _breitung_trend_unit(y::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    Tn = length(y)
    ns = Tn - p - 1                                   # s = 1 .. T-p-1
    ns < 3 && return nothing
    # Prewhiten on a constant and Δy_{t-1..t-p}; du = residualized Δy, ul = residualized level.
    du = Vector{T}(undef, ns)
    ul = Vector{T}(undef, ns)
    for s in 1:ns
        t = p + 1 + s
        du[s] = y[t] - y[t-1]                          # Δy_is
        ul[s] = y[t-1]                                 # y_{i,s-1}
    end
    if p > 0
        Z = Matrix{T}(undef, ns, p)
        for s in 1:ns
            t = p + 1 + s
            for j in 1:p
                Z[s, j] = y[t-j] - y[t-j-1]            # Δy_{s-j}
            end
        end
        # Regress Δy on [const, Δy lags]; du = residual w.r.t. lag terms only (keep drift).
        C = hcat(ones(T, ns), Z)
        b = robust_inv(C'C) * (C'du)
        du = du - Z * b[2:end]                         # subtract only lag effects
        # Level: subtract lagged-level effects Σ α̂_ij y_{s-j-1}.
        Zl = Matrix{T}(undef, ns, p)
        for s in 1:ns
            t = p + 1 + s
            for j in 1:p
                Zl[s, j] = y[t-1-j]                     # y_{s-j-1}
            end
        end
        ul = ul - Zl * b[2:end]
    end
    du_bar = mean(du)
    sig2 = zero(T)
    for s in 1:ns
        sig2 += (du[s] - du_bar) * du[s]
    end
    sig2 /= (Tn - p - 2)
    sig2 <= zero(T) && return nothing
    # Forward orthogonal deviation of Δu, and detrended level vℓ. Both sums run
    # s = 1..ns; at s=ns the Helmert factor is 0 (Δv=0) but vℓ still enters the
    # denominator, so include it (guarding the 0/0 in the mean of an empty tail).
    num = zero(T)
    d = zero(T)
    for s in 1:ns
        dv = zero(T)
        if s < ns
            fut = zero(T)
            @inbounds for j in (s+1):ns
                fut += du[j]
            end
            dv = sqrt(T(ns - s) / T(ns - s + 1)) * (du[s] - fut / (ns - s))
        end
        vl = ul[s] - ul[1] - ns * du_bar               # (T-p-1)·Δū = ns·du_bar
        num += vl * dv / sig2
        d += vl^2 / sig2
    end
    (num, d)
end
