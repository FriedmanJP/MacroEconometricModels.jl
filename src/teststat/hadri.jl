# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Hadri (2000) LM panel STATIONARITY test.

⚠ Unlike the other four first-generation panel unit-root tests (LLC / IPS /
Breitung / Fisher), Hadri's null hypothesis is *stationarity* — H0: all panels
are (trend-)stationary — and the standardized statistic is **RIGHT-tailed**:
very positive `Z` rejects. It is the panel analogue of the KPSS test.

References:
- Hadri, K. (2000). Testing for stationarity in heterogeneous panel data.
  Econometrics Journal, 3(2), 148-161.
"""

# =============================================================================
# Hadri LM test
# =============================================================================

"""
    hadri_test(X::AbstractMatrix{T}; deterministic=:constant, hetero=true,
               cs_demean=false) -> HadriResult{T}

Hadri (2000) LM panel stationarity test. `X` is `T×N`; a `PanelData` method is
also provided. Follows the Stata `xtunitroot hadri` Methods and Formulas.

For each unit, regress `yᵢₜ` on a panel-specific intercept (`:constant`) or
intercept+trend (`:trend`), form partial sums `Sᵢₜ = Σ_{j≤t} ε̂ᵢⱼ`, and the LM
statistic `LMᵢ = T⁻² Σₜ Sᵢₜ² / σ̂ᵢ²`. With `hetero=true` (default) each unit uses
its own `σ̂ᵢ²`; otherwise a pooled `σ̂²` is used. Standardize
`Z = √N (LM̄ − ξ) / ζ ~ N(0,1)`, `(ξ, ζ²) = (1/6, 1/45)` (`:constant`) or
`(1/15, 11/6300)` (`:trend`). **Very positive `Z` rejects H0 (stationarity).**

# Keyword Arguments
- `deterministic`: `:constant` (default) or `:trend`
- `hetero`: per-unit `σ̂ᵢ²` (default, robust to variance heterogeneity) vs. pooled
- `cs_demean`: subtract the cross-sectional mean at each `t` (crude CSD mitigation)

# Example
```julia
X = cumsum(randn(60, 20); dims=1)    # random-walk (non-stationary) panel
result = hadri_test(X; deterministic=:constant)
result.pvalue          # small ⇒ reject stationarity (evidence of a unit root)
```

# References
- Hadri (2000). Econometrics Journal, 3(2), 148-161.
"""
function hadri_test(X::AbstractMatrix{T};
                    deterministic::Symbol=:constant,
                    hetero::Bool=true,
                    cs_demean::Bool=false) where {T<:AbstractFloat}
    deterministic in (:constant, :trend) || throw(ArgumentError(
        "Hadri deterministic must be :constant or :trend, got :$deterministic"))
    Xw = cs_demean ? _cs_demean(X) : X
    T_obs, N = size(Xw)
    T_obs < 10 && throw(ArgumentError("Time dimension T=$T_obs too small for Hadri"))
    N < 2 && throw(ArgumentError("Hadri needs at least N=2 panel units, got N=$N"))

    # Deterministic design D and df correction T'.
    D = deterministic == :constant ? reshape(ones(T, T_obs), T_obs, 1) :
        hcat(ones(T, T_obs), T.(1:T_obs))
    Tprime = deterministic == :constant ? T_obs - 1 : T_obs - 2
    DtD_inv = robust_inv(D'D)

    resids = Matrix{T}(undef, T_obs, N)               # ε̂_it per unit
    for i in 1:N
        y = @view Xw[:, i]
        b = DtD_inv * (D'y)
        resids[:, i] = y - D * b
    end

    # Pooled short-run variance (used when hetero=false).
    sig2_pool = sum(abs2, resids) / (N * Tprime)

    LM = Vector{T}(undef, N)
    for i in 1:N
        S = zero(T)
        ssum = zero(T)                                # Σ_t S_it²
        @inbounds for t in 1:T_obs
            S += resids[t, i]
            ssum += S * S
        end
        s2 = hetero ? (sum(abs2, @view resids[:, i]) / Tprime) : sig2_pool
        s2 = max(s2, T(1e-30))
        LM[i] = ssum / (T_obs^2 * s2)
    end

    LM_bar = mean(LM)
    xi = deterministic == :constant ? T(1//6) : T(1//15)
    zeta2 = deterministic == :constant ? T(1//45) : T(11//6300)
    zeta = sqrt(zeta2)
    Z = sqrt(T(N)) * (LM_bar - xi) / zeta
    pval = T(ccdf(Normal(), Z))                       # RIGHT-tailed stationarity null

    HadriResult{T}(Z, pval, LM_bar, xi, zeta, hetero, deterministic, T_obs, N)
end

hadri_test(X::AbstractMatrix; kwargs...) = hadri_test(Float64.(X); kwargs...)
hadri_test(pd::PanelData; kwargs...) = hadri_test(_panel_to_matrix(pd); kwargs...)
