# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Fisher-type (Maddala-Wu 1999 / Choi 2001) combination panel unit root test.

Combines the per-unit ADF (or Phillips-Perron) p-values — reusing the existing
MacKinnon response-surface p-value path — into panel statistics under the
unit-root null. Assumes cross-sectional independence.

References:
- Maddala, G. S., & Wu, S. (1999). A comparative study of unit root tests with
  panel data and a new simple test. Oxford Bulletin of Economics and
  Statistics, 61(S1), 631-652.
- Choi, I. (2001). Unit root tests for panel data. Journal of International
  Money and Finance, 20(2), 249-272.
"""

# =============================================================================
# Fisher-type test
# =============================================================================

"""
    fisher_panel_test(X::AbstractMatrix{T}; base=:adf, combine=:mw, lags=:auto,
                      deterministic=:constant, cs_demean=false) -> FisherPanelResult{T}

Fisher-type combination panel unit root test. `X` is `T×N`; a `PanelData` method
is also provided.

Runs a per-unit unit-root test (`base = :adf` → [`adf_test`](@ref); `:pp` →
[`pp_test`](@ref)), collects the p-values `pᵢ`, and combines them:

- Maddala-Wu `P = −2 Σ ln pᵢ ~ χ²(2N)` — **upper-tailed** (large ⇒ reject)
- Choi inverse-normal `Z = N^{-1/2} Σ Φ⁻¹(pᵢ) ~ N(0,1)` — lower-tailed
- Choi logit `L* = √k · Σ ln(pᵢ/(1−pᵢ)) ~ t(5N+4)`, `k = 3(5N+4)/(π²N(5N+2))`
- Choi modified `Pm = −N^{-1/2} Σ (ln pᵢ + 1) ~ N(0,1)` — upper-tailed

All four are stored in the result; `combine ∈ (:mw, :choi, :logit, :pm)` selects
which is the primary `statistic`/`pvalue`. H0: all panels have a unit root.

With `N == 1` this reduces exactly to the single-series `adf_test`/`pp_test`
p-value (`P = −2 ln p₁`, `p_P = P(χ²(2) > P) = p₁`).

# Keyword Arguments
- `base`: `:adf` (default) or `:pp`
- `combine`: primary combination (`:mw` default)
- `lags`: passed to the per-unit ADF test (`:auto`/`:aic`/`:bic`/`:hqic`/Int)
- `deterministic`: `:none`, `:constant` (default), or `:trend`
- `cs_demean`: subtract the cross-sectional mean at each `t` (crude CSD mitigation)

# Example
```julia
X = randn(60, 20)
result = fisher_panel_test(X; base=:adf, combine=:mw)
result.pvalue        # small ⇒ reject the panel unit root
```

# References
- Maddala & Wu (1999); Choi (2001).
"""
function fisher_panel_test(X::AbstractMatrix{T};
                           base::Symbol=:adf,
                           combine::Symbol=:mw,
                           lags::Union{Int,Symbol}=:auto,
                           deterministic::Symbol=:constant,
                           cs_demean::Bool=false) where {T<:AbstractFloat}
    base in (:adf, :pp) || throw(ArgumentError("base must be :adf or :pp, got :$base"))
    combine in (:mw, :choi, :logit, :pm) || throw(ArgumentError(
        "combine must be :mw, :choi, :logit, or :pm, got :$combine"))
    deterministic in (:none, :constant, :trend) || throw(ArgumentError(
        "deterministic must be :none, :constant, or :trend, got :$deterministic"))
    Xw = cs_demean ? _cs_demean(X) : X
    T_obs, N = size(Xw)
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))
    N < 1 && throw(ArgumentError("Fisher test needs N ≥ 1 unit, got N=$N"))

    pv = Vector{T}(undef, N)
    lags_adf = lags === :auto ? :aic : lags
    for i in 1:N
        y = collect(@view Xw[:, i])
        r = base === :adf ?
            adf_test(y; lags=lags_adf, regression=deterministic) :
            pp_test(y; regression=deterministic)
        # Guard the log/Φ⁻¹ transforms against exact 0/1 p-values.
        pv[i] = clamp(r.pvalue, T(1e-12), one(T) - T(1e-12))
    end

    # Maddala-Wu P ~ χ²(2N), upper-tailed.
    P = -2 * sum(log, pv)
    P_pval = T(ccdf(Chisq(2N), P))
    # Choi inverse-normal Z ~ N(0,1), lower-tailed.
    Z = sum(x -> quantile(Normal(), x), pv) / sqrt(T(N))
    Z_pval = T(cdf(Normal(), Z))
    # Choi logit L* ~ t(5N+4), lower-tailed.
    L = sum(x -> log(x / (1 - x)), pv)
    k = 3 * (5N + 4) / (T(pi)^2 * N * (5N + 2))
    Lstar = sqrt(k) * L
    Lstar_pval = T(cdf(TDist(5N + 4), Lstar))
    # Choi modified Pm ~ N(0,1), upper-tailed.
    Pm = -sum(x -> log(x) + 1, pv) / sqrt(T(N))
    Pm_pval = T(ccdf(Normal(), Pm))

    stat, pval = combine === :mw ? (P, P_pval) :
                 combine === :choi ? (Z, Z_pval) :
                 combine === :logit ? (Lstar, Lstar_pval) : (Pm, Pm_pval)

    FisherPanelResult{T}(T(stat), T(pval), P, P_pval, Z, Z_pval,
                         Lstar, Lstar_pval, Pm, Pm_pval,
                         pv, base, combine, T_obs, N)
end

fisher_panel_test(X::AbstractMatrix; kwargs...) = fisher_panel_test(Float64.(X); kwargs...)
fisher_panel_test(pd::PanelData; kwargs...) = fisher_panel_test(_panel_to_matrix(pd); kwargs...)
