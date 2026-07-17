# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Fisher-type (Maddala-Wu 1999 / Choi 2001) combination of per-unit Johansen
cointegration tests. Each unit's series are run through [`johansen_test`](@ref);
the trace (and max-eigenvalue) p-values at each rank hypothesis `r = 0,1,…` are
combined across the `N` units:

- Maddala-Wu `P_r = −2 Σᵢ ln p_{i,r} ~ χ²(2N)` — **upper-tailed** (large ⇒ reject
  `H0: rank ≤ r`);
- Choi inverse-normal `Z_r = N^{-1/2} Σᵢ Φ⁻¹(p_{i,r}) ~ N(0,1)` — lower-tailed.

With `N = 1` the combination reduces exactly to the single unit's `johansen_test`
p-values (`P_r = −2 ln p_{1,r}`, `P(χ²(2) > P_r) = p_{1,r}`).

References:
- Maddala, G. S. & Wu, S. (1999). OBES 61(S1), 631-652.
- Choi, I. (2001). Journal of International Money and Finance, 20(2), 249-272.
- Johansen, S. (1991). Econometrica, 59(6), 1551-1580.
"""

# =============================================================================
# fisher_johansen_test
# =============================================================================

"""
    fisher_johansen_test(pd::PanelData, ys::Symbol...; deterministic=:constant,
                         lags=1, combine=:mw) -> FisherJohansenResult

Fisher-type combination of per-unit Johansen trace and max-eigenvalue tests for a
panel of `n = length(ys)` I(1) series. Returns combined statistics for each rank
hypothesis `r = 0 … n-1`.

# Keyword Arguments
- `deterministic`: passed to [`johansen_test`](@ref) — `:none`, `:constant`
  (default), or `:trend`.
- `lags`: VAR lag order `p` (≥ 1) for each per-unit Johansen test (default 2).
- `combine`: `:mw` (Maddala-Wu χ², default) or `:choi` (inverse-normal Z) —
  selects the primary `statistics`/`pvalues`; both are always stored.

# Example
```julia
pd = xtset(df, :country, :year)
res = fisher_johansen_test(pd, :m, :y, :r; lags=2)
res.trace_pvalues        # combined trace p-value per rank r = 0,1,2
```

# References
- Maddala & Wu (1999); Choi (2001); Johansen (1991).
"""
function fisher_johansen_test(pd::PanelData{TT}, ys::Symbol...;
                              deterministic::Symbol=:constant,
                              lags::Int=2, combine::Symbol=:mw) where {TT}
    length(ys) >= 2 || throw(ArgumentError(
        "fisher_johansen_test needs at least 2 series, got $(length(ys))"))
    combine in (:mw, :choi) || throw(ArgumentError(
        "combine must be :mw or :choi, got :$combine"))
    lags >= 1 || throw(ArgumentError("lags must be ≥ 1, got $lags"))
    T = float(TT)
    ycols = Int[_panel_varindex(pd, y) for y in ys]
    n = length(ycols)
    N = pd.n_groups

    trace_pv = Matrix{T}(undef, N, n)                 # per-unit trace p-values
    max_pv = Matrix{T}(undef, N, n)
    for g in 1:N
        gd = group_data(pd, g)
        ord = sortperm(gd.time_index)
        Yg = T.(gd.data[ord, ycols])
        jr = johansen_test(Yg, lags; deterministic=deterministic)
        for r in 1:n
            trace_pv[g, r] = clamp(jr.trace_pvalues[r], T(1e-12), one(T) - T(1e-12))
            max_pv[g, r] = clamp(jr.max_eigen_pvalues[r], T(1e-12), one(T) - T(1e-12))
        end
    end

    trace_stat, trace_p = _fisher_combine(trace_pv, combine)
    max_stat, max_p = _fisher_combine(max_pv, combine)

    # Estimated rank: first r (0-based) whose combined trace test fails to reject.
    rank = n
    for r in 1:n
        if trace_p[r] > T(0.05)
            rank = r - 1
            break
        end
    end

    FisherJohansenResult{T}(collect(0:(n-1)), trace_stat, trace_p, max_stat, max_p,
                            trace_pv, max_pv, combine, deterministic, lags,
                            rank, N, n)
end

# Combine an N×n matrix of per-unit p-values column-by-column (per rank).
function _fisher_combine(pv::Matrix{T}, combine::Symbol) where {T<:AbstractFloat}
    N, n = size(pv)
    stat = Vector{T}(undef, n)
    p = Vector{T}(undef, n)
    for r in 1:n
        col = @view pv[:, r]
        if combine === :mw
            P = -2 * sum(log, col)
            stat[r] = P
            p[r] = T(ccdf(Chisq(2N), P))              # upper-tailed
        else # :choi inverse-normal Z
            Z = sum(x -> quantile(Normal(), x), col) / sqrt(T(N))
            stat[r] = Z
            p[r] = T(cdf(Normal(), Z))                # lower-tailed
        end
    end
    (stat, p)
end
