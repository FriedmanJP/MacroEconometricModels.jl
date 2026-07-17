# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Arellano-Bond (1991) test for serial correlation in dynamic-panel GMM residuals.

The test is defined on the FIRST-DIFFERENCED residuals (for both :ab and :bb, matching
xtabond2). Under a correctly specified model the level idiosyncratic error ε is serially
uncorrelated, so Δε is MA(1): the AR(1) test rejects (E[Δε_t Δε_{t-1}] = −Var(ε) < 0)
while the AR(2) test does not (E[Δε_t Δε_{t-2}] = 0).
"""

using LinearAlgebra, Statistics, Distributions

"""
    _pvar_ar_stats(pd, depvar, indepvars, beta, p, V) -> (m1, p1, m2, p2)

Internal: Arellano-Bond m-statistics (orders 1 and 2). Rebuilds the FD residuals and FD
instruments per group with the SAME helpers the estimator uses (`_panel_lag_levels`,
`_panel_first_difference`, `_build_instruments_fd`) so the alignment matches; for :bb this
is the standard difference-based AB test on the FD moment block. `beta` is the single
equation's coefficient vector `[lag; exog…]`, `p` the level AR order, `V = coef_vcov[1]`.
Robust two-step variance (Arellano & Bond 1991, Appendix):
`v̂ = T1 − 2·eX·V·SXZ·A_N·Zee + eX·V·eX'`. Guards `v̂ ≤ 0 ⇒ (NaN, 1)`.
"""
function _pvar_ar_stats(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol},
                        beta::AbstractVector{T}, p::Int, V::Matrix{T}) where {T}
    dep_idx = findfirst(==(String(depvar)), pd.varnames)
    dep_idx === nothing && throw(ArgumentError("Dependent variable $depvar not found"))
    exog_idx = Int[findfirst(==(String(v)), pd.varnames) for v in indepvars]
    any(isnothing, exog_idx) && throw(ArgumentError("Some exogenous variables not found"))
    N = ngroups(pd)
    K = length(beta)

    function m_stat(lag::Int)
        es = Vector{Vector{T}}(); dXs = Matrix{T}[]; Zs = Matrix{T}[]; ws = Vector{Vector{T}}()
        for g in 1:N
            gd = group_data(pd, g)
            Y_g = Matrix{T}(gd.data[:, [dep_idx]])
            size(Y_g, 1) <= p + 1 && continue
            Y_eff, X_lag = _panel_lag_levels(Y_g, p)
            X_full = isempty(exog_idx) ? X_lag :
                     hcat(X_lag, Matrix{T}(gd.data[(p+1):end, exog_idx]))
            dY = _panel_first_difference(Y_eff)
            dX = _panel_first_difference(X_full)
            e = vec(dY) - dX * beta
            Z = _build_instruments_fd(Y_g, p, 1)
            Tc = min(length(e), size(Z, 1))
            Tc <= lag && continue
            e = e[end-Tc+1:end]
            dXc = dX[end-Tc+1:end, :]
            Zc = Z[end-Tc+1:end, :]
            w = zeros(T, Tc)
            @inbounds for t in (lag+1):Tc
                w[t] = e[t-lag]
            end
            push!(es, e); push!(dXs, dXc); push!(Zs, Zc); push!(ws, w)
        end
        isempty(es) && return (T(NaN), one(T))

        max_mz = maximum(size(Z, 2) for Z in Zs)
        r = zero(T); T1 = zero(T); eX = zeros(T, 1, K)
        SXZ = zeros(T, K, max_mz); Zee = zeros(T, max_mz); AN = zeros(T, max_mz, max_mz)
        for i in eachindex(es)
            e = es[i]; dXc = dXs[i]; Zc = Zs[i]; w = ws[i]
            if size(Zc, 2) < max_mz     # zero-pad narrower blocks (contributes nothing)
                Zc = hcat(Zc, zeros(T, size(Zc, 1), max_mz - size(Zc, 2)))
            end
            we = dot(w, e)
            r += we
            T1 += we^2
            eX .+= w' * dXc
            Ze = Zc' * e
            SXZ .+= dXc' * Zc
            Zee .+= Ze .* we
            AN .+= Ze * Ze'
        end
        A_N = Matrix(robust_inv(Hermitian((AN ./ N + (AN ./ N)') / 2)))
        term2 = (eX * V * SXZ * A_N * Zee)[1]
        term3 = (eX * V * eX')[1]
        vhat = T1 - 2 * term2 + term3
        # The β-estimation correction (term2/term3) is unreliable under instrument
        # proliferation (n_inst ≫ N ⇒ rank-deficient A_N). Fall back to the leading robust
        # meat T1 (Var(r) with β treated as known — always ≥ 0) so the statistic stays
        # valid; T1 dominates for large N, and the sign property is preserved.
        if !(vhat > zero(T)) || !isfinite(vhat)
            vhat = T1
        end
        (vhat <= zero(T) || !isfinite(vhat)) && return (T(NaN), one(T))
        m = r / sqrt(vhat)
        (m, T(2 * (1 - cdf(Normal(), abs(m)))))
    end

    m1, pv1 = m_stat(1)
    m2, pv2 = m_stat(2)
    (m1=m1, p1=pv1, m2=m2, p2=pv2)
end

"""
    arellano_bond_ar_test(m::PanelRegModel; order::Int=2) -> NamedTuple

Arellano-Bond (1991) test for `order`-th (1 or 2) order serial correlation in the
first-differenced residuals of a dynamic-panel GMM model (`:ab`/`:bb`). Returns
`(statistic, pvalue, order)`; the statistic is asymptotically N(0,1) under the null of no
order-`order` serial correlation. A correctly specified model rejects at order 1 (Δε is
MA(1)) but not at order 2.

# Example
```julia
m = estimate_xtreg(pd, :y, :x; method=:ab)
ar2 = arellano_bond_ar_test(m; order=2)   # expect ar2.pvalue > 0.05
```
"""
function arellano_bond_ar_test(m::PanelRegModel; order::Int=2)
    d = m.dynamic_diagnostics
    d === nothing && throw(ArgumentError(
        "arellano_bond_ar_test requires a dynamic-panel model (:ab/:bb); got :$(m.method)"))
    order == 1 && return (statistic=d.ar1, pvalue=d.ar1_p, order=1)
    order == 2 && return (statistic=d.ar2, pvalue=d.ar2_p, order=2)
    throw(ArgumentError("order must be 1 or 2, got $order"))
end
