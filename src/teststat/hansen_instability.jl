# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Hansen (1992) `L_c` parameter-instability test for cointegrating regressions. `H₀` is
cointegration with *stable* coefficients; the alternative is coefficient variation (a
martingale in the loadings), which is observationally equivalent to no cointegration.
Consumes an EV-10 [`CointRegModel`](@ref) and reuses its stored conditional long-run
variance `ω̂²_{u·v}`.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Hansen (1992) L_c statistic
# =============================================================================

"""
    hansen_instability_test(m::CointRegModel) -> HansenInstabilityResult

Hansen (1992) `L_c` test for parameter stability of the cointegrating regression `m`.

With full regressor row `Z_t = [D_t; x_t]` (deterministics + `I(1)` regressors, `p = d+k`
columns), residuals `û_t`, cumulative scores `Ŝ_t = Σ_{i≤t} Z_i û_i`, and the stored
conditional long-run variance `ω̂²_{u·v}`,

    L_c = ω̂_{u·v}^{-2} · T^{-1} · Σ_{t=1}^T Ŝ_t' (Σ_i Z_i Z_i')^{-1} Ŝ_t.

A **large** `L_c` rejects the null of stable cointegration. Critical values come from the
Monte-Carlo `HANSEN_LC_CV` table (see its provenance comment), indexed by the deterministic
case and `k` (number of `I(1)` regressors); the reported p-value brackets the 1/5/10% cells.

# References
- Hansen, B. E. (1992). Tests for parameter instability in regressions with I(1) processes.
  *Journal of Business & Economic Statistics* 10(3), 321–335.
"""
function hansen_instability_test(m::CointRegModel{T}) where {T<:AbstractFloat}
    regression = _coint_regression(m.trend)
    n = m.nobs
    D = _coint_deterministics(n, regression, T)
    Z = hcat(D, m.X)                                 # T×p, p = d+k
    p = size(Z, 2)
    resid_vec = m.residuals

    f = Z .* resid_vec                               # T×p scores Z_t û_t
    S = cumsum(f; dims=1)                            # cumulative scores Ŝ_t

    M = Symmetric(Z' * Z)
    Minv = robust_inv(M)
    omega2 = m.omega_uv
    omega2 > zero(T) || (omega2 = max(var(resid_vec), eps(T)))

    acc = zero(T)
    @inbounds for t in 1:n
        st = @view S[t, :]
        acc += dot(st, Minv * st)
    end
    Lc = acc / (T(n) * omega2)

    k = m.k
    pval = _hansen_lc_pvalue(Lc, regression, k)

    return HansenInstabilityResult(Lc, pval, regression, m.trend, p, k, n)
end

"""
    _hansen_lc_pvalue(Lc, regression, k) -> T

Bracketing p-value for Hansen's `L_c` from the Monte-Carlo `HANSEN_LC_CV` critical values
(10/5/1%, ascending) for deterministic case `regression` and `k` `I(1)` regressors. `L_c` is
right-tailed (large ⇒ reject), so p decreases as `L_c` grows.
"""
function _hansen_lc_pvalue(Lc::T, regression::Symbol, k::Int) where {T<:AbstractFloat}
    kc = clamp(k, 1, 5)
    cv = HANSEN_LC_CV[regression][kc]                # (cv10, cv5, cv1), ascending
    cv10, cv5, cv1 = T(cv[1]), T(cv[2]), T(cv[3])
    if Lc >= cv1
        return T(0.01)
    elseif Lc >= cv5
        return T(0.01) + (cv1 - Lc) / (cv1 - cv5) * T(0.04)
    elseif Lc >= cv10
        return T(0.05) + (cv5 - Lc) / (cv5 - cv10) * T(0.05)
    else
        return min(one(T), T(0.10) + (cv10 - Lc) / cv10 * T(0.40))
    end
end
