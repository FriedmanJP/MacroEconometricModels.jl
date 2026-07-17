# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Phillips–Ouliaris (1990) residual-based cointegration test: the semiparametric
(Phillips–Perron-style) normalized-bias `Ẑ_α` and `t`-ratio `Ẑ_t` statistics computed on the
static cointegrating-regression residuals, reusing the EV-12 long-run-variance toolkit
(`lrvar`). Mirrors `gregory_hansen.jl`'s `_gh_pp_on_residuals`.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Phillips–Perron-style statistics on cointegrating-regression residuals
# =============================================================================

"""
    _po_pp_stats(resid; kernel, bandwidth) -> (zt, za, bw)

Phillips–Ouliaris `Ẑ_t` and `Ẑ_α` on residuals `û_t`. Fits the AR(1) `û_t = ρ̂ û_{t-1} + ξ_t`,
takes the short-run variance `s² = ξ'ξ/T` and the long-run variance `ω² = lrvar(ξ)` (EV-12),
and applies the Phillips (1987) / Phillips–Ouliaris (1990) semiparametric corrections

    Ẑ_α = T(ρ̂−1) − (ω²−s²)·T² / (2·Σ û_{t-1}²)
    Ẑ_t = √(s²/ω²)·t_ρ − (ω²−s²) / (2·ω·(T⁻¹Σû_{t-1}²)^{1/2}·√T)

`bandwidth` is forwarded to `lrvar` (`:nw` ⇒ the fixed `⌊4(T/100)^{1/4}⌋` Bartlett lag).
Returns the two statistics and the resolved integer bandwidth.
"""
function _po_pp_stats(resid::AbstractVector{T}; kernel::Symbol=:bartlett,
                      bandwidth=:nw) where {T<:AbstractFloat}
    n = length(resid)
    e_lag = resid[1:n-1]
    e_curr = resid[2:n]
    nobs = n - 1

    Sll = dot(e_lag, e_lag)
    rho = dot(e_lag, e_curr) / Sll
    xi = e_curr .- rho .* e_lag
    s2 = dot(xi, xi) / nobs                          # short-run variance γ̂₀

    bw = bandwidth === :nw ? floor(Int, 4 * (nobs / 100)^0.25) : bandwidth
    omega2 = lrvar(xi; kernel=kernel, bandwidth=bw, demean=false)
    bw_used = bw isa Int ? bw : floor(Int, 4 * (nobs / 100)^0.25)

    lambda = (omega2 - s2) / 2                        # one-sided correction

    # Ẑ_α (normalized bias).
    za = nobs * (rho - 1) - lambda * nobs^2 / Sll

    # Ẑ_t (studentized).
    se_rho = sqrt(s2 / Sll)
    t_rho = (rho - 1) / se_rho
    zt = sqrt(s2 / omega2) * t_rho - lambda / (omega2^(one(T) / 2) * sqrt(Sll / nobs) * sqrt(T(nobs)))

    return T(zt), T(za), bw_used
end

"""
    _po_za_pvalue(za, regression, N) -> T

Bracketing p-value for the Phillips–Ouliaris `Ẑ_α` statistic from the simulated
`PO_ZA_CV` critical values (1/5/10%) for deterministic case `regression` and `N = k+1`
I(1) series, linearly interpolated between the tabulated levels (mirrors
`gregory_hansen.jl`'s `_gh_pvalue`). `Ẑ_α` has no MacKinnon response surface, so critical
values are Monte-Carlo (see `PO_ZA_CV` provenance).
"""
function _po_za_pvalue(za::T, regression::Symbol, N::Int) where {T<:AbstractFloat}
    Nc = clamp(N, 1, 6)
    cv = PO_ZA_CV[regression][Nc]                    # (cv1, cv5, cv10), all negative
    cv1, cv5, cv10 = T(cv[1]), T(cv[2]), T(cv[3])
    if za <= cv1
        return T(0.01)
    elseif za <= cv5
        return T(0.01) + (za - cv1) / (cv5 - cv1) * T(0.04)
    elseif za <= cv10
        return T(0.05) + (za - cv5) / (cv10 - cv5) * T(0.05)
    else
        return min(one(T), T(0.10) + (za - cv10) / abs(cv10) * T(0.30))
    end
end

# =============================================================================
# Phillips–Ouliaris (1990) test
# =============================================================================

"""
    phillips_ouliaris_test(y, X; trend=:constant, kernel=:bartlett, bandwidth=:nw)
        -> PhillipsOuliarisResult

Phillips–Ouliaris (1990) residual-based test for **no cointegration** between the `I(1)`
dependent variable `y` and `I(1)` regressors `X`. Regresses `y` on the deterministics and
`X` (levels OLS) and applies the Phillips–Perron-style semiparametric corrections to the
residual AR(1) root, producing the normalized-bias `Ẑ_α` and the `t`-ratio `Ẑ_t`.

# Keywords
- `trend`: deterministics — `:none`, `:constant` (default), `:trend`.
- `kernel`: HAC kernel for the residual long-run variance (forwarded to EV-12 `lrvar`).
- `bandwidth`: `:nw` (fixed `⌊4(T/100)^{1/4}⌋`, default), `:andrews`, `:nw94`, or a number.

# p-values
- `Ẑ_t`: MacKinnon (1996/2010) cointegration response surface (`N = k+1`) — asymptotically
  the residual Dickey–Fuller `t` distribution, shared with Engle–Granger.
- `Ẑ_α`: bracketing interpolation of the Monte-Carlo `PO_ZA_CV` critical values (no
  MacKinnon surface exists for the normalized-bias statistic).

`H₀`: no cointegration. Large-negative statistics / small p-values reject `H₀`.

# References
- Phillips, P. C. B. & Ouliaris, S. (1990). Asymptotic properties of residual based tests
  for cointegration. *Econometrica* 58(1), 165–193.
"""
function phillips_ouliaris_test(y::AbstractVector{T}, X::AbstractMatrix{T};
                                trend::Symbol=:constant,
                                kernel::Symbol=:bartlett,
                                bandwidth=:nw) where {T<:AbstractFloat}
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("length(y)=$n must equal size(X,1)=$(size(X,1))"))
    k = size(X, 2)
    k >= 1 || throw(ArgumentError("need at least one regressor column"))
    n > 3 * k + 12 || throw(ArgumentError("too few observations ($n) for $k regressor(s)"))

    regression = _coint_regression(trend)
    resid_vec = _coint_levels_resid(y, X, regression)

    zt, za, bw = _po_pp_stats(resid_vec; kernel=kernel, bandwidth=bandwidth)

    N = k + 1
    zt_pval = _mackinnon_coint_pvalue(zt, regression, N)
    za_pval = _po_za_pvalue(za, regression, N)

    return PhillipsOuliarisResult(zt, zt_pval, za, za_pval, regression, kernel, T(bw),
                                  k, N, n)
end

phillips_ouliaris_test(y::AbstractVector, X::AbstractVector; kwargs...) =
    phillips_ouliaris_test(Float64.(y), reshape(Float64.(X), :, 1); kwargs...)
phillips_ouliaris_test(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    phillips_ouliaris_test(Float64.(y), Float64.(X); kwargs...)

"""
    phillips_ouliaris_test(Y::AbstractMatrix; kwargs...)

Convenience: `Y[:,1]` dependent, `Y[:,2:end]` regressors.
"""
function phillips_ouliaris_test(Y::AbstractMatrix; kwargs...)
    size(Y, 2) >= 2 || throw(ArgumentError("Y needs ≥ 2 columns (dependent + regressor)"))
    return phillips_ouliaris_test(Float64.(Y[:, 1]), Float64.(Y[:, 2:end]); kwargs...)
end
