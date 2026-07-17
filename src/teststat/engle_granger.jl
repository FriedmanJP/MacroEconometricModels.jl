# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Engle–Granger (1987) two-step residual-based cointegration test, plus the shared
cointegration helpers (`_mackinnon_coint_pvalue`, `_coint_levels_resid`) reused by the
Phillips–Ouliaris test. Mirrors `src/teststat/gregory_hansen.jl`'s residual-ADF machinery
but keys p-values off the MacKinnon (1996/2010) **cointegration** response surfaces indexed
by `N = k+1` (number of I(1) series), not the univariate ADF surface.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Shared cointegration helpers (used by Engle–Granger AND Phillips–Ouliaris)
# =============================================================================

"""
    _coint_regression(trend) -> Symbol

Normalise a user-facing `trend` keyword (`:none`, `:const`/`:constant`, `:trend`/`:linear`)
to the internal MacKinnon-surface case symbol (`:none`, `:constant`, `:trend`).
"""
function _coint_regression(trend::Symbol)
    if trend === :none
        return :none
    elseif trend === :const || trend === :constant
        return :constant
    elseif trend === :trend || trend === :linear
        return :trend
    else
        throw(ArgumentError("trend must be :none, :constant, or :trend; got :$trend"))
    end
end

"""
    _coint_deterministics(n, regression, ::Type{T}) -> Matrix{T}

`n×d` deterministic block for the cointegrating regression: `0` columns (`:none`),
a constant (`:constant`), or a constant + linear trend (`:trend`).
"""
function _coint_deterministics(n::Int, regression::Symbol, ::Type{T}) where {T<:AbstractFloat}
    if regression === :none
        return Matrix{T}(undef, n, 0)
    elseif regression === :constant
        return ones(T, n, 1)
    else # :trend
        return hcat(ones(T, n), T.(1:n))
    end
end

"""
    _coint_levels_resid(y, X, regression) -> resid

OLS residuals of the static cointegrating regression `y_t = D_t'δ + x_t'β + u_t` in levels
(the Engle–Granger / Phillips–Ouliaris first stage). Deterministics `D` are governed by
`regression`. Uses `robust_inv` for the normal equations.
"""
function _coint_levels_resid(y::AbstractVector{T}, X::AbstractMatrix{T},
                             regression::Symbol) where {T<:AbstractFloat}
    n = length(y)
    D = _coint_deterministics(n, regression, T)
    Z = hcat(X, D)                                   # order irrelevant for residuals
    beta = robust_inv(Symmetric(Z' * Z)) * (Z' * y)
    return y .- Z * beta
end

"""
    _mackinnon_coint_pvalue(stat, regression, N) -> T

MacKinnon (1996/2010) asymptotic p-value for a residual-based cointegration `t`/`Ẑ_t`
statistic. `regression ∈ (:none, :constant, :trend)`; `N` = number of I(1) series in the
cointegrating vector (`= k+1`), clamped to `1:6`. Returns `p = Φ(P(τ))` with the polynomial
link `P` quadratic for `τ ≤ τ*` and cubic above, saturating to 0/1 outside `[τ_min, τ_max]`.
The Normal CDF is the response-surface LINK, not a Gaussian tail on the raw statistic.
"""
function _mackinnon_coint_pvalue(stat::T, regression::Symbol, N::Int) where {T<:AbstractFloat}
    haskey(MACKINNON_COINT_SMALLP, regression) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend; got :$regression"))
    Nc = clamp(N, 1, 6)
    stat > MACKINNON_COINT_TAUMAX[regression][Nc] && return one(T)
    stat < MACKINNON_COINT_TAUMIN[regression][Nc] && return zero(T)
    poly = if stat <= MACKINNON_COINT_TAUSTAR[regression][Nc]
        c = MACKINNON_COINT_SMALLP[regression]
        c[Nc, 1] + c[Nc, 2] * stat + c[Nc, 3] * stat^2
    else
        c = MACKINNON_COINT_LARGEP[regression]
        c[Nc, 1] + c[Nc, 2] * stat + c[Nc, 3] * stat^2 + c[Nc, 4] * stat^3
    end
    return T(cdf(Normal(), poly))
end

# =============================================================================
# Engle–Granger (1987) two-step cointegration test
# =============================================================================

"""
    engle_granger_test(y, X; trend=:constant, lags=:aic, max_lags=nothing) -> EngleGrangerResult

Engle–Granger (1987) two-step residual-based test for **no cointegration** between the
`I(1)` dependent variable `y` and the `I(1)` regressor matrix `X`.

Stage 1 regresses `y` on the deterministics and `X` (levels OLS); stage 2 runs an augmented
Dickey–Fuller regression **with no deterministic term** on the residuals `û_t`,

    Δû_t = ρ û_{t-1} + Σ_{j=1}^p γ_j Δû_{t-j} + e_t,

and reports the `t`-statistic on `ρ`. The p-value comes from the MacKinnon (1996/2010)
**cointegration** response surface indexed by `N = k+1` (number of I(1) series) and the
deterministic case — *not* the univariate ADF surface.

# Arguments
- `y::AbstractVector` — dependent variable (levels, `I(1)`).
- `X::AbstractVecOrMat` — `I(1)` regressor(s); a vector is treated as a single column.

# Keywords
- `trend`: deterministics in the cointegrating regression — `:none`, `:constant` (default),
  or `:trend` (constant + linear trend). Selects the MacKinnon surface (`n`/`c`/`ct`).
- `lags`: augmenting lags `p`, or `:aic`/`:bic` to select over `0:max_lags`.
- `max_lags`: IC search ceiling (default `⌊12(T/100)^{1/4}⌋`).

# Returns
An [`EngleGrangerResult`](@ref). `H₀`: no cointegration (unit root in `û`). A small p-value /
a statistic below the critical values rejects `H₀` in favour of cointegration.

# References
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction:
  representation, estimation, and testing. *Econometrica* 55(2), 251–276.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Univ. WP 1227.
"""
function engle_granger_test(y::AbstractVector{T}, X::AbstractMatrix{T};
                            trend::Symbol=:constant,
                            lags::Union{Int,Symbol}=:aic,
                            max_lags::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("length(y)=$n must equal size(X,1)=$(size(X,1))"))
    k = size(X, 2)
    k >= 1 || throw(ArgumentError("need at least one regressor column"))
    n > 3 * k + 12 || throw(ArgumentError("too few observations ($n) for $k regressor(s)"))

    regression = _coint_regression(trend)
    resid_vec = _coint_levels_resid(y, X, regression)

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags
    adf_stat, best_p = _gh_adf_on_residuals(resid_vec, n, max_p, lags, T)

    N = k + 1
    pval = _mackinnon_coint_pvalue(adf_stat, regression, N)

    return EngleGrangerResult(adf_stat, pval, best_p, regression, k, N, n)
end

engle_granger_test(y::AbstractVector, X::AbstractVector; kwargs...) =
    engle_granger_test(Float64.(y), reshape(Float64.(X), :, 1); kwargs...)
engle_granger_test(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    engle_granger_test(Float64.(y), Float64.(X); kwargs...)

"""
    engle_granger_test(Y::AbstractMatrix; kwargs...)

Convenience: `Y[:,1]` is the dependent variable, `Y[:,2:end]` the regressors.
"""
function engle_granger_test(Y::AbstractMatrix; kwargs...)
    size(Y, 2) >= 2 || throw(ArgumentError("Y needs ≥ 2 columns (dependent + regressor)"))
    return engle_granger_test(Float64.(Y[:, 1]), Float64.(Y[:, 2:end]); kwargs...)
end
