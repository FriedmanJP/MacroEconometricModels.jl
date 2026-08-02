# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Fat-tailed conditional distributions for the GARCH family
#
# Financial returns are leptokurtic, so a Gaussian conditional likelihood is misspecified.
# Bollerslev's standardized Student-t and Nelson's GED are the standard alternatives.
#
# Every density here is STANDARDIZED to unit variance. That is the whole point: the GARCH
# recursion already owns the scale through `h_t`, so an innovation distribution carrying its
# own free scale would be conflated with it and the model would not be identified. The
# standardizing constants are what make `Var(z_t) = 1` regardless of the shape parameter.
#
# References:
#   Bollerslev, T. (1987). A Conditionally Heteroskedastic Time Series Model for
#     Speculative Prices and Rates of Return. Review of Economics and Statistics 69(3).
#   Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
#     Econometrica 59(2), 347–370.

using SpecialFunctions

"""
    _vol_dist_nparams(dist) → Int

Number of extra estimated parameters the conditional distribution contributes: `0` for
`:normal`, `1` for `:student` and `:ged` (the shape).
"""
_vol_dist_nparams(dist::Symbol) = dist === :normal ? 0 : 1

"""
    _vol_dist_check(dist)

Validate a conditional-distribution symbol.
"""
function _vol_dist_check(dist::Symbol)
    dist in (:normal, :student, :ged) || throw(ArgumentError(
        "dist must be :normal, :student, or :ged; got :$dist"))
    return nothing
end

"""
    _std_t_logpdf(z, nu) → T

Log density of the **variance-standardized** Student-t with `nu > 2` degrees of freedom.

The raw `t_ν` has variance `ν/(ν−2)`, so `z` is scaled by `s = sqrt((ν−2)/ν)` to give unit
variance:

```math
\\log f(z) = \\log\\Gamma\\!\\left(\\tfrac{\\nu+1}{2}\\right) - \\log\\Gamma\\!\\left(\\tfrac{\\nu}{2}\\right)
- \\tfrac{1}{2}\\log\\!\\left(\\pi(\\nu-2)\\right) - \\tfrac{\\nu+1}{2}\\log\\!\\left(1 + \\tfrac{z^2}{\\nu-2}\\right)
```

Note `π(ν−2)` rather than `πν`: the Jacobian of the standardization is already folded in.
As `ν → ∞` this converges to the standard normal log density.
"""
@inline function _std_t_logpdf(z::T, nu::T) where {T<:Real}
    return SpecialFunctions.loggamma((nu + one(T)) / 2) -
           SpecialFunctions.loggamma(nu / 2) -
           log(T(π) * (nu - 2)) / 2 -
           (nu + one(T)) / 2 * log1p(z^2 / (nu - 2))
end

"""
    _std_ged_logpdf(z, nu) → T

Log density of the **unit-variance** generalized error distribution with shape `nu > 0`
(Nelson 1991). `nu = 2` is the standard normal, `nu = 1` the Laplace, and `nu < 2` is
fatter-tailed than Gaussian.

```math
f(z) = \\frac{\\nu}{2\\lambda\\,\\Gamma(1/\\nu)}\\exp\\!\\left(-\\left|\\tfrac{z}{\\lambda}\\right|^{\\nu}\\right),
\\qquad \\lambda = \\sqrt{\\Gamma(1/\\nu)\\,/\\,\\Gamma(3/\\nu)}
```

The `λ` above is exactly the constant that makes the variance 1 for every `nu`.
"""
@inline function _std_ged_logpdf(z::T, nu::T) where {T<:Real}
    lam = sqrt(exp(SpecialFunctions.loggamma(one(T) / nu) -
                   SpecialFunctions.loggamma(3 * one(T) / nu)))
    return log(nu) - log(T(2) * lam) - SpecialFunctions.loggamma(one(T) / nu) -
           abs(z / lam)^nu
end

"""
    _vol_innov_logpdf(z, dist, shape) → T

Standardized innovation log density for `dist ∈ (:normal, :student, :ged)`. `shape` is
ignored for `:normal`.
"""
@inline function _vol_innov_logpdf(z::T, dist::Symbol, shape::T) where {T<:Real}
    if dist === :student
        return _std_t_logpdf(z, shape)
    elseif dist === :ged
        return _std_ged_logpdf(z, shape)
    else
        return -(log(T(2π)) + z^2) / 2
    end
end

"""
    _vol_negloglik_dist(h, resid_sq, n, dist, shape) → T

Negative log likelihood of a volatility model under a standardized innovation distribution.

With `z_t = resid_t / sqrt(h_t)`, the observation density carries the Jacobian of that
transformation, so each term is `-0.5 log h_t + log f(z_t)`. For `:normal` this reduces
exactly to the Gaussian form in `_volatility_negloglik`.
"""
function _vol_negloglik_dist(h, resid_sq, n::Int, dist::Symbol, shape)
    ll = zero(eltype(h))
    @inbounds for t in 1:n
        ht = h[t]
        # A diverging variance recursion (EGARCH exponentiates, so it can overflow) must
        # become a large finite penalty, not an Inf: the line search asserts finiteness and
        # aborts the whole optimization on a NaN.
        (isfinite(ht) && ht > 0) || return oftype(ll, 1e10)
        z = sqrt(resid_sq[t] / ht)
        term = -log(ht) / 2 + _vol_innov_logpdf(z, dist, shape)
        isfinite(term) || return oftype(ll, 1e10)
        ll += term
    end
    return isfinite(ll) ? -ll : oftype(ll, 1e10)
end

"""
    _vol_loglik_contribs_dist(h, resid_sq, dist, shape) → Vector

Per-observation log-likelihood contributions under a standardized innovation distribution.
The element type follows `h`, so `ForwardDiff.Dual`s propagate and the score matrix used by
the QMLE sandwich can be built by automatic differentiation.
"""
function _vol_loglik_contribs_dist(h, resid_sq, dist::Symbol, shape)
    ll = similar(h)
    @inbounds for t in eachindex(h)
        z = sqrt(resid_sq[t] / h[t])
        ll[t] = -log(h[t]) / 2 + _vol_innov_logpdf(z, dist, shape)
    end
    return ll
end

"""
    _vol_shape_transform(x, dist) → shape
    _vol_shape_inverse(shape, dist) → x

Map the unconstrained optimizer coordinate to the shape parameter and back.

Student-t needs `ν > 2` for a finite variance (the standardization divides by `ν − 2`), so
the transform is `ν = 2 + exp(x)`; GED needs `ν > 0`, hence `ν = exp(x)`. Both are the
package's usual log-transform pattern for a positivity constraint, and both are then
**clamped** strictly inside their admissible range — `ν ∈ [2.01, 500]` for the t and
`ν ∈ [0.1, 50]` for the GED — see the comment in the body for why the boundary is not merely
theoretical in floating point.
"""
@inline function _vol_shape_transform(x::T, dist::Symbol) where {T<:Real}
    if dist === :student
        # Clamped away from 2: the variance standardization divides by (nu - 2), and
        # `2 + exp(x)` rounds to EXACTLY 2.0 in Float64 once x < -37, which would make the
        # density infinite and hand a NaN to the line search. The upper clamp keeps the
        # near-Gaussian region from wandering off to numerically indistinguishable values.
        return clamp(2 + exp(x), T(2.01), T(500))
    elseif dist === :ged
        return clamp(exp(x), T(0.1), T(50))
    else
        return zero(T)
    end
end

@inline function _vol_shape_inverse(shape::T, dist::Symbol) where {T<:Real}
    if dist === :student
        return log(max(shape - 2, T(1e-8)))
    elseif dist === :ged
        return log(max(shape, T(1e-8)))
    else
        return zero(T)
    end
end

"""
    _vol_shape_init(dist, ::Type{T}) → T

Starting value in the UNCONSTRAINED coordinate. Student-t starts at `ν ≈ 8` and GED at
`ν = 2` (the Gaussian case), both conventional and safely interior.
"""
function _vol_shape_init(dist::Symbol, ::Type{T}) where {T<:AbstractFloat}
    dist === :student && return log(T(6))       # nu = 2 + 6 = 8
    dist === :ged && return log(T(2))           # nu = 2 (Gaussian)
    return zero(T)
end
