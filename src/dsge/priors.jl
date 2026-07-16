# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Dynare prior-convention shims.

Published DSGE priors are almost always declared in Dynare's (mean, std)
parameterization. `dynare_prior` converts those inputs into correctly
parameterized `Distributions.jl` objects — including the inverse-gamma-on-σ
(`InverseGamma1`) that naive use of `Distributions.InverseGamma` silently gets
wrong.
"""

using Distributions: loggamma

# =============================================================================
# InverseGamma1 — Dynare's inv_gamma_pdf: inverse gamma on σ (not σ²)
# =============================================================================

"""
    InverseGamma1(s, ν)

Dynare's **type-1 inverse gamma**: the distribution of `σ` when
`σ² ~ InverseGamma(ν/2, s/2)` (type-2 on the variance). This is the density
behind Dynare's `inv_gamma_pdf` — it lives on the **standard deviation σ, not
the variance σ²**. Passing Dynare `inv_gamma` numbers to
`Distributions.InverseGamma` produces a completely different prior; use this
type (or [`dynare_prior`](@ref)`(:inv_gamma, mean, std)`) instead.

Density: `p(σ) ∝ σ^{-(ν+1)} exp(−s/(2σ²))` on `σ > 0`.

Moments (for `ν > 2`):
`E[σ] = √(s/2)·Γ((ν−1)/2)/Γ(ν/2)`, `E[σ²] = s/(ν−2)`.
"""
struct InverseGamma1{T<:Real} <: ContinuousUnivariateDistribution
    s::T
    nu::T
    v::InverseGamma{T}   # distribution of σ²

    function InverseGamma1{T}(s::T, nu::T) where {T<:Real}
        s > 0 || throw(ArgumentError("InverseGamma1 scale s must be positive"))
        nu > 0 || throw(ArgumentError("InverseGamma1 dof ν must be positive"))
        new{T}(s, nu, InverseGamma(nu / 2, s / 2))
    end
end
InverseGamma1(s::T, nu::T) where {T<:Real} = InverseGamma1{T}(s, nu)
InverseGamma1(s::Real, nu::Real) = InverseGamma1(promote(float(s), float(nu))...)

Distributions.minimum(::InverseGamma1{T}) where {T} = zero(T)
Distributions.maximum(::InverseGamma1{T}) where {T} = T(Inf)
Distributions.insupport(::InverseGamma1, x::Real) = x > 0

function Distributions.logpdf(d::InverseGamma1{T}, x::Real) where {T}
    x > 0 || return T(-Inf)
    return logpdf(d.v, x^2) + log(2 * T(x))
end
Distributions.pdf(d::InverseGamma1, x::Real) = exp(logpdf(d, x))
Distributions.cdf(d::InverseGamma1{T}, x::Real) where {T} =
    x <= 0 ? zero(T) : cdf(d.v, T(x)^2)
Distributions.quantile(d::InverseGamma1, p::Real) = sqrt(quantile(d.v, p))
Base.rand(rng::AbstractRNG, d::InverseGamma1) = sqrt(rand(rng, d.v))

function Statistics.mean(d::InverseGamma1{T}) where {T}
    d.nu > 1 || return T(NaN)
    return exp(log(d.s / 2) / 2 + loggamma((d.nu - 1) / 2) - loggamma(d.nu / 2))
end
function Statistics.var(d::InverseGamma1{T}) where {T}
    d.nu > 2 || return T(NaN)
    return d.s / (d.nu - 2) - mean(d)^2
end
Statistics.std(d::InverseGamma1) = sqrt(var(d))

# =============================================================================
# dynare_prior — (mean, std) constructors matching Dynare's pdf conventions
# =============================================================================

"""
    _ig1_from_moments(m, sd) → InverseGamma1

Solve Dynare's `(s, ν)` from the target mean `m` and std `sd` of σ. Uses
`E[σ²] = s/(ν−2) = m² + sd²` to eliminate `s`, then bisects on `ν` so that
`E[σ] = m`.
"""
function _ig1_from_moments(m::Float64, sd::Float64)
    c = m^2 + sd^2
    # E[σ](ν) with s = c(ν−2); monotone in ν from 0 (ν→2⁺) to √c (ν→∞)
    mean_at = nu -> exp(log(c * (nu - 2) / 2) / 2 +
                        loggamma((nu - 1) / 2) - loggamma(nu / 2))
    lo, hi = 2.0 + 1e-10, 1e10
    # expand hi is unnecessary: mean_at(hi) → √c > m guaranteed
    for _ in 1:500
        mid = (lo + hi) / 2
        if mean_at(mid) < m
            lo = mid
        else
            hi = mid
        end
        hi - lo < 1e-12 * hi && break
    end
    nu = (lo + hi) / 2
    s = c * (nu - 2)
    return InverseGamma1(s, nu)
end

"""
    dynare_prior(dist::Symbol, mean, std; lower=nothing, upper=nothing) → Distribution

Construct a prior from Dynare's (mean, std) convention, returning a correctly
parameterized `Distributions.jl` object usable directly in the `priors` dict of
[`estimate_dsge_bayes`](@ref) / [`posterior_mode`](@ref).

| Dynare pdf | `dist` | Returns |
|---|---|---|
| `normal_pdf` | `:normal` | `Normal(mean, std)` |
| `gamma_pdf` | `:gamma` | `Gamma(mean²/std², std²/mean)` |
| `beta_pdf` | `:beta` | `Beta(α, β)` moment-matched; `(lower, upper)` gives Dynare's generalized `(p3, p4)` beta |
| `inv_gamma_pdf` / `inv_gamma1_pdf` | `:inv_gamma` / `:inv_gamma1` | [`InverseGamma1`](@ref) **on σ** with the requested mean/std |
| `inv_gamma2_pdf` | `:inv_gamma2` | `InverseGamma(α, β)` **on σ²** with the requested mean/std |
| `uniform_pdf` | `:uniform` | `Uniform(a, b)`; pass `lower`/`upper` directly or derive from (mean, std) |

!!! warning "Dynare's inverse gamma is on σ, not σ²"
    Dynare's `inv_gamma_pdf` is the type-1 inverse gamma on the **standard
    deviation**. `Distributions.InverseGamma` is on the **variance**. Feeding
    Dynare's numbers to `Distributions.InverseGamma` — the natural-looking
    thing to do — silently produces a completely different prior. This
    constructor exists to prevent exactly that bug: `dynare_prior(:inv_gamma,
    m, s)` returns a distribution whose draws are σ values with mean `m` and
    std `s`.

# Examples
```julia
priors = Dict(
    :rho    => dynare_prior(:beta, 0.7, 0.1),          # persistence
    :sigma  => dynare_prior(:inv_gamma, 0.02, 0.05),   # shock SD (σ-space!)
    :phi    => dynare_prior(:gamma, 1.5, 0.25),        # elasticity
    :alpha  => dynare_prior(:normal, 0.3, 0.05),
    :theta  => dynare_prior(:beta, 0.5, 0.1; lower=0.0, upper=0.9),
)
```
"""
function dynare_prior(dist::Symbol, mean_v::Real, std_v::Real;
                      lower::Union{Nothing,Real}=nothing,
                      upper::Union{Nothing,Real}=nothing)
    m = Float64(mean_v)
    sd = Float64(std_v)

    if dist == :normal
        sd > 0 || throw(ArgumentError("normal prior needs std > 0"))
        return Normal(m, sd)

    elseif dist == :gamma
        (m > 0 && sd > 0) || throw(ArgumentError("gamma prior needs mean > 0, std > 0"))
        k = m^2 / sd^2
        th = sd^2 / m
        return Gamma(k, th)

    elseif dist == :beta
        a = lower === nothing ? 0.0 : Float64(lower)
        b = upper === nothing ? 1.0 : Float64(upper)
        b > a || throw(ArgumentError("beta prior needs upper > lower"))
        mu = (m - a) / (b - a)
        sig = sd / (b - a)
        (0 < mu < 1) ||
            throw(ArgumentError("beta prior mean must lie strictly inside ($a, $b)"))
        sig^2 < mu * (1 - mu) ||
            throw(ArgumentError("beta prior std too large: needs std² < mean(1−mean) " *
                                "on the (lower, upper) scale"))
        k0 = mu * (1 - mu) / sig^2 - 1
        alpha = mu * k0
        beta = (1 - mu) * k0
        base = Beta(alpha, beta)
        return (a == 0.0 && b == 1.0) ? base : a + (b - a) * base

    elseif dist == :inv_gamma || dist == :inv_gamma1
        (m > 0 && sd > 0) ||
            throw(ArgumentError("inverse-gamma prior needs mean > 0, std > 0"))
        return _ig1_from_moments(m, sd)

    elseif dist == :inv_gamma2
        (m > 0 && sd > 0) ||
            throw(ArgumentError("inverse-gamma-2 prior needs mean > 0, std > 0"))
        alpha = m^2 / sd^2 + 2
        beta = m * (alpha - 1)
        return InverseGamma(alpha, beta)

    elseif dist == :uniform
        if lower !== nothing || upper !== nothing
            (lower !== nothing && upper !== nothing) ||
                throw(ArgumentError("uniform prior needs both lower and upper"))
            Float64(upper) > Float64(lower) ||
                throw(ArgumentError("uniform prior needs upper > lower"))
            return Uniform(Float64(lower), Float64(upper))
        end
        sd > 0 || throw(ArgumentError("uniform prior needs std > 0"))
        half = sqrt(3.0) * sd
        return Uniform(m - half, m + half)

    else
        throw(ArgumentError("unknown Dynare prior :$dist — use :normal, :gamma, " *
                            ":beta, :inv_gamma (σ), :inv_gamma2 (σ²), or :uniform"))
    end
end
