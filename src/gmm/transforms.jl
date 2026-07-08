# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Parameter transforms for constrained optimization via bijective maps
between constrained (model) and unconstrained (optimizer) spaces.
"""

"""
    ParameterTransform{T<:AbstractFloat}

Bijective parameter transform specification for constrained optimization.

Transform rules per element:
- `(-Inf, Inf)` → identity
- `(0, Inf)` → exp/log
- `(-Inf, 0)` → -exp/-log
- `(a, b)` → logistic: `a + (b-a) / (1 + exp(-x))`

# Fields
- `lower::Vector{T}` — lower bounds (-Inf = unbounded below)
- `upper::Vector{T}` — upper bounds (Inf = unbounded above)
"""
struct ParameterTransform{T<:AbstractFloat}
    lower::Vector{T}
    upper::Vector{T}

    function ParameterTransform{T}(lower::Vector{T}, upper::Vector{T}) where {T<:AbstractFloat}
        @assert length(lower) == length(upper) "lower and upper must have same length"
        for i in eachindex(lower)
            @assert lower[i] < upper[i] || (isinf(lower[i]) && isinf(upper[i])) "lower[$i] must be < upper[$i]"
        end
        new{T}(lower, upper)
    end
end

ParameterTransform(lower::Vector{T}, upper::Vector{T}) where {T<:AbstractFloat} =
    ParameterTransform{T}(lower, upper)
ParameterTransform(lower::Vector{<:Real}, upper::Vector{<:Real}) =
    ParameterTransform(Float64.(lower), Float64.(upper))

"""
    to_unconstrained(pt::ParameterTransform, theta::AbstractVector) -> Vector

Map parameters from constrained (model) space to unconstrained (optimizer) space.
"""
function to_unconstrained(pt::ParameterTransform{T}, theta::AbstractVector) where {T}
    phi = similar(theta, T)
    for i in eachindex(theta)
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            # Identity
            phi[i] = theta[i]
        elseif isinf(hi) && lo == zero(T)
            # (0, Inf) → log
            phi[i] = log(theta[i])
        elseif isinf(hi) && isfinite(lo)
            # (a, Inf) → log(theta - a)
            phi[i] = log(theta[i] - lo)
        elseif isinf(lo) && hi == zero(T)
            # (-Inf, 0) → log(-theta)
            phi[i] = log(-theta[i])
        elseif isinf(lo) && isfinite(hi)
            # (-Inf, b) → log(b - theta)
            phi[i] = log(hi - theta[i])
        else
            # (a, b) → logit: log((theta - a) / (b - theta))
            phi[i] = log((theta[i] - lo) / (hi - theta[i]))
        end
    end
    phi
end

"""
    to_constrained(pt::ParameterTransform, phi::AbstractVector) -> Vector

Map parameters from unconstrained (optimizer) space to constrained (model) space.
"""
function to_constrained(pt::ParameterTransform{T}, phi::AbstractVector) where {T}
    theta = similar(phi, T)
    for i in eachindex(phi)
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            theta[i] = phi[i]
        elseif isinf(hi) && lo == zero(T)
            theta[i] = exp(phi[i])
        elseif isinf(hi) && isfinite(lo)
            theta[i] = lo + exp(phi[i])
        elseif isinf(lo) && hi == zero(T)
            theta[i] = -exp(phi[i])
        elseif isinf(lo) && isfinite(hi)
            theta[i] = hi - exp(phi[i])
        else
            theta[i] = lo + (hi - lo) / (one(T) + exp(-phi[i]))
        end
    end
    theta
end

"""
    _log1pexp(x) -> log(1 + exp(x)), numerically stable

Stable softplus: returns `x` for large `x` (where `log1p(exp(x))` would overflow)
and `log1p(exp(x))` otherwise.
"""
_log1pexp(x::T) where {T<:AbstractFloat} = x > T(35) ? x : log1p(exp(x))

"""
    log_jacobian(pt::ParameterTransform, phi::AbstractVector) -> T

Summed log absolute Jacobian `Σᵢ log|dθᵢ/dφᵢ|` of the inverse transform
(unconstrained → constrained) at `φ`, evaluated stably:

- identity: `0`
- `(a, ∞)` / `(-∞, b)` (exp-based): `φᵢ`
- `(a, b)` (logistic): `log(b−a) − log(1+e^{−φᵢ}) − log(1+e^{φᵢ})`

This is the Jacobian correction added to a log density when sampling in the
unconstrained space (Stan reference manual, constrained-parameter transforms).
"""
function log_jacobian(pt::ParameterTransform{T}, phi::AbstractVector) where {T<:AbstractFloat}
    lj = zero(T)
    for i in eachindex(phi)
        lo, hi = pt.lower[i], pt.upper[i]
        p = T(phi[i])
        if isinf(lo) && isinf(hi)
            # identity: log|1| = 0
        elseif isinf(hi) || isinf(lo)
            # exp-based one-sided transforms: |dθ/dφ| = e^φ
            lj += p
        else
            lj += log(hi - lo) - _log1pexp(-p) - _log1pexp(p)
        end
    end
    return lj
end

"""
    transform_jacobian(pt::ParameterTransform, phi::AbstractVector) -> Matrix

Diagonal Jacobian ∂θ/∂φ of the inverse transform (unconstrained → constrained).
Used for delta method SE correction.
"""
function transform_jacobian(pt::ParameterTransform{T}, phi::AbstractVector) where {T}
    n = length(phi)
    J = zeros(T, n, n)
    for i in 1:n
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            J[i, i] = one(T)
        elseif isinf(hi) && lo == zero(T)
            J[i, i] = exp(phi[i])
        elseif isinf(hi) && isfinite(lo)
            J[i, i] = exp(phi[i])
        elseif isinf(lo) && hi == zero(T)
            J[i, i] = -exp(phi[i])
        elseif isinf(lo) && isfinite(hi)
            J[i, i] = -exp(phi[i])
        else
            e = exp(-phi[i])
            J[i, i] = (hi - lo) * e / (one(T) + e)^2
        end
    end
    J
end
