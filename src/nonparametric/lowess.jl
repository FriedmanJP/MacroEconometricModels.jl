# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# LOWESS — Cleveland (1979) locally-weighted scatterplot smoother (EV-33, #441)
# =============================================================================
#
# Faithful port of R's `clowess`/`lowest` C routines (src/library/stats/src/
# lowess.c), including the tricube weights, the local-linear slope correction,
# the near-boundary h1/h9 tolerances, the `delta` interpolation skip, and the
# bisquare robustifying passes. Matches `stats::lowess` to machine precision.

@inline _fcube(x) = x * x * x
@inline _fsquare(x) = x * x

# One local fit at abscissa `xs` using the window [nleft, nright] (1-indexed).
function _lowest!(x::AbstractVector{T}, y::AbstractVector{T}, xs::T,
                  nleft::Int, nright::Int, w::AbstractVector{T},
                  userw::Bool, rw::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    range = x[n] - x[1]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = T(0.999) * h
    h1 = T(0.001) * h
    a = zero(T)           # sum of weights
    nrt = nleft
    j = nleft
    @inbounds while j <= n
        w[j] = zero(T)
        r = abs(x[j] - xs)
        if r <= h9
            if r <= h1
                w[j] = one(T)
            else
                w[j] = _fcube(one(T) - _fcube(r / h))
            end
            if userw
                w[j] *= rw[j]
            end
            a += w[j]
        elseif x[j] > xs
            break
        end
        nrt = j
        j += 1
    end
    if a <= 0
        return (false, zero(T))
    end
    @inbounds for k in nleft:nrt
        w[k] /= a
    end
    if h > 0
        aa = zero(T)
        @inbounds for k in nleft:nrt
            aa += w[k] * x[k]     # weighted centre of x
        end
        b = xs - aa
        c = zero(T)
        @inbounds for k in nleft:nrt
            c += w[k] * (x[k] - aa)^2
        end
        if sqrt(c) > T(0.001) * range
            b /= c
            @inbounds for k in nleft:nrt
                w[k] *= (b * (x[k] - aa) + one(T))
            end
        end
    end
    ys = zero(T)
    @inbounds for k in nleft:nrt
        ys += w[k] * y[k]
    end
    return (true, ys)
end

# Core smoother on already-sorted (x, y). Returns fitted values.
function _clowess(x::AbstractVector{T}, y::AbstractVector{T}, f::T,
                  nsteps::Int, delta::T) where {T<:AbstractFloat}
    n = length(x)
    ys = Vector{T}(undef, n)
    rw = Vector{T}(undef, n)
    res = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    if n < 2
        fill!(ys, n == 1 ? y[1] : zero(T))
        return ys
    end
    ns = max(min(Int(floor(f * n + 1e-7)), n), 2)
    for iter in 1:(nsteps + 1)
        nleft = 1
        nright = ns
        last = 0
        i = 1
        while true
            @inbounds while nright < n
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                d1 <= d2 && break          # radius won't shrink by moving right
                nleft += 1
                nright += 1
            end
            ok, yval = _lowest!(x, y, x[i], nleft, nright, w, iter > 1, rw)
            ys[i] = ok ? yval : y[i]
            # interpolate skipped points between `last` and `i`
            if last < i - 1
                denom = x[i] - x[last]
                @inbounds for j in (last + 1):(i - 1)
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (one(T) - alpha) * ys[last]
                end
            end
            last = i
            cut = x[last] + delta
            i = last + 1
            @inbounds while i <= n
                if x[i] > cut
                    break
                end
                if x[i] == x[last]
                    ys[i] = ys[last]
                    last = i
                end
                i += 1
            end
            i = max(last + 1, i - 1)
            last >= n && break
        end
        @inbounds for k in 1:n
            res[k] = y[k] - ys[k]
        end
        iter > nsteps && break
        # bisquare robustness weights from the residuals
        @inbounds for k in 1:n
            rw[k] = abs(res[k])
        end
        sorted = sort(rw)
        m1 = 1 + div(n, 2)
        m2 = n - m1 + 1
        cmad = T(3) * (sorted[m1] + sorted[m2])
        c9 = T(0.999) * cmad
        c1 = T(0.001) * cmad
        @inbounds for k in 1:n
            r = abs(res[k])
            if cmad == 0
                rw[k] = one(T)
            elseif r <= c1
                rw[k] = one(T)
            elseif r <= c9
                rw[k] = _fsquare(one(T) - _fsquare(r / cmad))
            else
                rw[k] = zero(T)
            end
        end
    end
    return ys
end

"""
    lowess(y, x; f=2/3, iter=3, delta=nothing)

Cleveland (1979) LOWESS: locally-weighted scatterplot smoother. Each fit uses
the `⌊f·n⌋` nearest neighbours with tricube weights `(1 − (d/dₘₐₓ)³)³`, a local
linear regression, and `iter` bisquare robustifying passes. `delta` skips
computation at abscissae within `delta` of the last computed point and linearly
interpolates (default `0.01·range(x)`, as in R).

Returns [`LowessFit`](@ref) with values sorted by `x`.
"""
function lowess(y::AbstractVector{<:Real}, x::AbstractVector{<:Real};
                f::Real=2//3, iter::Int=3, delta::Union{Real,Nothing}=nothing)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    T = promote_type(float(eltype(x)), float(eltype(y)))
    T = T <: AbstractFloat ? T : Float64
    n = length(x)
    n >= 2 || throw(ArgumentError("lowess requires at least 2 observations"))
    iter >= 0 || throw(ArgumentError("iter must be ≥ 0"))

    perm = sortperm(x)
    xs = collect(T, @view x[perm])
    ys = collect(T, @view y[perm])
    del = delta === nothing ? T(0.01) * (xs[end] - xs[1]) : T(delta)

    fitted = _clowess(xs, ys, T(f), iter, del)
    return LowessFit{T}(xs, fitted, ys, T(f), iter, n)
end
