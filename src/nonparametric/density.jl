# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Kernel density estimation (EV-33, #441)
# =============================================================================

# --- Kernel functions -------------------------------------------------------
# All kernels are scaled to UNIT VARIANCE (the R `density()` convention), so a
# common bandwidth `h` produces comparable smoothing across kernels and each
# kernel integrates to one. This keeps the KDE a proper density.
@inline function _np_kernel(u::T, kernel::Symbol) where {T<:AbstractFloat}
    if kernel === :gaussian
        return exp(-u * u / 2) / sqrt(T(2π))
    elseif kernel === :epanechnikov
        # unit-variance Epanechnikov: support |u| < √5
        a = sqrt(T(5))
        return abs(u) < a ? (3 / (4a)) * (1 - (u / a)^2) : zero(T)
    elseif kernel === :triangular
        # unit-variance triangular: support |u| < √6
        a = sqrt(T(6))
        return abs(u) < a ? (1 / a) * (1 - abs(u) / a) : zero(T)
    elseif kernel === :uniform
        # unit-variance uniform (rectangular): support |u| < √3
        a = sqrt(T(3))
        return abs(u) < a ? 1 / (2a) : zero(T)
    else
        throw(ArgumentError("unknown kernel :$kernel (use :gaussian, :epanechnikov, :triangular, :uniform)"))
    end
end

# Interquartile range using the type-7 (linear-interpolation) quantile — matches
# R's default `IQR()`/`quantile()` and Julia's `Statistics.quantile` default.
_np_iqr(x::AbstractVector{T}) where {T} = quantile(x, T(0.75)) - quantile(x, T(0.25))

"""
    _bw_silverman(x) -> h

Silverman rule-of-thumb bandwidth (identical to R `stats::bw.nrd0`):
`h = 0.9 · min(σ̂, IQR/1.349) · n^(−1/5)`, with the documented fall-backs when
the spread degenerates.
"""
function _bw_silverman(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    hi = std(x)                       # sample sd, (n-1) denominator
    lo = min(hi, _np_iqr(x) / T(1.349))
    if lo == 0
        lo = hi > 0 ? hi : (abs(x[1]) > 0 ? abs(x[1]) : one(T))
    end
    return T(0.9) * lo * T(n)^(-T(1) / 5)
end

# --- Sheather–Jones plug-in (solve-the-equation) ----------------------------
# Exact pairwise evaluation of the density-derivative functionals φ4, φ6 used by
# the SJ-STE bandwidth (mirrors R `bw.SJ(method="ste")`; R bins onto 1000 cells,
# here we sum pairs exactly — with these sample sizes the difference is < 1e-3).
function _phi4(x::AbstractVector{T}, h::T) where {T<:AbstractFloat}
    n = length(x)
    s = zero(T)
    @inbounds for i in 1:n, j in 1:n
        δ = (x[i] - x[j]) / h
        δ2 = δ * δ
        s += exp(-δ2 / 2) * (δ2 * δ2 - 6 * δ2 + 3)   # K^(4); diagonal ⇒ 3
    end
    return s / (n * (n - 1) * h^5 * sqrt(T(2π)))
end

function _phi6(x::AbstractVector{T}, h::T) where {T<:AbstractFloat}
    n = length(x)
    s = zero(T)
    @inbounds for i in 1:n, j in 1:n
        δ = (x[i] - x[j]) / h
        δ2 = δ * δ
        s += exp(-δ2 / 2) * (δ2 * δ2 * δ2 - 15 * δ2 * δ2 + 45 * δ2 - 15)  # K^(6); diag ⇒ -15
    end
    return s / (n * (n - 1) * h^7 * sqrt(T(2π)))
end

"""
    _bw_sj(x) -> h

Sheather–Jones (1991) plug-in bandwidth via the solve-the-equation rule
(mirrors R `stats::bw.SJ(method="ste")`). The root of `fSD(h) = (c₁/φ4(α₂(h)))^{1/5} − h`
is bracketed on `[0.1·hmax, hmax]` around the Silverman scale and located by
bisection; the SJ estimate targets the AMISE-optimal bandwidth through pilot
estimation of the integrated squared density derivative. Falls back to the
Silverman rule (with a warning) if the pilot functional is non-positive or the
bracket cannot be established.
"""
function _bw_sj(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    scale = min(std(x), _np_iqr(x) / T(1.349))
    if scale <= 0
        @warn "Sheather–Jones bandwidth: degenerate scale; falling back to Silverman rule"
        return _bw_silverman(x)
    end
    a = T(1.24) * scale * T(n)^(-T(1) / 7)
    b = T(1.23) * scale * T(n)^(-T(1) / 9)
    c1 = 1 / (2 * sqrt(T(π)) * n)
    TD = -_phi6(x, b)
    if !isfinite(TD) || TD <= 0
        @warn "Sheather–Jones bandwidth: non-positive pilot functional; falling back to Silverman rule"
        return _bw_silverman(x)
    end
    SDa = _phi4(x, a)
    alph2(h) = T(1.357) * (abs(SDa / TD))^(T(1) / 7) * h^(T(5) / 7)
    fSD(h) = (c1 / _phi4(x, alph2(h)))^(T(1) / 5) - h

    hmax = T(1.144) * scale * T(n)^(-T(1) / 5)
    lower = T(0.1) * hmax
    upper = hmax
    flo = fSD(lower); fhi = fSD(upper)
    itry = 0
    while flo * fhi > 0
        itry += 1
        if itry > 99
            @warn "Sheather–Jones bandwidth: could not bracket root; falling back to Silverman rule"
            return _bw_silverman(x)
        end
        if isodd(itry)
            upper *= T(1.2); fhi = fSD(upper)
        else
            lower /= T(1.2); flo = fSD(lower)
        end
    end
    # Bisection to a tight tolerance.
    for _ in 1:200
        mid = (lower + upper) / 2
        fm = fSD(mid)
        if abs(fm) < 1e-12 || (upper - lower) < 1e-10 * hmax
            return mid
        end
        if flo * fm <= 0
            upper = mid
        else
            lower = mid; flo = fm
        end
    end
    return (lower + upper) / 2
end

"""
    kernel_density(y; kernel=:gaussian, bw=:silverman, npoints=512, cut=3.0)

Univariate kernel density estimate on an equally-spaced grid of `npoints`.

`f̂(x₀) = (1/nh) Σᵢ K((x₀ − yᵢ)/h)`.

# Arguments
- `kernel ∈ (:gaussian, :epanechnikov, :triangular, :uniform)` — unit-variance kernels.
- `bw` — bandwidth rule `:silverman` (R `bw.nrd0`), `:sj` (Sheather–Jones plug-in,
  R `bw.SJ`), or a positive real value.
- `npoints` — number of grid points.
- `cut` — grid extends `cut·h` beyond the data range on each side.

Returns [`KernelDensity`](@ref). A direct `O(n·npoints)` evaluation is used
(exact; FFTW binning is a possible optimisation for very large `n`).
"""
function kernel_density(y::AbstractVector{<:Real}; kernel::Symbol=:gaussian,
                        bw::Union{Symbol,Real}=:silverman, npoints::Int=512,
                        cut::Real=3.0)
    T = float(eltype(y)) <: AbstractFloat ? float(eltype(y)) : Float64
    data = collect(T, y)
    n = length(data)
    n >= 2 || throw(ArgumentError("kernel_density requires at least 2 observations"))
    npoints >= 2 || throw(ArgumentError("npoints must be ≥ 2"))
    # kernel validity check (throws on unknown)
    _np_kernel(zero(T), kernel)

    local h::T
    bw_method::Symbol = :user
    if bw isa Symbol
        if bw === :silverman
            h = _bw_silverman(data); bw_method = :silverman
        elseif bw === :sj
            h = _bw_sj(data); bw_method = :sj
        else
            throw(ArgumentError("unknown bandwidth rule :$bw (use :silverman, :sj, or a real value)"))
        end
    else
        h = T(bw)
        h > 0 || throw(ArgumentError("numeric bandwidth must be positive"))
    end

    lo = minimum(data) - cut * h
    hi = maximum(data) + cut * h
    grid = collect(range(lo, hi; length=npoints))
    dens = Vector{T}(undef, npoints)
    invnh = one(T) / (n * h)
    @inbounds for g in 1:npoints
        s = zero(T)
        xg = grid[g]
        for i in 1:n
            s += _np_kernel((xg - data[i]) / h, kernel)
        end
        dens[g] = s * invnh
    end
    return KernelDensity{T}(grid, dens, h, kernel, bw_method, data, n)
end
