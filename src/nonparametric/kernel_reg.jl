# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Kernel / local-polynomial regression (EV-33, #441)
# =============================================================================

# Effective-weight vector ℓ(x₀) such that m̂(x₀) = Σᵢ ℓᵢ(x₀) yᵢ.
#   - degree 0 (Nadaraya–Watson): ℓᵢ = wᵢ / Σⱼ wⱼ
#   - degree ≥ 1 (Fan–Gijbels local polynomial): ℓ = e₁ᵀ (XᵀWX)⁻¹ XᵀW
# Falls back to the NW weights when the local design is rank-deficient (too few
# points carry weight — e.g. a tiny bandwidth), which yields interpolation of
# the data at the design points.
function _local_weights(xs::AbstractVector{T}, x0::T, h::T, degree::Int,
                        kernel::Symbol) where {T<:AbstractFloat}
    n = length(xs)
    w = Vector{T}(undef, n)
    sw = zero(T)
    @inbounds for i in 1:n
        w[i] = _np_kernel((xs[i] - x0) / h, kernel)
        sw += w[i]
    end
    if sw <= 0
        # no point in reach: put all mass on the nearest design point
        fill!(w, zero(T))
        w[argmin(abs.(xs .- x0))] = one(T)
        return w
    end
    if degree == 0
        return w ./ sw
    end
    p = degree + 1
    # local design matrix X (n × p): columns (xs - x0)^k, k = 0..degree
    A = zeros(T, p, p)
    @inbounds for i in 1:n
        w[i] == 0 && continue
        d = xs[i] - x0
        row = Vector{T}(undef, p)
        acc = one(T)
        for k in 1:p
            row[k] = acc
            acc *= d
        end
        for r in 1:p, c in 1:p
            A[r, c] += w[i] * row[r] * row[c]
        end
    end
    # rank check: need at least (degree+1) points with meaningful weight
    neff = count(>(1e-10 * sw), w)
    if neff < p || !isfinite(cond(A))
        return w ./ sw     # degenerate design ⇒ fall back to local constant
    end
    Ainv = robust_inv(Symmetric(A); silent=true)
    row1 = @view Ainv[1, :]
    ℓ = Vector{T}(undef, n)
    @inbounds for i in 1:n
        if w[i] == 0
            ℓ[i] = zero(T)
            continue
        end
        d = xs[i] - x0
        acc = one(T)
        s = zero(T)
        for k in 1:p
            s += row1[k] * acc
            acc *= d
        end
        ℓ[i] = w[i] * s
    end
    return ℓ
end

# Fitted value + effective-weight self-influence H_ii and ‖ℓ‖² at each design pt.
function _kreg_fit_at_design(xs::AbstractVector{T}, ys::AbstractVector{T}, h::T,
                             degree::Int, kernel::Symbol) where {T<:AbstractFloat}
    n = length(xs)
    fitted = Vector{T}(undef, n)
    hii = Vector{T}(undef, n)
    l2 = Vector{T}(undef, n)
    @inbounds for j in 1:n
        ℓ = _local_weights(xs, xs[j], h, degree, kernel)
        fitted[j] = dot(ℓ, ys)
        hii[j] = ℓ[j]
        l2[j] = dot(ℓ, ℓ)
    end
    return fitted, hii, l2
end

# Leave-one-out CV score for a candidate bandwidth (uses the H_ii shortcut).
function _kreg_cv(xs::AbstractVector{T}, ys::AbstractVector{T}, h::T,
                  degree::Int, kernel::Symbol) where {T<:AbstractFloat}
    fitted, hii, _ = _kreg_fit_at_design(xs, ys, h, degree, kernel)
    n = length(xs)
    s = zero(T)
    @inbounds for i in 1:n
        denom = one(T) - hii[i]
        abs(denom) < 1e-8 && return T(Inf)
        r = (ys[i] - fitted[i]) / denom
        s += r * r
    end
    return s / n
end

"""
    kernel_reg(y, x; method=:ll, degree=1, bw=:cv, kernel=:gaussian)

Nonparametric regression of `y` on a scalar `x`.

- `method=:nw` — Nadaraya–Watson (local constant): `m̂(x₀) = Σ K((xᵢ−x₀)/h) yᵢ / Σ K((xᵢ−x₀)/h)`.
- `method=:ll` — local linear (Fan–Gijbels); has automatic boundary-bias
  correction that Nadaraya–Watson lacks.
- `method=:lp` — local polynomial of the given `degree`.

Bandwidth `bw`: `:cv` (leave-one-out cross-validation over a bandwidth grid),
`:rot` (rule of thumb = Silverman scale of `x`), or a positive real value.

Pointwise standard errors use the effective-weight sandwich form
`Var(m̂(x₀)) = σ̂²·‖ℓ(x₀)‖²`, with `m̂(x₀) = Σ ℓᵢ(x₀) yᵢ` and `σ̂²` the residual
variance on effective degrees of freedom. Returns [`KernelRegression`](@ref);
the fit is evaluated at the sorted design points.
"""
function kernel_reg(y::AbstractVector{<:Real}, x::AbstractVector{<:Real};
                    method::Symbol=:ll, degree::Int=1,
                    bw::Union{Symbol,Real}=:cv, kernel::Symbol=:gaussian)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    T = promote_type(float(eltype(x)), float(eltype(y)))
    T = T <: AbstractFloat ? T : Float64
    n = length(x)
    n >= 3 || throw(ArgumentError("kernel_reg requires at least 3 observations"))
    _np_kernel(zero(T), kernel)  # validate kernel

    deg = method === :nw ? 0 : (method === :ll ? 1 : degree)
    (method in (:nw, :ll, :lp)) || throw(ArgumentError("unknown method :$method (use :nw, :ll, :lp)"))
    deg >= 0 || throw(ArgumentError("degree must be ≥ 0"))

    perm = sortperm(x)
    xs = collect(T, @view x[perm])
    ys = collect(T, @view y[perm])

    # --- bandwidth selection ---
    local h::T
    bw_method::Symbol = :user
    if bw isa Symbol
        href = _bw_silverman(xs)
        if bw === :rot
            h = href; bw_method = :rot
        elseif bw === :cv
            grid = href .* collect(range(T(0.25), T(3.0); length=30))
            best_h = href; best_cv = T(Inf)
            for hc in grid
                cv = _kreg_cv(xs, ys, hc, deg, kernel)
                if cv < best_cv
                    best_cv = cv; best_h = hc
                end
            end
            h = best_h; bw_method = :cv
        else
            throw(ArgumentError("unknown bandwidth rule :$bw (use :cv, :rot, or a real value)"))
        end
    else
        h = T(bw); h > 0 || throw(ArgumentError("numeric bandwidth must be positive"))
    end

    # --- fit + variability bands ---
    fitted, hii, l2 = _kreg_fit_at_design(xs, ys, h, deg, kernel)
    resid = ys .- fitted
    rss = dot(resid, resid)
    trH = sum(hii)
    dfres = max(T(n) - trH, one(T))
    sigma2 = rss / dfres
    se = sqrt.(sigma2 .* l2)

    return KernelRegression{T}(xs, fitted, se, xs, ys, h, method, deg, kernel,
                               bw_method, sigma2, n)
end
