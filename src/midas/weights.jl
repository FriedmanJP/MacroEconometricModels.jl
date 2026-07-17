# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
MIDAS weight functions and their analytic Jacobians.

`_midas_weights(theta, K, kind)` returns a length-`K` vector `w(θ)` normalized
to `sum(w) == 1`. Supported kinds:

- `:expalmon` — exponential Almon, `wₖ ∝ exp(θ₁k + θ₂k²)`, k=1..K.
- `:beta2`    — two-parameter Beta, `wₖ ∝ x^{θ₁−1}(1−x)^{θ₂−1}` on a guarded (0,1) grid.
- `:beta3`    — three-parameter Beta, adds a nonnegative level constant `θ₃`.
- `:almon`    — polynomial Almon of degree `d = length(θ)−1` (then normalized).
- `:umidas`   — placeholder equal weights (U-MIDAS is estimated by plain OLS).

`_midas_weights_jac(theta, K, kind)` returns the `K×p` Jacobian `∂w/∂θ`,
analytic for `:expalmon`/`:beta2`/`:beta3`, `ForwardDiff` fallback otherwise.
This helper is imported by the GARCH-MIDAS module (EV-02).
"""

# =============================================================================
# Grid helper
# =============================================================================

"""
    _beta_grid(K, T) -> Vector{T}

Return the length-`K` grid on (0,1) used by the Beta weight, guarded away from
the endpoints so `0^negative` never occurs. `x_k = (k−1)/(K−1)` clamped into
`[δ, 1−δ]` with `δ = 1e-8`.
"""
function _beta_grid(K::Int, ::Type{T}) where {T<:Real}
    K >= 2 || throw(ArgumentError("Beta weights require K ≥ 2 (got K=$K)"))
    delta = T(1e-8)
    x = Vector{T}(undef, K)
    @inbounds for k in 1:K
        xk = (T(k) - one(T)) / (T(K) - one(T))
        x[k] = clamp(xk, delta, one(T) - delta)
    end
    x
end

# =============================================================================
# Weight functions
# =============================================================================

"""
    _midas_weights(theta, K, kind) -> Vector{T}

Length-`K` normalized MIDAS weight vector (sums to 1). See module docstring.
"""
function _midas_weights(theta::AbstractVector{T}, K::Int, kind::Symbol) where {T<:Real}
    K >= 1 || throw(ArgumentError("K must be ≥ 1 (got K=$K)"))
    if kind === :expalmon
        length(theta) == 2 || throw(ArgumentError(":expalmon expects 2 parameters"))
        k = T.(1:K)
        z = theta[1] .* k .+ theta[2] .* (k .^ 2)
        z = z .- maximum(z)                      # stabilize; exact at θ=0 (all zeros ⇒ 1/K)
        u = exp.(z)
        return u ./ sum(u)
    elseif kind === :beta2 || kind === :beta3
        x = _beta_grid(K, T)
        a = theta[1]; b = theta[2]
        u = (x .^ (a - one(T))) .* ((one(T) .- x) .^ (b - one(T)))
        if kind === :beta3
            length(theta) == 3 || throw(ArgumentError(":beta3 expects 3 parameters"))
            u = u .+ theta[3]
        else
            length(theta) == 2 || throw(ArgumentError(":beta2 expects 2 parameters"))
        end
        return u ./ sum(u)
    elseif kind === :almon
        length(theta) >= 1 || throw(ArgumentError(":almon expects ≥ 1 parameter"))
        k = T.(1:K)
        u = zeros(T, K)
        @inbounds for (j, tj) in enumerate(theta)
            u .+= tj .* (k .^ (j - 1))
        end
        return u ./ sum(u)
    elseif kind === :umidas
        return fill(one(T) / T(K), K)
    else
        throw(ArgumentError("unknown MIDAS weight kind: $kind"))
    end
end

# =============================================================================
# Analytic Jacobians
# =============================================================================

"""
    _normalize_jac(u, du) -> Matrix

Given an unnormalized weight vector `u` (length K) and its Jacobian `du`
(`K×p`, `∂uₖ/∂θⱼ`), return the Jacobian of the normalized weights
`wₖ = uₖ/Σu`:

    ∂wₖ/∂θⱼ = (1/S)·(duₖⱼ − wₖ Σᵢ duᵢⱼ),  S = Σu.
"""
function _normalize_jac(u::AbstractVector{T}, du::AbstractMatrix{T}) where {T<:Real}
    S = sum(u)
    w = u ./ S
    colsum = vec(sum(du, dims=1))            # length p
    return (du .- w * colsum') ./ S          # K×p
end

"""
    _midas_weights_jac(theta, K, kind) -> Matrix{T}

`K×p` Jacobian `∂w/∂θ`. Analytic for `:expalmon`/`:beta2`/`:beta3`;
`ForwardDiff.jacobian` fallback for `:almon` (and any other kind).
"""
function _midas_weights_jac(theta::AbstractVector{T}, K::Int, kind::Symbol) where {T<:AbstractFloat}
    if kind === :expalmon
        k = T.(1:K)
        z = theta[1] .* k .+ theta[2] .* (k .^ 2)
        z = z .- maximum(z)
        u = exp.(z)
        du = hcat(u .* k, u .* (k .^ 2))     # ∂u/∂θ₁ = u·k, ∂u/∂θ₂ = u·k²
        return _normalize_jac(u, du)
    elseif kind === :beta2 || kind === :beta3
        x = _beta_grid(K, T)
        a = theta[1]; b = theta[2]
        base = (x .^ (a - one(T))) .* ((one(T) .- x) .^ (b - one(T)))
        d_a = base .* log.(x)                 # ∂base/∂a
        d_b = base .* log.(one(T) .- x)       # ∂base/∂b
        if kind === :beta3
            u = base .+ theta[3]
            du = hcat(d_a, d_b, ones(T, K))   # ∂u/∂θ₃ = 1
            return _normalize_jac(u, du)
        else
            u = base
            du = hcat(d_a, d_b)
            return _normalize_jac(u, du)
        end
    else
        # :almon and any other kind — ForwardDiff fallback
        return ForwardDiff.jacobian(t -> _midas_weights(t, K, kind), collect(theta))
    end
end

# =============================================================================
# Default starting values for multi-start NLS
# =============================================================================

"""
    _midas_theta_starts(kind, poly_degree) -> Vector{Vector{Float64}}

Documented multi-start grid for the profiled-SSR minimization. The exp-Almon and
Beta profiled objectives have flat ridges, so several starts guard against
stalling in a local minimum.
"""
function _midas_theta_starts(kind::Symbol, poly_degree::Int)
    if kind === :expalmon
        return [[0.0, 0.0], [0.1, -0.01], [-0.1, -0.02], [0.5, -0.05], [-0.5, -0.1]]
    elseif kind === :beta2
        return [[1.0, 3.0], [1.0, 5.0], [2.0, 2.0], [1.0, 1.5], [3.0, 3.0]]
    elseif kind === :beta3
        return [[1.0, 3.0, 0.0], [1.0, 5.0, 0.0], [2.0, 2.0, 0.01], [1.0, 1.5, 0.0]]
    elseif kind === :almon
        d = poly_degree + 1
        base = zeros(d); base[1] = 1.0       # ⇒ equal weights at start
        s2 = zeros(d); s2[1] = 1.0; d >= 2 && (s2[2] = -0.1)
        return [copy(base), s2]
    else
        throw(ArgumentError("no NLS starts for kind $kind"))
    end
end

"""Number of weight parameters for a restricted MIDAS `kind`."""
function _n_theta(kind::Symbol, poly_degree::Int)
    kind === :expalmon && return 2
    kind === :beta2    && return 2
    kind === :beta3    && return 3
    kind === :almon    && return poly_degree + 1
    kind === :umidas   && return 0
    throw(ArgumentError("unknown MIDAS weight kind: $kind"))
end
