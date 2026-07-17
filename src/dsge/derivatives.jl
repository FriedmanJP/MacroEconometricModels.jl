# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Exact higher-order derivatives for DSGE residual functions.

Computes Hessians (second derivatives) and third-derivative tensors of the residual
functions `f_i(y_t, y_{t-1}, y_{t+1}, ε, θ)` via **nested ForwardDiff** on the stacked map
`R(v)`, `v = [y_t; y_lag; y_lead; ε]`. This is machine-accurate (no finite-difference step
error) and builds one full `n × N × N` (resp. `n × N × N × N`) tensor per order that the
per-slot blocks are sliced from.

The derivative dimensions correspond to the four argument slots:
- `:current`  — y_t      (n variables)
- `:lag`      — y_{t-1}  (n variables)
- `:lead`     — y_{t+1}  (n variables)
- `:shock`    — ε_t      (n_ε shocks)

`_step_size_*` and `_make_args*` are retained for the finite-difference helpers exercised by
the coverage tests; they are no longer used by the derivative computation itself.
"""

# =============================================================================
# Helpers — argument slot dimensions and perturbation application
# =============================================================================

"""
    _slot_dim(spec::DSGESpec, which::Symbol) → Int

Return the dimension of the given argument slot.
"""
function _slot_dim(spec::DSGESpec{T}, which::Symbol) where {T}
    if which == :shock
        return spec.n_exog
    else
        return spec.n_endog
    end
end

"""
    _slot_offset(spec, which) → Int

Offset of the given argument slot within the stacked vector `v = [y_t; y_lag; y_lead; ε]`.
"""
function _slot_offset(spec::DSGESpec, which::Symbol)
    n = spec.n_endog
    which === :current ? 0 :
    which === :lag     ? n :
    which === :lead    ? 2n : 3n   # :shock
end

"""
    _stacked_residual(spec) → (v -> R(v))

Stacked residual map `R(v) = [f_i(y_t, y_lag, y_lead, ε, θ)]` with `v = [y_t; y_lag; y_lead; ε]`,
generic in `eltype(v)` so it can be differentiated by (nested) ForwardDiff.
"""
function _stacked_residual(spec::DSGESpec{T}) where {T}
    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    fns = spec.residual_fns
    return function (v)
        y_t    = v[1:n]
        y_lag  = v[(n + 1):(2n)]
        y_lead = v[(2n + 1):(3n)]
        ε      = v[(3n + 1):(3n + n_ε)]
        [fns[i](y_t, y_lag, y_lead, ε, θ) for i in 1:n]
    end
end

"""
    _full_hessian(spec, y_ss) → Array{T,3}    # n × N × N,  N = 3·n_endog + n_exog

Exact second-derivative tensor `∂²R_i/∂v_a ∂v_b` of the stacked residual via nested ForwardDiff.
"""
function _full_hessian(spec::DSGESpec{T}, y_ss::AbstractVector{T}) where {T}
    n = spec.n_endog
    N = 3n + spec.n_exog
    R = _stacked_residual(spec)
    v0 = vcat(y_ss, y_ss, y_ss, zeros(T, spec.n_exog))
    Hflat = ForwardDiff.jacobian(v -> vec(ForwardDiff.jacobian(R, v)), v0)   # (n·N) × N
    reshape(Hflat, n, N, N)
end

"""
    _full_third(spec, y_ss) → Array{T,4}      # n × N × N × N

Exact third-derivative tensor `∂³R_i/∂v_a ∂v_b ∂v_c` via triply-nested ForwardDiff.
"""
function _full_third(spec::DSGESpec{T}, y_ss::AbstractVector{T}) where {T}
    n = spec.n_endog
    N = 3n + spec.n_exog
    R = _stacked_residual(spec)
    v0 = vcat(y_ss, y_ss, y_ss, zeros(T, spec.n_exog))
    D3flat = ForwardDiff.jacobian(
        v -> vec(ForwardDiff.jacobian(w -> vec(ForwardDiff.jacobian(R, w)), v)), v0)  # (n·N²) × N
    reshape(D3flat, n, N, N, N)
end

"""
    _step_size_hessian(::Type{T}, y_ss::Vector{T}, which::Symbol, j::Int) → T

Compute the step size for Hessian computation. Uses adaptive scaling for
variable dimensions and a fixed step for shocks.
"""
function _step_size_hessian(::Type{T}, y_ss::Vector{T}, which::Symbol, j::Int) where {T}
    if which == :shock
        return T(1e-5)
    else
        return max(T(1e-5), T(1e-5) * abs(y_ss[j]))
    end
end

"""
    _step_size_third(::Type{T}, y_ss::Vector{T}, which::Symbol, j::Int) → T

Compute the step size for third-derivative computation.
"""
function _step_size_third(::Type{T}, y_ss::Vector{T}, which::Symbol, j::Int) where {T}
    if which == :shock
        return T(1e-4)
    else
        return max(T(1e-4), T(1e-4) * abs(y_ss[j]))
    end
end

"""
    _make_args(y_ss, ε_zero, which, j, h) → (y_t, y_lag, y_lead, ε)

Build the 4-tuple of arguments with a perturbation of size `h` applied to
element `j` of the `which` slot. All other slots stay at steady state / zero.
"""
function _make_args(y_ss::Vector{T}, ε_zero::Vector{T},
                    which::Symbol, j::Int, h::T) where {T}
    y_t = copy(y_ss)
    y_lag = copy(y_ss)
    y_lead = copy(y_ss)
    ε = copy(ε_zero)

    if which == :current
        y_t[j] += h
    elseif which == :lag
        y_lag[j] += h
    elseif which == :lead
        y_lead[j] += h
    else  # :shock
        ε[j] += h
    end
    (y_t, y_lag, y_lead, ε)
end

"""
    _make_args_two(y_ss, ε_zero, w1, j1, h1, w2, j2, h2) → (y_t, y_lag, y_lead, ε)

Build the 4-tuple with perturbations applied to TWO slots simultaneously.
When both perturbations target the same slot, both are applied additively.
"""
function _make_args_two(y_ss::Vector{T}, ε_zero::Vector{T},
                        w1::Symbol, j1::Int, h1::T,
                        w2::Symbol, j2::Int, h2::T) where {T}
    y_t = copy(y_ss)
    y_lag = copy(y_ss)
    y_lead = copy(y_ss)
    ε = copy(ε_zero)

    # Apply first perturbation
    if w1 == :current
        y_t[j1] += h1
    elseif w1 == :lag
        y_lag[j1] += h1
    elseif w1 == :lead
        y_lead[j1] += h1
    else
        ε[j1] += h1
    end

    # Apply second perturbation
    if w2 == :current
        y_t[j2] += h2
    elseif w2 == :lag
        y_lag[j2] += h2
    elseif w2 == :lead
        y_lead[j2] += h2
    else
        ε[j2] += h2
    end

    (y_t, y_lag, y_lead, ε)
end

"""
    _make_args_three(y_ss, ε_zero, w1, j1, h1, w2, j2, h2, w3, j3, h3)

Build the 4-tuple with perturbations applied to THREE slots simultaneously.
"""
function _make_args_three(y_ss::Vector{T}, ε_zero::Vector{T},
                          w1::Symbol, j1::Int, h1::T,
                          w2::Symbol, j2::Int, h2::T,
                          w3::Symbol, j3::Int, h3::T) where {T}
    y_t = copy(y_ss)
    y_lag = copy(y_ss)
    y_lead = copy(y_ss)
    ε = copy(ε_zero)

    for (w, j, h) in ((w1, j1, h1), (w2, j2, h2), (w3, j3, h3))
        if w == :current
            y_t[j] += h
        elseif w == :lag
            y_lag[j] += h
        elseif w == :lead
            y_lead[j] += h
        else
            ε[j] += h
        end
    end

    (y_t, y_lag, y_lead, ε)
end

# =============================================================================
# Hessian (second derivatives)
# =============================================================================

"""
    _compute_hessian(spec::DSGESpec{T}, y_ss::Vector{T}, which1::Symbol, which2::Symbol)
        → Array{T, 3}   # n × dim1 × dim2

Compute the Hessian ∂²f_i / ∂a_j ∂b_k for all residual equations i, using the
4-point central difference stencil:

    H[i,j,k] = (f(+h_j,+h_k) - f(+h_j,-h_k) - f(-h_j,+h_k) + f(-h_j,-h_k)) / (4 h_j h_k)

Arguments:
- `which1`, `which2` — argument slots (`:current`, `:lag`, `:lead`, `:shock`)
"""
function _compute_hessian(spec::DSGESpec{T}, y_ss::Vector{T},
                          which1::Symbol, which2::Symbol) where {T}
    Hfull = _full_hessian(spec, y_ss)
    o1 = _slot_offset(spec, which1)
    o2 = _slot_offset(spec, which2)
    d1 = _slot_dim(spec, which1)
    d2 = _slot_dim(spec, which2)
    Hfull[:, (o1 + 1):(o1 + d1), (o2 + 1):(o2 + d2)]
end

"""
    _compute_all_hessians(spec::DSGESpec{T}, y_ss::Vector{T})
        → Dict{Tuple{Symbol,Symbol}, Array{T,3}}

Compute all 10 unique Hessian blocks for the 4 argument slots
{current, lag, lead, shock}. Only the unique pairs (with which1 ≤ which2 in
canonical ordering) are computed; transpose can be obtained by permuting axes 2–3.

Returns a dictionary mapping `(which1, which2) => H` where `H` is `n × dim1 × dim2`.
"""
function _compute_all_hessians(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    Hfull = _full_hessian(spec, y_ss)          # build the full tensor ONCE, slice all blocks
    slots = [:current, :lag, :lead, :shock]
    result = Dict{Tuple{Symbol,Symbol}, Array{T,3}}()

    for (a, s1) in enumerate(slots)
        o1 = _slot_offset(spec, s1)
        d1 = _slot_dim(spec, s1)
        for b in a:length(slots)
            s2 = slots[b]
            o2 = _slot_offset(spec, s2)
            d2 = _slot_dim(spec, s2)
            result[(s1, s2)] = Hfull[:, (o1 + 1):(o1 + d1), (o2 + 1):(o2 + d2)]
        end
    end

    result
end

# =============================================================================
# Third derivatives
# =============================================================================

"""
    _third_derivative(spec::DSGESpec{T}, y_ss::Vector{T},
                      w1::Symbol, w2::Symbol, w3::Symbol)
        → Array{T, 4}   # n × dim1 × dim2 × dim3

Compute the third derivative ∂³f_i / ∂a_j ∂b_k ∂c_l using the 8-point stencil:

    D3[i,j,k,l] = Σ_{s1,s2,s3 ∈ {-1,+1}} s1·s2·s3 · f(s1·h_j, s2·h_k, s3·h_l) / (8 h_j h_k h_l)

Arguments:
- `w1`, `w2`, `w3` — argument slots (`:current`, `:lag`, `:lead`, `:shock`)
"""
function _third_derivative(spec::DSGESpec{T}, y_ss::Vector{T},
                           w1::Symbol, w2::Symbol, w3::Symbol) where {T}
    D3full = _full_third(spec, y_ss)
    o1 = _slot_offset(spec, w1)
    o2 = _slot_offset(spec, w2)
    o3 = _slot_offset(spec, w3)
    d1 = _slot_dim(spec, w1)
    d2 = _slot_dim(spec, w2)
    d3 = _slot_dim(spec, w3)
    D3full[:, (o1 + 1):(o1 + d1), (o2 + 1):(o2 + d2), (o3 + 1):(o3 + d3)]
end
