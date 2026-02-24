# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Numerical higher-order derivatives for DSGE residual functions.

Computes Hessians (second derivatives) and third-derivative tensors of the residual
functions `f_i(y_t, y_{t-1}, y_{t+1}, ε, θ)` via central finite differences.

The derivative dimensions correspond to the four argument slots:
- `:current`  — y_t      (n variables)
- `:lag`      — y_{t-1}  (n variables)
- `:lead`     — y_{t+1}  (n variables)
- `:shock`    — ε_t      (n_ε shocks)
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
    n = spec.n_endog
    θ = spec.param_values
    n_ε = spec.n_exog
    ε_zero = zeros(T, n_ε)

    dim1 = _slot_dim(spec, which1)
    dim2 = _slot_dim(spec, which2)

    H = zeros(T, n, dim1, dim2)

    for j in 1:dim1
        hj = _step_size_hessian(T, y_ss, which1, j)
        for k in 1:dim2
            hk = _step_size_hessian(T, y_ss, which2, k)

            # Four perturbed argument tuples
            args_pp = _make_args_two(y_ss, ε_zero, which1, j, +hj, which2, k, +hk)
            args_pm = _make_args_two(y_ss, ε_zero, which1, j, +hj, which2, k, -hk)
            args_mp = _make_args_two(y_ss, ε_zero, which1, j, -hj, which2, k, +hk)
            args_mm = _make_args_two(y_ss, ε_zero, which1, j, -hj, which2, k, -hk)

            inv_4hh = one(T) / (T(4) * hj * hk)

            for i in 1:n
                fn = spec.residual_fns[i]
                f_pp = fn(args_pp..., θ)
                f_pm = fn(args_pm..., θ)
                f_mp = fn(args_mp..., θ)
                f_mm = fn(args_mm..., θ)
                H[i, j, k] = (f_pp - f_pm - f_mp + f_mm) * inv_4hh
            end
        end
    end

    H
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
    slots = [:current, :lag, :lead, :shock]
    result = Dict{Tuple{Symbol,Symbol}, Array{T,3}}()

    for (a, s1) in enumerate(slots)
        for b in a:length(slots)
            s2 = slots[b]
            result[(s1, s2)] = _compute_hessian(spec, y_ss, s1, s2)
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
    n = spec.n_endog
    θ = spec.param_values
    n_ε = spec.n_exog
    ε_zero = zeros(T, n_ε)

    dim1 = _slot_dim(spec, w1)
    dim2 = _slot_dim(spec, w2)
    dim3 = _slot_dim(spec, w3)

    D3 = zeros(T, n, dim1, dim2, dim3)

    signs = (T(-1), T(1))

    for j in 1:dim1
        hj = _step_size_third(T, y_ss, w1, j)
        for k in 1:dim2
            hk = _step_size_third(T, y_ss, w2, k)
            for l in 1:dim3
                hl = _step_size_third(T, y_ss, w3, l)

                inv_8hhh = one(T) / (T(8) * hj * hk * hl)

                # Evaluate all 8 sign combinations
                for s1 in signs
                    for s2 in signs
                        for s3 in signs
                            args = _make_args_three(y_ss, ε_zero,
                                                    w1, j, s1 * hj,
                                                    w2, k, s2 * hk,
                                                    w3, l, s3 * hl)
                            coeff = s1 * s2 * s3 * inv_8hhh
                            for i in 1:n
                                fn = spec.residual_fns[i]
                                D3[i, j, k, l] += coeff * fn(args..., θ)
                            end
                        end
                    end
                end
            end
        end
    end

    D3
end
