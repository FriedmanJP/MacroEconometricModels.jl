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
Auto-linearization of DSGE models via numerical Jacobians.

Produces the Sims canonical form: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
"""

"""
    linearize(spec::DSGESpec{T}) → LinearDSGE{T}

Linearize a DSGE model around its steady state using numerical Jacobians.

The model `f(y_t, y_{t-1}, y_{t+1}, ε, θ) = 0` is expanded to first order.
Rearranging into Sims form: `Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t`.

Requires `compute_steady_state` to have been called first.
"""
function linearize(spec::DSGESpec{T}) where {T<:AbstractFloat}
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    n = spec.n_endog
    n_ε = spec.n_exog
    n_η = spec.n_expect
    y_ss = spec.steady_state
    θ = spec.param_values
    ε_zero = zeros(T, n_ε)

    # Compute numerical Jacobians via central differences
    f_0 = _dsge_jacobian(spec, y_ss, :current)     # n × n
    f_1 = _dsge_jacobian(spec, y_ss, :lag)          # n × n
    f_lead = _dsge_jacobian(spec, y_ss, :lead)      # n × n
    f_ε = _dsge_jacobian_shocks(spec, y_ss)         # n × n_ε

    # Sims canonical form: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
    # From the first-order expansion: f_0·ŷ_t + f_1·ŷ_{t-1} + f_lead·ŷ_{t+1} + f_ε·ε = 0
    # Rearranging to isolate ŷ_t: f_0·ŷ_t = -f_1·ŷ_{t-1} - f_ε·ε - f_lead·ŷ_{t+1}
    # Under RE: ŷ_{t+1} = E_t[ŷ_{t+1}] + η_{t+1}, so f_lead·ŷ_{t+1} = f_lead·(…+η)
    # The gensys form Γ₀·y_t = Γ₁·y_{t-1} + Ψ·ε_t + Π·η_t captures:
    # Matching: Γ₀ = f_0, Γ₁ = -f_1, Ψ = -f_ε, Π = -f_lead columns for forward vars

    Gamma0 = f_0                            # n × n
    Gamma1 = -f_1                           # n × n
    C = zeros(T, n)                         # constants (zero at SS)
    Psi = -f_ε                              # n × n_ε

    # Π: select columns of -f_lead corresponding to forward-looking variables
    if n_η > 0
        fwd_var_indices = _forward_variable_indices(spec)
        Pi = -f_lead[:, fwd_var_indices]    # n × n_η
    else
        Pi = zeros(T, n, 0)
    end

    LinearDSGE{T}(Gamma0, Gamma1, C, Psi, Pi, spec)
end

"""Compute Jacobian of residual vector w.r.t. y_t, y_{t-1}, or y_{t+1}."""
function _dsge_jacobian(spec::DSGESpec{T}, y_ss::Vector{T}, which::Symbol) where {T}
    n = spec.n_endog
    θ = spec.param_values
    ε_zero = zeros(T, spec.n_exog)

    J = zeros(T, n, n)
    for j in 1:n
        h = max(T(1e-7), T(1e-7) * abs(y_ss[j]))
        y_plus = copy(y_ss)
        y_minus = copy(y_ss)
        y_plus[j] += h
        y_minus[j] -= h

        for i in 1:n
            fn = spec.residual_fns[i]
            if which == :current
                f_plus = fn(y_plus, y_ss, y_ss, ε_zero, θ)
                f_minus = fn(y_minus, y_ss, y_ss, ε_zero, θ)
            elseif which == :lag
                f_plus = fn(y_ss, y_plus, y_ss, ε_zero, θ)
                f_minus = fn(y_ss, y_minus, y_ss, ε_zero, θ)
            else  # :lead
                f_plus = fn(y_ss, y_ss, y_plus, ε_zero, θ)
                f_minus = fn(y_ss, y_ss, y_minus, ε_zero, θ)
            end
            J[i, j] = (f_plus - f_minus) / (2h)
        end
    end
    J
end

"""Compute Jacobian of residual vector w.r.t. shocks ε."""
function _dsge_jacobian_shocks(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    ε_zero = zeros(T, n_ε)

    J = zeros(T, n, n_ε)
    h = T(1e-7)
    for j in 1:n_ε
        ε_plus = copy(ε_zero)
        ε_minus = copy(ε_zero)
        ε_plus[j] += h
        ε_minus[j] -= h

        for i in 1:n
            fn = spec.residual_fns[i]
            f_plus = fn(y_ss, y_ss, y_ss, ε_plus, θ)
            f_minus = fn(y_ss, y_ss, y_ss, ε_minus, θ)
            J[i, j] = (f_plus - f_minus) / (2h)
        end
    end
    J
end

"""Find indices of endogenous variables that appear with [t+1] in any equation."""
function _forward_variable_indices(spec::DSGESpec{T}) where {T}
    fwd_vars = Set{Int}()
    n = spec.n_endog
    y_ss = spec.steady_state
    θ = spec.param_values
    ε_zero = zeros(T, spec.n_exog)

    # Use the Jacobian approach: any variable where ∂f/∂y_lead is non-zero
    f_lead = _dsge_jacobian(spec, y_ss, :lead)
    for j in 1:n
        if any(abs(f_lead[i, j]) > T(1e-10) for i in 1:n)
            push!(fwd_vars, j)
        end
    end
    sort(collect(fwd_vars))
end
