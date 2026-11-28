# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Auto-linearization of DSGE models via numerical Jacobians.

Produces the Sims canonical form: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
"""

"""Internal helper: linearize `spec` and also return the lead Jacobian `f_lead` to avoid recomputing it."""
function _linearize_with_lead(spec::ModelSpec{T}) where {T<:AbstractFloat}
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    n = spec.n_endog
    n_ε = spec.n_exog
    n_η = spec.n_expect  # #223 ([T124]): distinct lead variables == #η shocks == size(Π, 2)
    y_ss = spec.steady_state
    θ = spec.param_values
    ε_zero = zeros(T, n_ε)

    # Compute numerical Jacobians via central differences (each computed once; #947)
    f_0 = _dsge_jacobian(spec, y_ss, :current)     # n × n
    f_1 = _dsge_jacobian(spec, y_ss, :lag)          # n × n
    f_lead = _dsge_jacobian(spec, y_ss, :lead)      # n × n
    f_ε = _dsge_jacobian_shocks(spec, y_ss)         # n × n_ε

    Gamma0 = f_0                            # n × n
    Gamma1 = -f_1                           # n × n
    Psi = -f_ε                              # n × n_ε

    if spec.linear
        resid_ss = zeros(T, n)
        for i in 1:n
            resid_ss[i] = spec.residual_fns[i](y_ss, y_ss, y_ss, ε_zero, θ)
        end
        C = -resid_ss
    else
        C = zeros(T, n)
    end

    # Π: select columns of -f_lead corresponding to forward-looking variables (reusing f_lead; #947)
    if n_η > 0
        fwd_var_indices = _forward_variable_indices(spec, f_lead)
        Pi = -f_lead[:, fwd_var_indices]    # n × n_η
    else
        Pi = zeros(T, n, 0)
    end

    return LinearDSGE{T}(Gamma0, Gamma1, C, Psi, Pi, spec), f_lead
end

"""
    linearize(spec::ModelSpec{T}) → LinearDSGE{T}

Linearize a DSGE model around its steady state using numerical Jacobians.

The model `f(y_t, y_{t-1}, y_{t+1}, ε, θ) = 0` is expanded to first order.
Rearranging into Sims form: `Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t`.

Requires `compute_steady_state` to have been called first.
"""
linearize(spec::ModelSpec{T}) where {T<:AbstractFloat} = first(_linearize_with_lead(spec))

"""Compute Jacobian of residual vector w.r.t. y_t, y_{t-1}, or y_{t+1}."""
function _dsge_jacobian(spec::ModelSpec{T}, y_ss::Vector{T}, which::Symbol) where {T}
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
function _dsge_jacobian_shocks(spec::ModelSpec{T}, y_ss::Vector{T}) where {T}
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
function _forward_variable_indices(spec::ModelSpec{T}, f_lead::AbstractMatrix{T}) where {T}
    n = spec.n_endog
    fwd_vars = Int[]
    for j in 1:n
        if any(abs(f_lead[i, j]) > T(1e-10) for i in 1:n)
            push!(fwd_vars, j)
        end
    end
    return fwd_vars
end

function _forward_variable_indices(spec::ModelSpec{T}) where {T}
    f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)
    return _forward_variable_indices(spec, f_lead)
end
