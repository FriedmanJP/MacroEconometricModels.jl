# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Klein (2000) generalized Schur decomposition solver for linear RE models.

Solves: Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t
Returns: y_t = G1 * y_{t-1} + impact * eps_t + C_sol

Uses the QZ decomposition of the pencil (Gamma0, Gamma1) with eigenvalue
reordering and the Blanchard-Kahn counting condition on predetermined variables.

Reference:
Klein, Paul. 2000. "Using the Generalized Schur Form to Solve a Multivariate
Linear Rational Expectations Model." Journal of Economic Dynamics and Control
24 (10): 1405-1423.
"""

"""
    _count_predetermined(ld::LinearDSGE{T}) → Int

Count the number of predetermined (state) variables by detecting non-zero columns
in Gamma1. A variable is predetermined if it appears with a lag (y[t-1]).
"""
function _count_predetermined(ld::LinearDSGE{T}) where {T}
    n = size(ld.Gamma1, 2)
    tol = eps(T) * T(100)
    count(j -> any(x -> abs(x) > tol, @view(ld.Gamma1[:, j])), 1:n)
end

"""
    _state_control_indices(ld::LinearDSGE{T}) → (state_idx::Vector{Int}, control_idx::Vector{Int})

Partition variables into state (predetermined) and control (jump) indices.
State variables have non-zero columns in Γ₁; the rest are controls.
"""
function _state_control_indices(ld::LinearDSGE{T}) where {T}
    n = size(ld.Gamma1, 2)
    tol = eps(T) * T(100)
    state_idx = Int[]
    control_idx = Int[]
    for j in 1:n
        if any(x -> abs(x) > tol, @view(ld.Gamma1[:, j]))
            push!(state_idx, j)
        else
            push!(control_idx, j)
        end
    end
    (state_idx, control_idx)
end

"""
    klein(ld::LinearDSGE{T}, spec::DSGESpec{T}; div=1.0+1e-8) → DSGESolution{T}

Solve the linear RE model represented by `ld`/`spec` via the companion-QZ quadratic
matrix equation, returning a `DSGESolution{T}`.

The system `Gamma0·y_t = Gamma1·y_{t-1} + C + Psi·ε_t + Pi·η_t` is recast as the
quadratic equation `f_lead·G² + f_0·G + f_1 = 0` and solved via `_solve_qz_quadratic`,
which sizes determinacy by the 2n companion pencil's stable-root count (correct for
forward-looking, backward-looking, and mixed models).

# Arguments
- `ld`   — `LinearDSGE{T}` from `linearize(spec)`
- `spec` — `DSGESpec{T}` (must have `steady_state` set)

# Keywords
- `div::Real=1.0+1e-8` — stable/unstable boundary for eigenvalue sorting

# Returns
`DSGESolution{T}` with `method=:klein`; `eu=[1,1]` signals determinacy.
"""
function klein(ld::LinearDSGE{T}, spec::DSGESpec{T}; div::Real=1.0 + 1e-8) where {T<:AbstractFloat}
    n = spec.n_endog

    f_0 = ld.Gamma0
    f_1 = -ld.Gamma1
    f_ε = -ld.Psi
    f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)

    res = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=div)
    G1 = res.G
    impact = res.impact

    # Constants: C_sol = (I - G1)·y_ss, y_ss = (Γ0 - Γ1)⁻¹·C
    C_sol = if norm(ld.C) > eps(T)
        y_bar = real(Vector{T}((complex(ld.Gamma0) - complex(ld.Gamma1)) \ complex(ld.C)))
        Vector{T}((I - G1) * y_bar)
    else
        zeros(T, n)
    end

    eigenvalues = Vector{ComplexF64}(eigvals(G1))
    DSGESolution{T}(G1, impact, C_sol, res.eu, :klein, eigenvalues, spec, ld)
end
