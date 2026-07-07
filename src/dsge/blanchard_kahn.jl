# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Blanchard-Kahn (1980) solver for linear RE models, implemented via the
companion-QZ quadratic matrix equation.

Solves: Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t
Returns: y_t = G1 * y_{t-1} + impact * eps_t + C_sol

The system is recast as `f_lead·G² + f_0·G + f_1 = 0` and solved via
`_solve_qz_quadratic`, which sizes determinacy by the 2n companion pencil's
stable-root count — correct for forward-looking, backward-looking, and mixed
models.

Reference:
Blanchard, Olivier J., and Charles M. Kahn. 1980. "The Solution of Linear
Difference Models under Rational Expectations." Econometrica 48 (5): 1305-1311.
"""

"""
    blanchard_kahn(ld::LinearDSGE{T}, spec::DSGESpec{T}; div=1.0+1e-8) → DSGESolution{T}

Solve the linearized DSGE model via the Blanchard-Kahn method.

Uses the companion-QZ quadratic matrix equation approach via `_solve_qz_quadratic`,
which correctly counts stable roots from the 2n companion pencil (including
forward-looking roots). `eu=[1,1]` signals existence and uniqueness (determinacy).

# Arguments
- `ld`   — `LinearDSGE{T}` from `linearize(spec)`
- `spec` — `DSGESpec{T}` (must have `steady_state` set)

# Keywords
- `div::Real=1.0+1e-8` — stable/unstable boundary for eigenvalue sorting

# Returns
`DSGESolution{T}` with `method=:blanchard_kahn`; `eu=[1,1]` signals determinacy.
"""
function blanchard_kahn(ld::LinearDSGE{T}, spec::DSGESpec{T}; div::Real=1.0 + 1e-8) where {T<:AbstractFloat}
    n = spec.n_endog

    f_0 = ld.Gamma0
    f_1 = -ld.Gamma1
    f_ε = -ld.Psi
    f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)

    res = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=div)
    G1 = res.G
    impact = res.impact

    # y_ss = (f₀+f₁+f_lead)⁻¹·C — include the lead block (Γ0-Γ1 = f₀+f₁ omits it), else the
    # SS is wrong for forward-looking models with a constant (audit S-06 / #114).
    C_sol = if norm(ld.C) > eps(T)
        y_bar = real(Vector{T}(complex(f_0 + f_1 + f_lead) \ complex(ld.C)))
        Vector{T}((I - G1) * y_bar)
    else
        zeros(T, n)
    end

    eigenvalues = Vector{ComplexF64}(eigvals(G1))
    DSGESolution{T}(G1, impact, C_sol, res.eu, :blanchard_kahn, eigenvalues, spec, ld)
end
