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
    klein(Gamma0, Gamma1, C, Psi, n_predetermined; div=1.0) → NamedTuple

Solve the linear RE system via the Klein (2000) QZ decomposition method.

The system is in Sims canonical form:
`Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t`

Klein computes the generalized Schur decomposition of the pencil `(Gamma0, Gamma1)`,
reorders eigenvalues so stable roots (|λ| < div) come first, and checks the
Blanchard-Kahn condition: n_stable == n_predetermined.

# Arguments
- `Gamma0` — n × n coefficient on y_t
- `Gamma1` — n × n coefficient on y_{t-1}
- `C` — n × 1 constant vector
- `Psi` — n × n_shocks shock loading matrix
- `n_predetermined` — number of predetermined (state) variables

# Keywords
- `div::Real=1.0` — dividing line for stable vs unstable eigenvalues

# Returns
Named tuple `(G1, impact, C_sol, eu, eigenvalues)` where:
- `G1` — n × n state transition matrix
- `impact` — n × n_shocks impact matrix
- `C_sol` — n × 1 constants
- `eu` — `[existence, uniqueness]`: 1=yes, 0=no
- `eigenvalues` — generalized eigenvalues from QZ decomposition
"""
function klein(Gamma0::AbstractMatrix{T}, Gamma1::AbstractMatrix{T},
               C::AbstractVector{T}, Psi::AbstractMatrix{T},
               n_predetermined::Int;
               div::Real=1.0) where {T<:AbstractFloat}
    n = size(Gamma0, 1)
    eu = [0, 0]

    # QZ decomposition of pencil (Gamma0, Gamma1)
    # Q * Gamma0 * Z = S, Q * Gamma1 * Z = T (upper triangular)
    # Transition eigenvalues: λ_i = T_ii / S_ii (eigenvalues of Gamma0^{-1} * Gamma1)
    F = schur(complex(Gamma0), complex(Gamma1))

    # Compute generalized eigenvalue magnitudes
    gev_mag = zeros(n)
    for i in 1:n
        if abs(F.S[i,i]) > eps(T)
            gev_mag[i] = abs(F.T[i,i] / F.S[i,i])
        else
            gev_mag[i] = Inf
        end
    end

    # Reorder: stable eigenvalues (|λ| < div) first
    stable_select = BitVector(gev_mag .< T(div))
    F_ordered = ordschur(F, stable_select)

    n_stable = count(stable_select)
    n_unstable = n - n_stable

    # Compute eigenvalues after reordering
    eigenvalues = Vector{ComplexF64}(undef, n)
    for i in 1:n
        if abs(F_ordered.S[i,i]) > eps(T)
            eigenvalues[i] = F_ordered.T[i,i] / F_ordered.S[i,i]
        else
            eigenvalues[i] = complex(T(Inf))
        end
    end

    # Blanchard-Kahn condition: n_stable must equal n_predetermined
    if n_stable == n_predetermined
        eu = [1, 1]  # existence and uniqueness
    elseif n_stable > n_predetermined
        eu = [1, 0]  # indeterminate (multiple solutions)
    else
        eu = [0, 0]  # no stable solution (explosive)
    end

    # Extract ordered Schur matrices
    S = F_ordered.S
    TT = F_ordered.T
    Z = F_ordered.Z
    Q = F_ordered.Q
    Qp = Q'  # conjugate transpose

    # Build solution matrices
    if n_stable > 0
        Z1 = Z[:, 1:n_stable]
        S11 = S[1:n_stable, 1:n_stable]
        T11 = TT[1:n_stable, 1:n_stable]
        Q1 = Qp[1:n_stable, :]

        # State transition: G1 = Z1 * S11^{-1} * T11 * Z1'
        G1_c = Z1 * (S11 \ T11) * Z1'

        # Impact: impact = Z1 * S11^{-1} * Q1 * Psi
        impact_c = Z1 * (S11 \ (Q1 * complex(Psi)))

        G1 = real(Matrix{T}(G1_c))
        impact = real(Matrix{T}(impact_c))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, size(Psi, 2))
    end

    # Constants: C_sol = (I - G1)^{-1} * C
    C_sol = if norm(C) > eps(T)
        real(Vector{T}((I - complex(G1)) \ complex(C)))
    else
        zeros(T, n)
    end

    (G1=G1, impact=impact, C_sol=C_sol, eu=eu, eigenvalues=eigenvalues)
end
