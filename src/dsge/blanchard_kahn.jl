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
Blanchard-Kahn (1980) eigenvalue counting solver for linear RE models.

Solves the system by decomposing A = Gamma0^{-1} * Gamma1 and checking the
eigenvalue counting condition: n_unstable eigenvalues == n_forward-looking variables.
"""

"""
    blanchard_kahn(ld::LinearDSGE{T}, spec::DSGESpec{T}) -> DSGESolution{T}

Solve the linearized DSGE model via the Blanchard-Kahn method.

The BK condition requires that the number of eigenvalues with modulus > 1
equals the number of forward-looking (non-predetermined) variables.
"""
function blanchard_kahn(ld::LinearDSGE{T}, spec::DSGESpec{T}) where {T<:AbstractFloat}
    n = spec.n_endog
    n_eps = spec.n_exog
    n_fwd = size(ld.Pi, 2)  # number of forward-looking/expectation errors
    eu = [0, 0]

    Gamma0 = ld.Gamma0
    Gamma1 = ld.Gamma1
    Psi = ld.Psi
    C = ld.C

    # Generalized Schur (QZ) decomposition: Q'*Gamma0*Z = S, Q'*Gamma1*Z = T
    # Generalized eigenvalues are T_ii / S_ii
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

    # Count unstable eigenvalues (|lambda| > 1)
    n_unstable = count(m -> m > 1.0, gev_mag)
    n_stable = n - n_unstable

    # BK condition check
    if n_unstable == n_fwd
        eu = [1, 1]  # existence and uniqueness
    elseif n_unstable < n_fwd
        eu = [1, 0]  # exists but indeterminate (multiple solutions)
    else
        eu = [0, 0]  # no stable solution (explosive)
    end

    # Reorder: stable eigenvalues first
    stable_select = BitVector(gev_mag .<= 1.0)
    F_ordered = ordschur(F, stable_select)

    # Recompute eigenvalues after reordering
    eigenvalues = Vector{ComplexF64}(undef, n)
    for i in 1:n
        if abs(F_ordered.S[i,i]) > eps(T)
            eigenvalues[i] = F_ordered.T[i,i] / F_ordered.S[i,i]
        else
            eigenvalues[i] = complex(T(Inf))
        end
    end

    Z = F_ordered.Z
    Q = F_ordered.Q
    Qp = Q'
    S = F_ordered.S
    TT = F_ordered.T

    # Build solution matrices (same algebra as gensys)
    if n_stable > 0
        Z1 = Z[:, 1:n_stable]
        S11 = S[1:n_stable, 1:n_stable]
        T11 = TT[1:n_stable, 1:n_stable]
        Q1 = Qp[1:n_stable, :]

        # G1 = Z1 * S11^{-1} * T11 * Z1'
        G1_c = Z1 * (S11 \ T11) * Z1'

        # impact = Z1 * S11^{-1} * Q1' * Psi
        impact_c = Z1 * (S11 \ (Q1 * complex(Psi)))

        G1 = real(Matrix{T}(G1_c))
        impact = real(Matrix{T}(impact_c))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, n_eps)
    end

    # Solve for constants: C_sol = (I - G1)^{-1} * C
    C_sol = if norm(C) > eps(T)
        real(Vector{T}((I - complex(G1)) \ complex(C)))
    else
        zeros(T, n)
    end

    DSGESolution{T}(G1, impact, C_sol, eu, :blanchard_kahn, eigenvalues, spec, ld)
end
