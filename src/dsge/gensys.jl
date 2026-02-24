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
Sims (2002) gensys solver for linear rational expectations models.

Solves: Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t
Returns: y_t = G1 * y_{t-1} + impact * eps_t + C_sol
"""

"""
    gensys(Gamma0, Gamma1, C, Psi, Pi; div=1.0+1e-8) -> NamedTuple

Solve the linear RE system via QZ decomposition (Sims 2002).

Returns `(G1, impact, C_sol, eu, eigenvalues)` where:
- `eu = [exist, unique]`: 1=yes, 0=no
- `div` is the dividing line between stable (|lambda| < div) and unstable eigenvalues.
"""
function gensys(Gamma0::AbstractMatrix{T}, Gamma1::AbstractMatrix{T},
                C::AbstractVector{T}, Psi::AbstractMatrix{T}, Pi::AbstractMatrix{T};
                div::Real=1.0 + 1e-8) where {T<:AbstractFloat}
    n = size(Gamma0, 1)
    eu = [0, 0]

    # QZ (generalized Schur) decomposition
    F = schur(complex(Gamma0), complex(Gamma1))
    S = F.S
    TT = F.T

    # Compute generalized eigenvalues |T_ii/S_ii|
    # Select which to put first (stable = |eigenvalue| < div)
    gev_mag = zeros(n)
    for i in 1:n
        if abs(S[i,i]) > eps(T)
            gev_mag[i] = abs(TT[i,i] / S[i,i])
        else
            gev_mag[i] = Inf
        end
    end

    # Reorder: stable eigenvalues first using ordschur
    stable_select = BitVector(gev_mag .< div)
    F_ordered = ordschur(F, stable_select)
    S = F_ordered.S
    TT = F_ordered.T
    Q = F_ordered.Q  # Q such that Q*Gamma0*Z = S, Q*Gamma1*Z = T
    Z = F_ordered.Z

    # Recompute eigenvalues after reordering
    nstab = count(stable_select)
    nunstab = n - nstab

    eigenvalues = Vector{ComplexF64}(undef, n)
    for i in 1:n
        if abs(S[i,i]) > eps(T)
            eigenvalues[i] = TT[i,i] / S[i,i]
        else
            eigenvalues[i] = complex(T(Inf))
        end
    end

    # Partition: Q'*Gamma0*Z = S, Q'*Gamma1*Z = T  (schur convention: Q is unitary)
    # After ordschur: rows 1:nstab are stable, (nstab+1):n are unstable
    # Q is unitary so Q' = inv(Q)
    # We need Q' (conjugate transpose) applied to vectors
    Qp = Q'  # Q' = Q^H

    # Existence check
    if size(Pi, 2) > 0 && nunstab > 0
        Q2Pi = Qp[nstab+1:n, :] * complex(Pi)
        # SVD to check rank
        sv = svd(Q2Pi)
        rank_Q2Pi = count(s -> s > eps(T) * 1e6 * maximum(sv.S), sv.S)

        if rank_Q2Pi >= nunstab
            eu[1] = 1  # existence
        end

        # Uniqueness
        n_fwd = size(Pi, 2)
        if nunstab == n_fwd
            eu[2] = 1
        elseif nunstab < n_fwd
            eu[2] = 1  # over-determined forward block
        end
    else
        # No forward-looking variables or no unstable eigenvalues
        if nunstab == 0
            eu = [1, 1]
        end
    end

    # Build solution matrices
    if nstab > 0
        Z1 = Z[:, 1:nstab]
        S11 = S[1:nstab, 1:nstab]
        T11 = TT[1:nstab, 1:nstab]

        # G1 = Z1 * S11^{-1} * T11 * Z1'
        G1_c = Z1 * (S11 \ T11) * Z1'

        # impact = Z1 * S11^{-1} * Q1' * Psi
        Q1 = Qp[1:nstab, :]
        impact_c = Z1 * (S11 \ (Q1 * complex(Psi)))

        G1 = real(Matrix{T}(G1_c))
        impact = real(Matrix{T}(impact_c))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, size(Psi, 2))
    end

    # Constants
    C_sol = if norm(C) > eps(T)
        real(Vector{T}((I - complex(G1)) \ complex(C)))
    else
        zeros(T, n)
    end

    (G1=G1, impact=impact, C_sol=C_sol, eu=eu, eigenvalues=eigenvalues)
end

"""
    solve(spec::DSGESpec{T}; method=:gensys, kwargs...) -> DSGESolution or PerfectForesightPath or PerturbationSolution

Solve a DSGE model.

# Methods
- `:gensys` -- Sims (2002) QZ decomposition (default)
- `:blanchard_kahn` -- Blanchard-Kahn (1980) eigenvalue counting
- `:klein` -- Klein (2000) generalized Schur decomposition
- `:perturbation` -- Higher-order perturbation (Schmitt-Grohe & Uribe 2004); pass `order=2` for second-order
- `:perfect_foresight` -- deterministic Newton solver
"""
function solve(spec::DSGESpec{T}; method::Symbol=:gensys, kwargs...) where {T<:AbstractFloat}
    # Ensure steady state is computed
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end

    if method == :gensys
        ld = linearize(spec)
        result = gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi)
        return DSGESolution{T}(
            result.G1, result.impact, result.C_sol, result.eu,
            :gensys, result.eigenvalues, spec, ld
        )
    elseif method == :blanchard_kahn
        ld = linearize(spec)
        return blanchard_kahn(ld, spec)
    elseif method == :perfect_foresight
        return perfect_foresight(spec; kwargs...)
    elseif method == :klein
        ld = linearize(spec)
        n_pre = _count_predetermined(ld)
        result = klein(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, n_pre)
        return DSGESolution{T}(
            result.G1, result.impact, result.C_sol, result.eu,
            :klein, result.eigenvalues, spec, ld
        )
    elseif method == :perturbation
        order = get(kwargs, :order, 2)
        return perturbation_solver(spec; order=order)
    else
        throw(ArgumentError("method must be :gensys, :blanchard_kahn, :klein, :perturbation, or :perfect_foresight"))
    end
end
