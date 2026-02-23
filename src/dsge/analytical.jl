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
Analytical moment computation for linear DSGE models via discrete Lyapunov equation.

References:
- Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. Ch. 10.
"""

using LinearAlgebra

"""
    solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) -> Matrix{T}

Solve the discrete Lyapunov equation: `Sigma = G1 * Sigma * G1' + impact * impact'`.

Uses Kronecker vectorization: `vec(Sigma) = (I - G1 kron G1)^{-1} * vec(impact * impact')`.

Returns the unconditional covariance matrix `Sigma` (n x n, symmetric positive semi-definite).

Throws `ArgumentError` if G1 is not stable (max |eigenvalue| >= 1).
"""
function solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(G1, 1)
    size(G1) == (n, n) || throw(ArgumentError("G1 must be square, got $(size(G1))"))
    size(impact, 1) == n || throw(ArgumentError("impact must have $n rows, got $(size(impact, 1))"))

    # Check stability
    max_eig = maximum(abs.(eigvals(G1)))
    max_eig >= one(T) && throw(ArgumentError(
        "G1 is not stable (max |eigenvalue| = $(max_eig)). Lyapunov equation has no solution."))

    Q = impact * impact'
    # Vectorize: vec(Sigma) = (I_n^2 - G1 kron G1)^{-1} * vec(Q)
    n2 = n * n
    A = Matrix{T}(I, n2, n2) - kron(G1, G1)
    sigma_vec = A \ vec(Q)
    Sigma = reshape(sigma_vec, n, n)

    # Enforce exact symmetry
    Sigma = (Sigma + Sigma') / 2
    return Sigma
end

# Float64 fallback
solve_lyapunov(G1::AbstractMatrix{<:Real}, impact::AbstractMatrix{<:Real}) =
    solve_lyapunov(Float64.(G1), Float64.(impact))

"""
    analytical_moments(sol::DSGESolution{T}; lags::Int=1) -> Vector{T}

Compute analytical moment vector from a solved DSGE model.

Uses the discrete Lyapunov equation to compute the unconditional covariance,
then extracts the same moment format as `autocovariance_moments`:

1. Upper-triangle of variance-covariance matrix: k*(k+1)/2 elements
2. Diagonal autocovariances at each lag: k elements per lag

# Arguments
- `sol` -- solved DSGE model (must be stable/determined)
- `lags` -- number of autocovariance lags (default: 1)
"""
function analytical_moments(sol::DSGESolution{T}; lags::Int=1) where {T<:AbstractFloat}
    k = nvars(sol)
    Sigma = solve_lyapunov(sol.G1, sol.impact)

    moments = T[]

    # Upper triangle of variance-covariance matrix (matching autocovariance_moments order)
    for i in 1:k
        for j in i:k
            push!(moments, Sigma[i, j])
        end
    end

    # Autocovariances at each lag: Gamma_h = G1^h * Sigma, extract diagonal
    G1_power = copy(sol.G1)
    for lag in 1:lags
        Gamma_h = G1_power * Sigma
        for i in 1:k
            push!(moments, Gamma_h[i, i])
        end
        G1_power = G1_power * sol.G1
    end

    moments
end
