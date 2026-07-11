# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Analytical moment computation for linear DSGE models via discrete Lyapunov equation.

References:
- Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. Ch. 10, §10.2.
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., & Schorfheide, F. (2016). Solution and
  Estimation Methods for DSGE Models. Handbook of Macroeconomics, 2, 527--724.
"""

using LinearAlgebra

"""
    solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) -> Matrix{T}

Solve the discrete Lyapunov equation: `Sigma = G1 * Sigma * G1' + impact * impact'`.

Uses the smart-doubling (squaring) iteration [`_dlyap_doubling`](@ref) — O(n³ log(1/ε)) — instead
of the dense `(I - G1 ⊗ G1)⁻¹` solve, which is O(n⁶) and forms an n²×n² system.

Returns the unconditional covariance matrix `Sigma` (n x n, symmetric positive semi-definite).

Throws `ArgumentError` if G1 is not stable (max |eigenvalue| >= 1).
"""
function solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(G1, 1)
    size(G1) == (n, n) || throw(ArgumentError("G1 must be square, got $(size(G1))"))
    size(impact, 1) == n || throw(ArgumentError("impact must have $n rows, got $(size(impact, 1))"))

    # Check stability — a unit-root/explosive G1 has no unconditional covariance.
    max_eig = maximum(abs.(eigvals(G1)); init=zero(T))
    max_eig >= one(T) && throw(ArgumentError(
        "G1 is not stable (max |eigenvalue| = $(max_eig)). Lyapunov equation has no solution."))

    Sigma = _dlyap_doubling(G1, impact * impact')
    return (Sigma + Sigma') / 2   # enforce exact symmetry
end

# Float64 fallback
solve_lyapunov(G1::AbstractMatrix{<:Real}, impact::AbstractMatrix{<:Real}) =
    solve_lyapunov(Float64.(G1), Float64.(impact))

"""
    _lyapunov_from_cov(A, C) -> P

Internal: solve the discrete Lyapunov equation `A P A' - P + C = 0` for a **stable**
`A` (all `|eig(A)| < 1`) given the covariance-form driving term `C` (as opposed to
`solve_lyapunov`, which takes an impact matrix and forms `C = impact*impact'`). Used
for the stationary subspace inside `_diffuse_initial_covariance`.
"""
function _lyapunov_from_cov(A::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    n == 0 && return zeros(T, 0, 0)
    M = Matrix{T}(I, n*n, n*n) - kron(A, A)
    P = reshape(M \ vec(C), n, n)
    return (P + P') / 2
end

"""
    _diffuse_initial_covariance(G1, RQR; kappa=1e6, tol=1e-6) -> P0

Internal: eigenvalue-partitioned diffuse initial state covariance for the Kalman
filter/smoother when the transition matrix `G1` has unit roots (Durbin & Koopman
2012, ch. 5; Dynare `lik_init=3`). Detects nonstationary directions (Schur
eigenvalues with modulus `≥ 1 - tol`), applies a large diffuse variance `kappa` on
that subspace and the finite stationary Lyapunov covariance (driven by `RQR = R Q R'`)
on the complement. If `G1` is fully stable this reduces to the stationary solution.

This is the finite-`kappa` approximation to exact diffuse initialization (the exact
diffuse recursion is deferred to the consolidated Kalman kernel, [T147]); it replaces
the old scale-dependent `P0 = 10*I` fallback and emits a one-time warning.
"""
function _diffuse_initial_covariance(G1::AbstractMatrix{T}, RQR::AbstractMatrix{T};
                                      kappa::T=T(1e6), tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = size(G1, 1)
    n == 0 && return zeros(T, 0, 0)
    Sch = schur(Matrix{T}(G1))
    nonstat = abs.(Sch.values) .>= (one(T) - tol)
    d = count(nonstat)
    if d == 0
        return _lyapunov_from_cov(G1, RQR)
    end
    @warn "Diffuse Kalman init: $d nonstationary direction(s) (|eig|≥1); κ=$(kappa) diffuse " *
          "prior on that subspace (P0=10*I removed; exact diffuse à la Durbin-Koopman 2012 " *
          "ch.5 deferred to [T147])." maxlog=1
    ordschur!(Sch, nonstat)           # nonstationary block first
    U = Sch.Z
    Qrot = U' * RQR * U
    Prot = zeros(T, n, n)
    for i in 1:d
        Prot[i, i] = kappa
    end
    if d < n
        S22 = Sch.T[d+1:n, d+1:n]
        Q22 = Qrot[d+1:n, d+1:n]
        Q22 = (Q22 + Q22') / 2
        Prot[d+1:n, d+1:n] .= _lyapunov_from_cov(S22, Q22)
    end
    P0 = U * Prot * U'
    return (P0 + P0') / 2
end

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
