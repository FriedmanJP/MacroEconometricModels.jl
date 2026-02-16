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
Mountford & Uhlig (2009) penalty function identification for SVAR.

Uses Nelder-Mead optimization over spherical coordinates to find the rotation
matrix Q that best satisfies sign restrictions, with zero restrictions enforced
as hard constraints via Gram-Schmidt orthogonalization.

Reference:
Mountford, A. & Uhlig, H. (2009). "What Are the Effects of Fiscal Policy Shocks?"
Journal of Applied Econometrics 24(6): 960–992.
"""

using LinearAlgebra, Random, Statistics

# =============================================================================
# Result Type
# =============================================================================

"""
    UhligSVARResult{T<:AbstractFloat}

Result from Mountford-Uhlig (2009) penalty function identification.

# Fields
- `Q::Matrix{T}`: Optimal rotation matrix
- `irf::Array{T,3}`: Impulse responses (horizon × n × n)
- `penalty::T`: Total penalty at optimum (negative = better)
- `shock_penalties::Vector{T}`: Per-shock penalty values
- `restrictions::SVARRestrictions`: The imposed restrictions
- `converged::Bool`: Whether all sign restrictions are satisfied
"""
struct UhligSVARResult{T<:AbstractFloat}
    Q::Matrix{T}
    irf::Array{T,3}
    penalty::T
    shock_penalties::Vector{T}
    restrictions::SVARRestrictions
    converged::Bool
end

# =============================================================================
# Spherical Coordinate Helpers
# =============================================================================

"""
Convert (m-1) angles θ ∈ [0, 2π] to a unit vector in R^m using spherical coordinates.

For m=1, returns [1.0]. For m≥2, uses the standard hyperspherical parameterization:
  x_1 = cos(θ_1)
  x_k = cos(θ_k) * prod(sin(θ_j) for j=1:k-1),  k=2,...,m-1
  x_m = prod(sin(θ_j) for j=1:m-1)
"""
function _spherical_to_unit_vector(theta::AbstractVector{T}, m::Int) where {T<:AbstractFloat}
    m == 1 && return ones(T, 1)
    @assert length(theta) == m - 1 "Need $(m-1) angles for R^$m, got $(length(theta))"

    x = zeros(T, m)
    x[1] = cos(theta[1])
    sin_prod = one(T)
    for k in 2:m-1
        sin_prod *= sin(theta[k-1])
        x[k] = cos(theta[k]) * sin_prod
    end
    sin_prod *= sin(theta[m-1])
    x[m] = sin_prod
    x
end

"""
Build column j of Q from angle parameters, enforcing orthogonality to previous
columns and zero restrictions via Gram-Schmidt projection into null space.

Returns a unit vector in the null space of [Q_prev columns; zero constraint rows].
"""
function _uhlig_build_q_column(theta_j::AbstractVector{T}, j::Int, Q_prev::Matrix{T},
                                restrictions::SVARRestrictions,
                                Phi::Vector{Matrix{T}},
                                L::LowerTriangular{T,Matrix{T}},
                                n::Int) where {T<:AbstractFloat}
    # Build constraint matrix: orthogonality to previous columns + zero restrictions
    constraint_rows = Vector{Vector{T}}()

    # Orthogonality constraints from previous columns
    for k in 1:j-1
        push!(constraint_rows, Q_prev[:, k])
    end

    # Zero restriction constraints for shock j
    for zr in restrictions.zeros
        zr.shock == j || continue
        push!(constraint_rows, Vector{T}((Phi[zr.horizon + 1] * L)[zr.variable, :]))
    end

    n_constraints = length(constraint_rows)
    free_dim = n - n_constraints

    # Over-constrained check
    free_dim <= 0 && error("Zero restrictions over-constrain shock $j (n=$n, constraints=$n_constraints)")

    # Find null space basis
    if n_constraints == 0
        # No constraints — full space available
        N = Matrix{T}(I, n, n)
    else
        C = reduce(vcat, [c' for c in constraint_rows])
        svd_result = svd(C, full=true)
        V = transpose(svd_result.Vt)
        tol = max(size(C)...) * eps(T) * (isempty(svd_result.S) ? one(T) : maximum(svd_result.S))
        rank_C = sum(svd_result.S .> tol)
        N = V[:, (rank_C + 1):n]
    end

    # Convert spherical coordinates to unit vector in free_dim space
    if free_dim == 1
        # Only one free dimension — direction is determined (up to sign)
        u = ones(T, 1)
    else
        u = _spherical_to_unit_vector(theta_j, free_dim)
    end

    # Map back to R^n via null space basis
    q = N * u
    q / norm(q)  # Ensure unit norm
end

"""
Build full Q matrix from concatenated angle parameters.

Returns an n×n orthogonal matrix satisfying all zero restrictions.
"""
function _uhlig_build_Q(theta_all::AbstractVector{T}, restrictions::SVARRestrictions,
                         Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                         n::Int) where {T<:AbstractFloat}
    Q = zeros(T, n, n)
    offset = 0

    for j in 1:n
        # Count zero restrictions for shock j
        n_zeros_j = count(zr -> zr.shock == j, restrictions.zeros)
        n_constraints = (j - 1) + n_zeros_j
        free_dim = n - n_constraints
        free_dim <= 0 && error("Zero restrictions over-constrain shock $j")

        n_angles = max(free_dim - 1, 0)
        theta_j = theta_all[offset+1:offset+n_angles]
        offset += n_angles

        Q[:, j] = _uhlig_build_q_column(theta_j, j, Q, restrictions, Phi, L, n)
    end

    Q
end

"""
Count total free angle parameters for the Uhlig penalty function optimization.
"""
function _uhlig_n_params(n::Int, restrictions::SVARRestrictions)
    total = 0
    for j in 1:n
        n_zeros_j = count(zr -> zr.shock == j, restrictions.zeros)
        free_dim = n - (j - 1) - n_zeros_j
        free_dim <= 0 && error("Zero restrictions over-constrain shock $j")
        total += max(free_dim - 1, 0)
    end
    total
end

# =============================================================================
# Penalty Function
# =============================================================================

"""
Uhlig (2009) penalty function.

For each sign restriction, computes the normalized impulse response and assigns:
- Weight 100 if the sign is satisfied (reward)
- Weight 1 if violated (penalty)

The function returns the negative of the total weighted response, so minimization
yields the Q that best satisfies sign restrictions.
"""
function _uhlig_penalty(theta_all::AbstractVector{T}, restrictions::SVARRestrictions,
                         Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                         model::VARModel{T}, horizon::Int, n::Int) where {T<:AbstractFloat}
    # Guard: return large penalty for degenerate inputs
    any(isnan, theta_all) && return T(1e10)

    Q = try
        _uhlig_build_Q(theta_all, restrictions, Phi, L, n)
    catch
        return T(1e10)
    end

    # Compute IRF
    irf = _compute_irf_for_Q(model, Q, Phi, L, horizon)

    # Compute standard deviations for normalization
    sigma = zeros(T, n)
    for i in 1:n
        sigma[i] = sqrt(max(model.Sigma[i, i], eps(T)))
    end

    # Penalty computation: Uhlig (2009) Eq. (6)
    total_penalty = zero(T)
    for sr in restrictions.signs
        h_idx = sr.horizon + 1
        h_idx > horizon && continue

        response = irf[h_idx, sr.variable, sr.shock]
        sigma_v = sigma[sr.variable]

        # Normalized response in direction of required sign
        normalized = sr.sign * response / sigma_v

        # Weight: 100 if satisfied, 1 if violated
        weight = normalized >= zero(T) ? T(100) : one(T)
        total_penalty -= weight * normalized
    end

    total_penalty
end

"""
Compute per-shock penalty diagnostics.
"""
function _uhlig_shock_penalties(Q::Matrix{T}, restrictions::SVARRestrictions,
                                 Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                                 model::VARModel{T}, horizon::Int) where {T<:AbstractFloat}
    n = size(Q, 1)
    irf = _compute_irf_for_Q(model, Q, Phi, L, horizon)

    sigma = zeros(T, n)
    for i in 1:n
        sigma[i] = sqrt(max(model.Sigma[i, i], eps(T)))
    end

    shock_penalties = zeros(T, n)
    for sr in restrictions.signs
        h_idx = sr.horizon + 1
        h_idx > horizon && continue

        response = irf[h_idx, sr.variable, sr.shock]
        normalized = sr.sign * response / sigma[sr.variable]
        weight = normalized >= zero(T) ? T(100) : one(T)
        shock_penalties[sr.shock] -= weight * normalized
    end

    shock_penalties
end

# =============================================================================
# Main Identification Function
# =============================================================================

"""
    identify_uhlig(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
        n_starts=50, n_refine=10, max_iter_coarse=500, max_iter_fine=2000,
        tol_coarse=1e-4, tol_fine=1e-8) -> UhligSVARResult{T}

Identify SVAR using Mountford & Uhlig (2009) penalty function approach.

Uses Nelder-Mead optimization over spherical coordinates to find the rotation
matrix ``Q`` that best satisfies sign restrictions, with zero restrictions
enforced as hard constraints via null-space projection.

# Algorithm
1. Precompute MA coefficients and Cholesky factor ``L``
2. **Phase 1** (coarse): `n_starts` Nelder-Mead runs from random ``\\theta_0 \\in [0, 2\\pi]``
3. **Phase 2** (refinement): `n_refine` local re-optimizations from best solution
4. Build final ``Q``, compute IRFs, check convergence

# Keywords
- `n_starts::Int=50`: Number of random starting points (Phase 1)
- `n_refine::Int=10`: Number of local refinements (Phase 2)
- `max_iter_coarse::Int=500`: Max iterations per Phase 1 run
- `max_iter_fine::Int=2000`: Max iterations per Phase 2 run
- `tol_coarse::T=1e-4`: Convergence tolerance for Phase 1
- `tol_fine::T=1e-8`: Convergence tolerance for Phase 2

# Returns
`UhligSVARResult{T}` with optimal rotation matrix, IRFs, penalty values,
and convergence indicator.

# Example
```julia
model = estimate_var(Y, 2)
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(3, 1)],
    signs = [sign_restriction(1, 1, :positive),
             sign_restriction(2, 1, :positive)]
)
result = identify_uhlig(model, restrictions, 20)
```

**Reference**: Mountford & Uhlig (2009)
"""
function identify_uhlig(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                         n_starts::Int=50, n_refine::Int=10,
                         max_iter_coarse::Int=500, max_iter_fine::Int=2000,
                         tol_coarse::T=T(1e-4), tol_fine::T=T(1e-8)) where {T<:AbstractFloat}
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension ($( restrictions.n_vars)) must match model ($n)"

    # Need sign restrictions for penalty function
    isempty(restrictions.signs) && throw(ArgumentError(
        "identify_uhlig requires at least one sign restriction"))

    # Determine required horizon for restrictions
    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(zr.horizon for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(sr.horizon for sr in restrictions.signs) + 1)

    # Precompute MA coefficients and Cholesky factor
    Phi = _compute_ma_coefficients(model, max_h)
    L = safe_cholesky(model.Sigma)

    # Count free parameters
    n_params = _uhlig_n_params(n, restrictions)

    # Objective closure
    obj = theta -> _uhlig_penalty(theta, restrictions, Phi, L, model, max_h, n)

    # =========================================================================
    # Phase 1: Coarse search from random starting points
    # =========================================================================
    best_val = T(Inf)
    best_theta = zeros(T, n_params)

    for _ in 1:n_starts
        theta0 = rand(T, n_params) .* T(2π)

        result = try
            Optim.optimize(obj, theta0, Optim.NelderMead(),
                Optim.Options(iterations=max_iter_coarse,
                              f_reltol=tol_coarse))
        catch
            continue
        end

        val = Optim.minimum(result)
        if isfinite(val) && val < best_val
            best_val = val
            best_theta = Optim.minimizer(result)
        end
    end

    best_val == T(Inf) && error("All starting points failed in Phase 1")

    # =========================================================================
    # Phase 2: Local refinement from best solution
    # =========================================================================
    for i in 1:n_refine
        # Perturb best solution slightly for diversity
        if i == 1
            theta0 = copy(best_theta)
        else
            theta0 = best_theta .+ T(0.01) .* randn(T, n_params)
        end

        result = try
            Optim.optimize(obj, theta0, Optim.NelderMead(),
                Optim.Options(iterations=max_iter_fine,
                              f_reltol=tol_fine))
        catch
            continue
        end

        val = Optim.minimum(result)
        if isfinite(val) && val < best_val
            best_val = val
            best_theta = Optim.minimizer(result)
        end
    end

    # =========================================================================
    # Build final result
    # =========================================================================
    Q = _uhlig_build_Q(best_theta, restrictions, Phi, L, n)
    irf = _compute_irf_for_Q(model, Q, Phi, L, horizon)

    # Check convergence: all sign restrictions satisfied?
    converged = _check_sign_restrictions(irf, restrictions)

    # Per-shock penalty diagnostics
    shock_penalties = _uhlig_shock_penalties(Q, restrictions, Phi, L, model, max_h)

    UhligSVARResult{T}(Q, irf, best_val, shock_penalties, restrictions, converged)
end
