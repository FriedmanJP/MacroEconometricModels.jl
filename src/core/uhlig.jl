# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
- `penalty::T`: Total penalty at optimum (lower is better)
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
    varnames::Vector{String}
end

# Back-compatible arity without varnames
function UhligSVARResult{T}(Q, irf, penalty, shock_penalties, restrictions,
                            converged) where {T<:AbstractFloat}
    nv = restrictions.n_vars
    UhligSVARResult{T}(Q, irf, penalty, shock_penalties, restrictions, converged,
                       ["var$i" for i in 1:nv])
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
                                n::Int; B=nothing, C1=nothing) where {T<:AbstractFloat}
    # Build constraint matrix: orthogonality to previous columns + zero restrictions
    constraint_rows = Vector{Vector{T}}()

    # Orthogonality constraints from previous columns
    for k in 1:j-1
        push!(constraint_rows, Q_prev[:, k])
    end

    # Zero restriction constraints for shock j (finite, long-run, A0, A+)
    ZF = _compute_ZF(restrictions, Phi, L, j; B=B, C1=C1)
    for i in axes(ZF, 1)
        push!(constraint_rows, Vector{T}(ZF[i, :]))
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
                         n::Int; B=nothing, C1=nothing) where {T<:AbstractFloat}
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

        Q[:, j] = _uhlig_build_q_column(theta_j, j, Q, restrictions, Phi, L, n; B=B, C1=C1)
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

"""Uhlig's penalty is defined only for IRF sign restrictions."""
function _uhlig_assert_sign_only_rejections(restrictions::SVARRestrictions)
    bad = unique(typeof(s) for s in restrictions.signs if !(s isa SignRestriction))
    isempty(bad) && return nothing
    throw(ArgumentError(
        "identify_uhlig only evaluates SignRestriction; got $(join(string.(bad), ", ")). " *
        "Use identify_arias for elasticity, magnitude, FEVD, cumulative, and A₀/A₊ sign " *
        "restrictions. A₀/A₊ and long-run zeros remain valid null-space constraints."))
end

# =============================================================================
# Penalty Function
# =============================================================================

"""
Uhlig (2005) penalty function.

For each sign restriction, let `normalized` be the IRF in the required-sign
direction divided by the residual standard deviation, and `x = -normalized`
(so `x > 0` ⇔ the restriction is violated). Then

- weight 1 if satisfied (`x ≤ 0`): add `x` (small reward)
- weight `penalty_weight` (default 100) if violated (`x > 0`): add `penalty_weight * x`

Lower penalty is better. Minimization therefore makes violations prohibitively
expensive rather than rewarding large satisfied responses.

Reference: Uhlig (2005, JME 52, §3.3); Mountford & Uhlig (2009, JAE 24, §3).
"""
function _uhlig_penalty(theta_all::AbstractVector{T}, restrictions::SVARRestrictions,
                         Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                         model::VARModel{T}, horizon::Int, n::Int;
                         penalty_weight::Real=100, C1=nothing) where {T<:AbstractFloat}
    pw = T(penalty_weight)
    # Guard: return large penalty for degenerate inputs
    any(isnan, theta_all) && return T(1e10)

    Q = try
        _uhlig_build_Q(theta_all, restrictions, Phi, L, n; B=model.B, C1=C1)
    catch err
        _is_rejectable_draw_error(err) || rethrow(err)
        return T(1e10)
    end

    # Compute IRF
    irf = structural_irf(Phi, L, Q, horizon)

    # Compute standard deviations for normalization
    sigma = zeros(T, n)
    for i in 1:n
        sigma[i] = sqrt(max(model.Sigma[i, i], eps(T)))
    end

    # Penalty computation: Uhlig (2005) §3.3
    total_penalty = zero(T)
    for sr in restrictions.signs
        sr isa SignRestriction || continue
        h_idx = sr.horizon + 1
        response = irf[h_idx, sr.variable, sr.shock]
        normalized = sr.sign * response / sigma[sr.variable]
        x = -normalized                    # >0 ⇔ violated
        total_penalty += x <= zero(T) ? x : pw * x
    end

    total_penalty
end

"""
Compute per-shock penalty diagnostics.

Same `f(x)` as `_uhlig_penalty`: weight 1 if satisfied, `penalty_weight` if
violated. Lower is better.
"""
function _uhlig_shock_penalties(Q::Matrix{T}, restrictions::SVARRestrictions,
                                 Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                                 model::VARModel{T}, horizon::Int;
                                 penalty_weight::Real=100) where {T<:AbstractFloat}
    pw = T(penalty_weight)
    n = size(Q, 1)
    irf = structural_irf(Phi, L, Q, horizon)

    sigma = zeros(T, n)
    for i in 1:n
        sigma[i] = sqrt(max(model.Sigma[i, i], eps(T)))
    end

    shock_penalties = zeros(T, n)
    for sr in restrictions.signs
        sr isa SignRestriction || continue
        h_idx = sr.horizon + 1
        response = irf[h_idx, sr.variable, sr.shock]
        normalized = sr.sign * response / sigma[sr.variable]
        x = -normalized
        shock_penalties[sr.shock] += x <= zero(T) ? x : pw * x
    end

    shock_penalties
end

# =============================================================================
# Main Identification Function
# =============================================================================

"""
    identify_uhlig(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
        n_starts=50, n_refine=10, max_iter_coarse=500, max_iter_fine=2000,
        tol_coarse=1e-4, tol_fine=1e-8, penalty_weight=100) -> UhligSVARResult{T}

Identify SVAR using Mountford & Uhlig (2009) penalty function approach.

Uses Nelder-Mead optimization over spherical coordinates to find the rotation
matrix ``Q`` that best satisfies sign restrictions, with zero restrictions
enforced as hard constraints via null-space projection.

The penalty is Uhlig (2005): `x = -normalized`, then `x` if satisfied and
`penalty_weight * x` if violated. Lower `penalty` is better.

Rejection restrictions other than [`SignRestriction`](@ref) are not part of
the penalty or the `converged` flag; mixed containers throw `ArgumentError`.
Use [`identify_arias`](@ref) for elasticity, magnitude, FEVD, cumulative, and
``A_0``/``A_+`` signs. Linear zeros (finite IRF, long-run, ``A_0``, ``A_+``)
remain null-space constraints.

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
- `penalty_weight::T=T(100)`: Multiplier on violated sign restrictions (Uhlig 2005)

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
                         tol_coarse::T=T(1e-4), tol_fine::T=T(1e-8),
                         penalty_weight::Real=100,
                         rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension ($( restrictions.n_vars)) must match model ($n)"

    # Need sign restrictions for penalty function
    any(s -> s isa SignRestriction, restrictions.signs) || throw(ArgumentError(
        "identify_uhlig requires at least one sign restriction"))
    _uhlig_assert_sign_only_rejections(restrictions)

    # Determine required horizon for restrictions
    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(_restriction_horizon(zr) for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(_restriction_horizon(sr) for sr in restrictions.signs) + 1)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         first(_long_run_multiplier(model.B, model.Sigma, n, model.p)) : nothing

    # Precompute MA coefficients and Cholesky factor
    Phi = ma_coefficients(model, max_h + 1)
    L = safe_cholesky(model.Sigma)

    # Count free parameters
    n_params = _uhlig_n_params(n, restrictions)

    pw = T(penalty_weight)
    # Objective closure
    obj = theta -> _uhlig_penalty(theta, restrictions, Phi, L, model, max_h, n;
                                  penalty_weight=pw, C1=C1)

    # =========================================================================
    # Phase 1: Coarse search from random starting points (multi-threaded)
    # =========================================================================
    results_phase1 = Vector{Tuple{T, Vector{T}}}(undef, n_starts)
    fill!(results_phase1, (T(Inf), zeros(T, n_params)))

    seeds1 = rand(rng, UInt64, n_starts)
    Threads.@threads for i in 1:n_starts
        local_rng = Random.MersenneTwister(seeds1[i])
        theta0 = rand(local_rng, T, n_params) .* T(2π)

        res = try
            Optim.optimize(obj, theta0, Optim.NelderMead(),
                Optim.Options(iterations=max_iter_coarse,
                              f_reltol=tol_coarse))
        catch err
            _is_rejectable_draw_error(err) ? nothing : rethrow(err)
        end

        if res !== nothing
            val = Optim.minimum(res)
            if isfinite(val)
                results_phase1[i] = (val, Optim.minimizer(res))
            end
        end
    end

    best_idx = argmin(first.(results_phase1))
    best_val, best_theta = results_phase1[best_idx]
    best_val == T(Inf) && error("All starting points failed in Phase 1")

    # =========================================================================
    # Phase 2: Local refinement from best solution (multi-threaded)
    # =========================================================================
    results_phase2 = Vector{Tuple{T, Vector{T}}}(undef, n_refine)
    fill!(results_phase2, (T(Inf), zeros(T, n_params)))
    best_theta_snap = copy(best_theta)

    seeds2 = rand(rng, UInt64, n_refine)
    Threads.@threads for i in 1:n_refine
        local_rng = Random.MersenneTwister(seeds2[i])
        theta0 = if i == 1
            copy(best_theta_snap)
        else
            best_theta_snap .+ T(0.01) .* randn(local_rng, T, n_params)
        end

        res = try
            Optim.optimize(obj, theta0, Optim.NelderMead(),
                Optim.Options(iterations=max_iter_fine,
                              f_reltol=tol_fine))
        catch err
            _is_rejectable_draw_error(err) ? nothing : rethrow(err)
        end

        if res !== nothing
            val = Optim.minimum(res)
            if isfinite(val)
                results_phase2[i] = (val, Optim.minimizer(res))
            end
        end
    end

    for (val, theta) in results_phase2
        if val < best_val
            best_val = val
            best_theta = theta
        end
    end

    # =========================================================================
    # Build final result
    # =========================================================================
    Q = _uhlig_build_Q(best_theta, restrictions, Phi, L, n; B=model.B, C1=C1)
    irf_full = structural_irf(Phi, L, Q, max_h)

    # Check convergence: all sign restrictions satisfied?
    converged = _check_sign_restrictions(irf_full, restrictions)
    irf = irf_full[1:horizon, :, :]

    # Per-shock penalty diagnostics
    shock_penalties = _uhlig_shock_penalties(Q, restrictions, Phi, L, model, max_h;
                                             penalty_weight=pw)

    UhligSVARResult{T}(Q, irf, best_val, shock_penalties, restrictions, converged,
                       copy(model.varnames))
end
