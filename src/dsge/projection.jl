# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Chebyshev Collocation Projection Solver
#
# References:
#   Judd (1998), Numerical Methods in Economics
#   Malin-Krueger-Kubler (2011), Solving the Multi-Country RBC Model
#   Judd-Maliar-Maliar-Valero (2014), Smolyak Method for Nonlinear Dynamic Models

# =============================================================================
# Chebyshev Basis Helpers
# =============================================================================

"""
    _chebyshev_nodes(n::Int) -> Vector{Float64}

Chebyshev extrema (Gauss-Lobatto) nodes on [-1,1]: `x_j = cos(πj/(n-1))` for j=0,...,n-1.
"""
function _chebyshev_nodes(n::Int)
    n >= 2 || throw(ArgumentError("n must be >= 2 for Chebyshev nodes"))
    [cos(π * j / (n - 1)) for j in 0:(n - 1)]
end

"""
    _chebyshev_eval(x::Real, degree::Int) -> Vector{Float64}

Evaluate Chebyshev polynomials T_0(x), T_1(x), ..., T_degree(x) at scalar x.
Uses the recurrence T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x).
"""
function _chebyshev_eval(x::Real, degree::Int)
    vals = zeros(degree + 1)
    vals[1] = 1.0       # T_0 = 1
    if degree >= 1
        vals[2] = Float64(x)  # T_1 = x
    end
    @inbounds for k in 2:degree
        vals[k + 1] = 2.0 * Float64(x) * vals[k] - vals[k - 1]
    end
    return vals
end

"""
    _chebyshev_basis_multi(X::AbstractMatrix, multi_indices::AbstractMatrix{Int}) -> Matrix{Float64}

Evaluate tensor-product Chebyshev basis at points X (n_points x nx).
multi_indices is n_basis x nx, each row gives (i_1,...,i_nx) polynomial degrees.
Returns n_points x n_basis basis matrix.
"""
function _chebyshev_basis_multi(X::AbstractMatrix, multi_indices::AbstractMatrix{Int})
    n_points, nx = size(X)
    n_basis = size(multi_indices, 1)
    max_deg = maximum(multi_indices)

    # Pre-compute 1D evaluations for each dimension
    T_vals = [zeros(n_points, max_deg + 1) for _ in 1:nx]
    for d in 1:nx
        for i in 1:n_points
            tv = _chebyshev_eval(X[i, d], max_deg)
            T_vals[d][i, :] = tv
        end
    end

    # Build basis matrix via tensor products
    B = ones(n_points, n_basis)
    @inbounds for k in 1:n_basis
        for d in 1:nx
            deg = multi_indices[k, d]
            B[:, k] .*= T_vals[d][:, deg + 1]
        end
    end

    return B
end

# =============================================================================
# Scaling
# =============================================================================

"""
    _scale_to_unit(x::AbstractVector, bounds::AbstractMatrix) -> Vector

Affine map from [a_i, b_i] to [-1, 1] for each dimension.
bounds is nx x 2 with bounds[i,:] = [a_i, b_i].
"""
function _scale_to_unit(x::AbstractVector, bounds::AbstractMatrix)
    a = @view bounds[:, 1]
    b = @view bounds[:, 2]
    return 2.0 .* (x .- a) ./ (b .- a) .- 1.0
end

"""
    _scale_from_unit(z::AbstractVector, bounds::AbstractMatrix) -> Vector

Affine map from [-1, 1] to [a_i, b_i] for each dimension.
"""
function _scale_from_unit(z::AbstractVector, bounds::AbstractMatrix)
    a = @view bounds[:, 1]
    b = @view bounds[:, 2]
    return a .+ (z .+ 1.0) ./ 2.0 .* (b .- a)
end

# Matrix versions for multiple points
function _scale_to_unit(X::AbstractMatrix, bounds::AbstractMatrix)
    n = size(X, 1)
    Z = similar(X, Float64)
    for i in 1:n
        Z[i, :] = _scale_to_unit(X[i, :], bounds)
    end
    return Z
end

function _scale_from_unit(Z::AbstractMatrix, bounds::AbstractMatrix)
    n = size(Z, 1)
    X = similar(Z, Float64)
    for i in 1:n
        X[i, :] = _scale_from_unit(Z[i, :], bounds)
    end
    return X
end

# =============================================================================
# Grid Construction
# =============================================================================

"""
    _tensor_grid(nx::Int, degree::Int) -> (nodes, multi_indices)

Tensor-product Chebyshev grid. Returns:
- `nodes`: (degree+1)^nx x nx matrix of grid points in [-1,1]
- `multi_indices`: (degree+1)^nx x nx matrix of polynomial multi-indices
"""
function _tensor_grid(nx::Int, degree::Int)
    n1d = degree + 1
    nodes1d = _chebyshev_nodes(n1d)
    n_total = n1d^nx

    nodes = zeros(n_total, nx)
    mi = zeros(Int, n_total, nx)

    for idx in 0:(n_total - 1)
        rem = idx
        for d in nx:-1:1
            j = rem % n1d
            rem = div(rem, n1d)
            nodes[idx + 1, d] = nodes1d[j + 1]
            mi[idx + 1, d] = j
        end
    end

    return nodes, mi
end

"""
    _cc_points(level::Int) -> Vector{Float64}

Nested Clenshaw-Curtis (Chebyshev extrema) points for a one-dimensional Smolyak level.
Level 0 is the single midpoint; level `l >= 1` has `m_l = 2^l + 1` points, and the
level-`l` set nests the level-`(l-1)` set.
"""
function _cc_points(level::Int)
    if level == 0
        return [0.0]
    else
        m = 2^level + 1
        return [cos(π * j / (m - 1)) for j in 0:(m - 1)]
    end
end

"""
    _smolyak_level_vector(nx::Int, mu) -> Vector{Int}

Normalize a Smolyak approximation level into a per-dimension level vector of length `nx`.
A scalar `mu` gives the isotropic vector `fill(mu, nx)`; an `nx`-vector is validated and
returned as-is (the anisotropic case).
"""
function _smolyak_level_vector(nx::Int, mu)
    if mu isa Integer
        mu >= 0 || throw(ArgumentError("smolyak_mu must be >= 0, got $mu"))
        return fill(Int(mu), nx)
    end
    v = collect(Int, mu)
    length(v) == nx || throw(ArgumentError(
        "smolyak_mu must be a scalar or a vector of length nx=$nx, got length $(length(v))"))
    all(>=(0), v) || throw(ArgumentError("all smolyak_mu entries must be >= 0, got $v"))
    return v
end

"""
    _smolyak_admissible_levels(mu_vec::Vector{Int}) -> Vector{Vector{Int}}

Admissible (downward-closed) Smolyak level set for per-dimension levels `mu_vec`.

The dimension-adaptive weighting of Gerstner & Griebel (2003) admits the levels
`{l ∈ ℕ₀^d : Σ_k l_k / μ_k ≤ 1}`. To keep the test exact in integer arithmetic the
rule is scaled by `P = lcm(μ_k)`, giving weights `w_k = P ÷ μ_k` and the test
`Σ_k w_k l_k ≤ P`. A dimension with `μ_k = 0` is pinned at level 0.

With `μ_k = μ` for all `k` this reduces to `P = μ`, `w_k = 1`, i.e. the isotropic
rule `|l|_1 ≤ μ` (equivalently `|α|_1 ≤ μ + d` for `α = l + 1`).
"""
function _smolyak_admissible_levels(mu_vec::Vector{Int})
    d = length(mu_vec)
    d >= 1 || throw(ArgumentError("Smolyak level vector must be non-empty"))
    positives = filter(>(0), mu_vec)
    isempty(positives) && return [zeros(Int, d)]

    P = foldl(lcm, positives)
    # μ_k = 0 ⇒ weight P+1 > P ⇒ only l_k = 0 is admissible in that dimension.
    w = [m == 0 ? P + 1 : P ÷ m for m in mu_vec]

    levels = Vector{Vector{Int}}()
    l = zeros(Int, d)
    function _rec(k::Int, budget::Int)
        if k > d
            push!(levels, copy(l))
            return nothing
        end
        lk = 0
        while lk * w[k] <= budget
            l[k] = lk
            _rec(k + 1, budget - lk * w[k])
            lk += 1
        end
        l[k] = 0
        return nothing
    end
    _rec(1, P)
    sort!(levels)
    return levels
end

"""
    _smolyak_block_nodes(l::AbstractVector{Int}) -> Vector{Vector{Float64}}

Tensor-product Clenshaw-Curtis nodes contributed by the single Smolyak level block `l`.
Coordinates are rounded to 14 digits so nested points compare equal across blocks.
"""
function _smolyak_block_nodes(l::AbstractVector{Int})
    nx = length(l)
    pts_per_dim = [_cc_points(lk) for lk in l]
    sizes = [length(p) for p in pts_per_dim]
    n_combo = prod(sizes)
    out = Vector{Vector{Float64}}(undef, n_combo)
    for idx in 0:(n_combo - 1)
        pt = zeros(nx)
        rem = idx
        for d in nx:-1:1
            j = rem % sizes[d]
            rem = div(rem, sizes[d])
            pt[d] = pts_per_dim[d][j + 1]
        end
        out[idx + 1] = round.(pt; digits=14)
    end
    return out
end

"""
    _smolyak_grid_from_levels(levels::Vector{Vector{Int}}) -> (nodes, multi_indices)

Build the sparse grid and the matching polynomial multi-index set from an admissible
Smolyak level set.

Both come out of the SAME combination loop, so the collocation basis is unisolvent on the
sparse grid: a nested Clenshaw-Curtis level `l` contributes both its `m_l` points and the
degrees `0:(m_l - 1)`. (The pre-#218 code took the full total-degree set `|α|₁ ≤ μ + nx` and
clipped it by row sum, which is NOT unisolvent — for `d=2, μ=1` it kept `(1,1)`, whose `x·y`
basis vanishes at every one of the 5 nodes, while dropping `(2,0)`/`(0,2)`.)
"""
function _smolyak_grid_from_levels(levels::Vector{Vector{Int}})
    isempty(levels) && throw(ArgumentError("Smolyak level set must be non-empty"))
    nx = length(first(levels))

    all_points = Set{Vector{Float64}}()
    index_set = Set{Vector{Int}}()

    for l in levels
        length(l) == nx || throw(ArgumentError("All Smolyak levels must have length $nx"))
        pts_per_dim = [_cc_points(lk) for lk in l]
        sizes = [length(p) for p in pts_per_dim]
        n_combo = prod(sizes)
        for idx in 0:(n_combo - 1)
            pt = zeros(nx)
            alpha = zeros(Int, nx)
            rem = idx
            for d in nx:-1:1
                j = rem % sizes[d]
                rem = div(rem, sizes[d])
                pt[d] = pts_per_dim[d][j + 1]
                alpha[d] = j                 # admissible degree 0:(m_d-1) for this tensor block
            end
            push!(all_points, round.(pt; digits=14))
            push!(index_set, alpha)
        end
    end

    nodes_list = collect(all_points)
    sort!(nodes_list)
    n_nodes = length(nodes_list)
    nodes = zeros(n_nodes, nx)
    for (i, pt) in enumerate(nodes_list)
        nodes[i, :] = pt
    end

    mi_list = sort(collect(index_set))
    n_basis = length(mi_list)
    mi_final = zeros(Int, n_basis, nx)
    for (i, a) in enumerate(mi_list)
        mi_final[i, :] = a
    end

    return nodes, mi_final
end

"""
    _smolyak_combination_coefficients(levels::Vector{Vector{Int}}) -> Vector{Int}

Smolyak combination coefficients for an admissible level set, using the general
Gerstner-Griebel rule that holds for ANY downward-closed set (isotropic, anisotropic, or
adaptively grown):

```math
c_l = \\sum_{z \\in \\{0,1\\}^d,\\; l + z \\in A} (-1)^{|z|_1}
```

For the isotropic set `|l|_1 ≤ μ` this reduces to the textbook closed form
`c_l = (-1)^{μ - |l|_1} \\binom{d-1}{μ - |l|_1}`. The coefficients always sum to 1.
"""
function _smolyak_combination_coefficients(levels::Vector{Vector{Int}})
    isempty(levels) && return Int[]
    d = length(first(levels))
    S = Set(levels)
    coeffs = zeros(Int, length(levels))
    z = zeros(Int, d)
    for (i, l) in enumerate(levels)
        c = 0
        for m in 0:(2^d - 1)
            s = 0
            for k in 1:d
                z[k] = (m >> (k - 1)) & 1
                s += z[k]
            end
            (l .+ z) in S && (c += iseven(s) ? 1 : -1)
        end
        coeffs[i] = c
    end
    return coeffs
end

"""
    _smolyak_forward_neighbours(levels::Vector{Vector{Int}}) -> Vector{Vector{Int}}

Admissible forward neighbours of a downward-closed level set: the candidates `l + e_k` that
are not yet in the set but whose every backward neighbour is. Adding any one of them keeps
the set downward-closed, which is what makes the adaptive refinement a valid Smolyak
construction (Gerstner & Griebel 2003).
"""
function _smolyak_forward_neighbours(levels::Vector{Vector{Int}})
    isempty(levels) && return Vector{Vector{Int}}()
    d = length(first(levels))
    S = Set(levels)
    cands = Vector{Vector{Int}}()
    for l in levels, k in 1:d
        c = copy(l)
        c[k] += 1
        c in S && continue
        c in cands && continue
        admissible = true
        for j in 1:d
            c[j] == 0 && continue
            b = copy(c)
            b[j] -= 1
            if !(b in S)
                admissible = false
                break
            end
        end
        admissible && push!(cands, c)
    end
    return cands
end

"""
    _smolyak_grid(nx::Int, mu) -> (nodes, multi_indices)

Smolyak sparse grid at approximation level `mu`, which may be a scalar (isotropic) or an
`nx`-vector of per-dimension levels (anisotropic).

Uses nested Chebyshev extrema (Clenshaw-Curtis) points. The isotropic selection rule is
`|alpha|_1 <= mu + nx` for multi-indices `alpha`; see `_smolyak_admissible_levels`
for the anisotropic generalization.

Returns:
- `nodes`: n_nodes x nx grid points in [-1,1]
- `multi_indices`: n_basis x nx polynomial multi-indices
"""
function _smolyak_grid(nx::Int, mu)
    return _smolyak_grid_from_levels(_smolyak_admissible_levels(_smolyak_level_vector(nx, mu)))
end

# =============================================================================
# State Bounds Computation
# =============================================================================

"""
    _compute_state_bounds(spec, linear, state_idx, scale) -> Matrix

Compute ergodic state bounds: SS_i +/- scale * sigma_i using first-order solution.
Returns nx x 2 matrix with [lower upper] per state.
"""
function _compute_state_bounds(spec::ModelSpec{T}, linear::LinearDSGE{T},
                                state_idx::Vector{Int}, scale::Real) where {T}
    nx = length(state_idx)
    result = _gensys_qz(spec, linear)
    G1 = result.G
    impact = result.impact

    # Unconditional variance via Lyapunov equation. A unit-root/explosive first-order solution
    # has no finite unconditional covariance (solve_lyapunov throws); fall back to zero variance
    # so the min_half floor below still supplies finite, usable state bounds (#220).
    Var_y = try
        solve_lyapunov(G1, impact)
    catch
        @warn "State-bounds Lyapunov solve failed (non-stationary first-order solution); " *
              "falling back to the minimum-width floor for state bounds."
        zeros(T, size(G1, 1), size(G1, 1))
    end

    ss = spec.steady_state
    bounds = zeros(T, nx, 2)
    for (i, si) in enumerate(state_idx)
        sigma_i = sqrt(max(Var_y[si, si], zero(T)))
        half_width = T(scale) * sigma_i
        # Minimum bound width: 10% of |SS| or 0.1 (whichever is larger)
        # This prevents degenerate zero-width bounds when the linearized
        # variance is near zero (e.g., poorly conditioned level models)
        min_half = max(T(0.1) * abs(ss[si]), T(0.1))
        half_width = max(half_width, min_half)
        bounds[i, 1] = ss[si] - half_width
        bounds[i, 2] = ss[si] + half_width
    end

    return bounds
end

# =============================================================================
# Collocation Residual
# =============================================================================

"""
    _collocation_residual(coeffs_vec, args...) -> Vector{T}

Compute residual vector R(c) for the collocation system.
At each node j, evaluates equilibrium equations using current policy (from coefficients),
quadrature-based expectations for next period, and model residual functions.
"""
function _collocation_residual(coeffs_vec::AbstractVector{T},
                                n_vars::Int, n_basis::Int,
                                basis_matrix::Matrix{T},
                                nodes_phys::Matrix{T},
                                state_idx::Vector{Int},
                                control_idx::Vector{Int},
                                spec::ModelSpec{T},
                                quad_nodes::Matrix{T},
                                quad_weights::Vector{T},
                                state_bounds::Matrix{T},
                                multi_indices::Matrix{Int},
                                steady_state::Vector{T},
                                impact::Matrix{T}) where {T}

    coeffs = reshape(coeffs_vec, n_vars, n_basis)
    n_nodes = size(basis_matrix, 1)
    n_eq = spec.n_endog
    n_quad = length(quad_weights)
    n_eps = spec.n_exog
    nx = length(state_idx)
    θ = spec.param_values

    R = zeros(T, n_eq * n_nodes)

    for j in 1:n_nodes
        # Current policy at this node: deviations from SS
        y_dev = zeros(T, n_eq)
        for v in 1:n_vars
            y_dev[v] = dot(@view(basis_matrix[j, :]), @view(coeffs[v, :]))
        end
        y_t = y_dev .+ steady_state  # levels

        # y_lag: the node represents the lagged state
        y_lag = copy(steady_state)
        for (ii, si) in enumerate(state_idx)
            y_lag[si] = nodes_phys[j, ii]
        end

        # Compute expected next-period variables via quadrature
        y_lead_expected = zeros(T, n_eq)
        for q in 1:n_quad
            iszero(quad_weights[q]) && continue   # center node contributes 0 (S-19 / #224)
            # Next-period states = current policy state components (deviation)
            x_next_dev = zeros(T, nx)
            for (ii, si) in enumerate(state_idx)
                x_next_dev[ii] = y_dev[si]
            end
            x_next_level = x_next_dev .+ steady_state[state_idx]

            # Integrate over the next-period shock at this quadrature node: add the linear
            # shock impact so E_t[y_{t+1}] is a genuine quadrature, not n_quad identical
            # deterministic evaluations (the certainty-equivalent bug — audit S-02 / #120).
            for (ii, si) in enumerate(state_idx)
                for k in 1:n_eps
                    x_next_level[ii] += impact[si, k] * quad_nodes[q, k]
                end
            end

            # Clamp to state bounds
            for d in 1:nx
                x_next_level[d] = clamp(x_next_level[d], state_bounds[d, 1], state_bounds[d, 2])
            end

            # Map to [-1,1] and evaluate basis
            z_next = _scale_to_unit(x_next_level, state_bounds)
            z_next = clamp.(z_next, T(-1), T(1))
            B_next = _chebyshev_basis_multi(reshape(z_next, 1, nx), multi_indices)

            y_next = zeros(T, n_eq)
            for v in 1:n_vars
                y_next[v] = dot(@view(B_next[1, :]), @view(coeffs[v, :]))
            end
            y_next_level = y_next .+ steady_state

            y_lead_expected .+= quad_weights[q] .* y_next_level
        end

        # Evaluate equilibrium residuals (with domain error protection)
        ε_zero = zeros(T, n_eps)
        for i in 1:n_eq
            try
                R[(j - 1) * n_eq + i] = spec.residual_fns[i](y_t, y_lag, y_lead_expected, ε_zero, θ)
            catch e
                if e isa DomainError || e isa InexactError
                    R[(j - 1) * n_eq + i] = T(1e10)  # large penalty
                else
                    rethrow(e)
                end
            end
        end
    end

    # Replace NaN/Inf with large penalty for robustness
    for i in eachindex(R)
        if !isfinite(R[i])
            R[i] = T(1e10)
        end
    end

    return R
end

# =============================================================================
# Collocation Solver
# =============================================================================

# Gauss-Newton solve of the collocation system on a FIXED grid/basis. Returns
# `(coeffs_vec, converged, iterations, residual_norm)`. Factored out of `collocation_solver`
# so the adaptive-refinement loop can re-solve on each grown grid.
function _collocation_newton(coeffs_vec::Vector{T}, n_vars::Int, n_basis::Int,
                             basis_matrix::Matrix{T}, nodes_phys::Matrix{T},
                             state_idx::Vector{Int}, control_idx::Vector{Int},
                             spec::ModelSpec{T}, quad_nodes::Matrix{T},
                             quad_weights::Vector{T}, state_bounds::Matrix{T},
                             multi_indices::Matrix{Int}, ss::Vector{T},
                             impact_mat::Matrix{T};
                             tol::Real, max_iter::Int, threaded::Bool,
                             verbose::Bool) where {T}
    converged = false
    iter = 0
    residual_norm = T(Inf)

    # The per-iteration finite-difference Jacobian costs one residual evaluation per unknown
    # and dominates the solve. Reuse it as a chord (modified-Newton) step, recomputing only on
    # a fresh start, periodically, or when a reused step stalls. The QR least-squares step below
    # (column-pivoted, rank-revealing) replaces the former J'J normal equations. (#225 part 2)
    J = Matrix{T}(undef, 0, 0)
    jac_stale = true
    jac_refresh_period = 5

    for k in 1:max_iter
        iter = k

        R = _collocation_residual(coeffs_vec, n_vars, n_basis,
                                   basis_matrix, nodes_phys,
                                   state_idx, control_idx, spec,
                                   quad_nodes, quad_weights,
                                   state_bounds, multi_indices, ss, impact_mat)

        residual_norm = norm(R)

        if verbose
            @info "Iteration $k: ||R|| = $(residual_norm)"
        else
            @debug "Iteration $k: ||R|| = $(residual_norm)"
        end

        if residual_norm < tol
            converged = true
            break
        end

        # (Re)compute the finite-difference Jacobian only when needed (chord-step reuse — #225).
        if jac_stale || (k % jac_refresh_period == 0)
            n_unknowns = length(coeffs_vec)
            n_residuals = length(R)
            J = zeros(T, n_residuals, n_unknowns)
            h_fd = max(T(1e-7), sqrt(eps(T)))

            if threaded && Threads.nthreads() > 1
                Threads.@threads for i in 1:n_unknowns
                    c_plus = copy(coeffs_vec)
                    c_plus[i] += h_fd
                    R_plus = _collocation_residual(c_plus, n_vars, n_basis,
                                                    basis_matrix, nodes_phys,
                                                    state_idx, control_idx, spec,
                                                    quad_nodes, quad_weights,
                                                    state_bounds, multi_indices, ss, impact_mat)
                    J[:, i] = (R_plus .- R) ./ h_fd
                end
            else
                for i in 1:n_unknowns
                    c_plus = copy(coeffs_vec)
                    c_plus[i] += h_fd
                    R_plus = _collocation_residual(c_plus, n_vars, n_basis,
                                                    basis_matrix, nodes_phys,
                                                    state_idx, control_idx, spec,
                                                    quad_nodes, quad_weights,
                                                    state_bounds, multi_indices, ss, impact_mat)
                    J[:, i] = (R_plus .- R) ./ h_fd
                end
            end
            jac_stale = false
        end

        # Gauss-Newton step via column-pivoted QR least squares (solves min‖J·δ + R‖; avoids
        # squaring cond(J) through the former J'J normal equations — #225 part 2).
        delta = -(qr(J, ColumnNorm()) \ R)

        # Line search
        alpha = one(T)
        best_norm = residual_norm
        best_alpha = zero(T)
        for _ in 1:8
            c_trial = coeffs_vec .+ alpha .* delta
            R_trial = _collocation_residual(c_trial, n_vars, n_basis,
                                             basis_matrix, nodes_phys,
                                             state_idx, control_idx, spec,
                                             quad_nodes, quad_weights,
                                             state_bounds, multi_indices, ss, impact_mat)
            trial_norm = norm(R_trial)
            if trial_norm < best_norm
                best_norm = trial_norm
                best_alpha = alpha
            end
            alpha *= T(0.5)
        end

        if best_alpha > 0
            coeffs_vec .+= best_alpha .* delta
        else
            coeffs_vec .+= T(0.01) .* delta
            jac_stale = true      # no descent with the reused Jacobian ⇒ refresh next iteration
        end
        # A reused (chord) step that barely reduces the residual signals a stale Jacobian.
        if best_norm > T(0.9) * residual_norm
            jac_stale = true
        end
    end

    return coeffs_vec, converged, iter, residual_norm
end

# Carry coefficients from a coarser Smolyak basis onto a refined one. The refined multi-index
# set is a SUPERSET of the old one (adding a block to a downward-closed level set only adds
# degrees), so padding with zeros represents exactly the same policy function — an exact warm
# start, not an approximation.
function _pad_coefficients(old_coeffs::Matrix{T}, old_mi::Matrix{Int},
                           new_mi::Matrix{Int}) where {T}
    n_vars = size(old_coeffs, 1)
    new_coeffs = zeros(T, n_vars, size(new_mi, 1))
    lookup = Dict{Vector{Int},Int}(new_mi[i, :] => i for i in 1:size(new_mi, 1))
    for k in 1:size(old_mi, 1)
        j = get(lookup, old_mi[k, :], 0)
        j > 0 && (new_coeffs[:, j] = @view old_coeffs[:, k])
    end
    return new_coeffs
end

"""
    collocation_solver(spec::ModelSpec{T}; kwargs...) -> ProjectionSolution{T}

Solve DSGE model via Chebyshev collocation (projection method).

# Keyword Arguments
- `degree::Int=5`: Chebyshev polynomial degree (tensor grid)
- `grid::Symbol=:auto`: `:tensor`, `:smolyak`, or `:auto`
- `smolyak_mu=3`: Smolyak exactness level. A scalar gives the isotropic rule `|l|₁ ≤ μ`;
  an `nx`-vector `(μ_1,…,μ_nx)` gives the **anisotropic** rule `Σ_k l_k/μ_k ≤ 1`
  (Gerstner & Griebel 2003), which spends resolution only on the states that need it.
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `scale::Real=3.0`: state bounds = SS +/- scale * sigma
- `tol::Real=1e-8`: Newton convergence tolerance
- `max_iter::Int=100`: maximum Newton iterations
- `threaded::Bool=false`: enable multi-threaded Jacobian evaluation
- `verbose::Bool=false`: print iteration info
- `initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing`: warm-start coefficients (n_vars x n_basis)
- `adaptive::Bool=false`: enable dimension-adaptive Smolyak refinement (requires a Smolyak
  grid; `grid=:auto` selects one automatically when `adaptive=true`)
- `euler_tol::Real=1e-6`: target max Euler error for adaptive refinement
- `max_nodes::Int=1000`: node budget — refinement stops before exceeding it
- `max_refinements::Int=10`: maximum refinement rounds
- `n_euler_test::Int=200`: random test points used for the Euler-error target
- `rng=Random.default_rng()`: rng for the Euler-error test points

# Adaptive refinement
With `adaptive=true` the solver grows the Smolyak level set one block at a time. Each
admissible forward neighbour of the current set is scored by the maximum absolute Euler
residual at the **new** nodes it would introduce, and the highest-scoring block is added — so
basis functions are added only where the residual is large. Refinement stops when the max
Euler error falls below `euler_tol`, when the node budget would be exceeded, or after
`max_refinements` rounds. The achieved accuracy is reported in `sol.euler_error` and the
final level set in `sol.smolyak_levels`.
"""
function collocation_solver(spec::ModelSpec{T};
                            degree::Int=5,
                            grid::Symbol=:auto,
                            smolyak_mu::Union{Integer,AbstractVector{<:Integer}}=3,
                            quadrature::Symbol=:auto,
                            n_quad::Int=5,
                            scale::Real=3.0,
                            tol::Real=1e-8,
                            max_iter::Int=100,
                            threaded::Bool=false,
                            verbose::Bool=false,
                            initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                            adaptive::Bool=false,
                            euler_tol::Real=1e-6,
                            max_nodes::Int=1000,
                            max_refinements::Int=10,
                            n_euler_test::Int=200,
                            rng=Random.default_rng()) where {T<:AbstractFloat}

    n_eq = spec.n_endog
    n_eps = spec.n_exog
    ss = spec.steady_state

    # Step 1: Linearize to get state/control partition
    ld = linearize(spec)
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)

    nx > 0 || throw(ArgumentError("Model has no state variables — projection requires at least one"))

    # Auto-select grid type. Adaptive refinement grows a Smolyak level set, so :auto picks
    # :smolyak whenever adaptivity is requested regardless of nx.
    if grid == :auto
        grid = (adaptive || nx > 4) ? :smolyak : :tensor
    end

    grid in (:tensor, :smolyak) ||
        throw(ArgumentError("grid must be :tensor, :smolyak, or :auto"))
    adaptive && grid != :smolyak && throw(ArgumentError(
        "adaptive=true requires a Smolyak grid (refinement grows the Smolyak level set); " *
        "got grid=:$grid"))

    # Auto-select quadrature
    if quadrature == :auto
        quadrature = n_eps <= 2 ? :gauss_hermite : :monomial
    end

    if grid == :tensor && nx > 4
        @warn "Tensor grid with nx=$nx states is expensive. Consider grid=:smolyak." maxlog=1
    end

    # Step 2: Compute state bounds
    state_bounds = _compute_state_bounds(spec, ld, state_idx, scale)
    state_bounds_T = Matrix{T}(state_bounds)

    # Step 3: Smolyak level set (anisotropic when `smolyak_mu` is a vector)
    mu_vec = grid == :smolyak ? _smolyak_level_vector(nx, smolyak_mu) : Int[]
    levels = grid == :smolyak ? _smolyak_admissible_levels(mu_vec) : Vector{Vector{Int}}()

    # Step 4: Set up quadrature
    Sigma_e = Matrix{T}(I, n_eps, n_eps)
    if quadrature == :gauss_hermite
        quad_nodes, quad_weights = _gauss_hermite_scaled(n_quad, Sigma_e)
    elseif quadrature == :monomial
        quad_nodes, quad_weights = _monomial_nodes_weights(n_eps)
    else
        throw(ArgumentError("quadrature must be :gauss_hermite, :monomial, or :auto"))
    end
    quad_nodes = Matrix{T}(quad_nodes)
    quad_weights = Vector{T}(quad_weights)

    # First-order shock-impact matrix for genuine quadrature over next-period shocks (S-02 / #120)
    result_1st = _gensys_qz(spec, ld)
    G1 = result_1st.G
    impact_mat = result_1st.impact

    n_vars = n_eq

    # Euler-error test points are drawn ONCE and reused across refinement rounds, so the
    # round-to-round accuracy comparison reflects the grid and not resampling noise.
    euler_qn, euler_qw = _euler_quadrature(T, quadrature, n_eps)
    test_points = adaptive ? _random_state_points(state_bounds_T, n_euler_test, rng) :
                             zeros(T, 0, nx)

    warm_coeffs = initial_coeffs === nothing ? nothing : Matrix{T}(initial_coeffs)
    warm_mi = nothing

    local nodes_unit, multi_indices, coeffs_final
    converged = false
    iter = 0
    residual_norm = T(Inf)
    euler_err = T(NaN)
    n_refine = 0
    hit_target = false

    for refine_round in 0:(adaptive ? max_refinements : 0)
        # Step 5: Build collocation grid for this round
        if grid == :tensor
            nodes_unit, multi_indices = _tensor_grid(nx, degree)
        else
            nodes_unit, multi_indices = _smolyak_grid_from_levels(levels)
        end

        n_nodes = size(nodes_unit, 1)
        n_basis = size(multi_indices, 1)
        nodes_phys_T = Matrix{T}(_scale_from_unit(nodes_unit, state_bounds))
        basis_matrix = Matrix{T}(_chebyshev_basis_multi(nodes_unit, multi_indices))

        # Step 6: Initial guess — refined-grid warm start, user warm start, or first order
        coeffs = if warm_mi !== nothing
            # The refined basis is a superset of the previous one: pad exactly.
            _pad_coefficients(warm_coeffs, warm_mi, multi_indices)
        elseif warm_coeffs !== nothing && size(warm_coeffs) == (n_vars, n_basis)
            copy(warm_coeffs)
        else
            c0 = zeros(T, n_vars, n_basis)
            for v in 1:n_vars
                y_nodes = zeros(T, n_nodes)
                for j in 1:n_nodes
                    x_dev = nodes_phys_T[j, :] .- ss[state_idx]
                    y_nodes[j] = dot(G1[v, state_idx], x_dev)
                end
                c0[v, :] = basis_matrix \ y_nodes
            end
            c0
        end

        # Step 7: Newton iteration on this grid
        coeffs_vec, converged, iter, residual_norm = _collocation_newton(
            vec(coeffs), n_vars, n_basis, basis_matrix, nodes_phys_T,
            state_idx, control_idx, spec, quad_nodes, quad_weights,
            state_bounds_T, multi_indices, ss, impact_mat;
            tol=tol, max_iter=max_iter, threaded=threaded, verbose=verbose)

        coeffs_final = reshape(coeffs_vec, n_vars, n_basis)

        if !adaptive
            break
        end

        # Step 8: Achieved accuracy on the fixed test set
        euler_err = _max_euler_error_at(spec, coeffs_final, multi_indices, state_bounds_T,
                                        state_idx, ss, euler_qn, euler_qw, impact_mat,
                                        test_points)
        if verbose
            @info "Refinement round $refine_round: $(size(nodes_unit, 1)) nodes, " *
                  "max Euler error = $euler_err"
        end
        if euler_err <= T(euler_tol)
            hit_target = true
            break
        end
        refine_round == max_refinements && break

        # Step 9: Score each admissible forward neighbour by the residual at its NEW nodes
        existing = Set(round.(Vector{Float64}(nodes_unit[j, :]); digits=14)
                       for j in 1:size(nodes_unit, 1))
        best_block = nothing
        best_score = T(-Inf)
        for cand in _smolyak_forward_neighbours(levels)
            new_pts = [p for p in _smolyak_block_nodes(cand) if !(p in existing)]
            isempty(new_pts) && continue
            score = zero(T)
            for p in new_pts
                x_phys = Vector{T}(_scale_from_unit(Vector{T}(p), state_bounds))
                r = _euler_residuals_at(spec, coeffs_final, multi_indices, state_bounds_T,
                                        state_idx, ss, euler_qn, euler_qw, impact_mat, x_phys)
                score = max(score, maximum(r))
            end
            if score > best_score
                best_score = score
                best_block = cand
            end
        end
        best_block === nothing && break

        trial_levels = sort(vcat(levels, [best_block]))
        n_trial = size(_smolyak_grid_from_levels(trial_levels)[1], 1)
        if n_trial > max_nodes
            verbose && @info "Refinement stopped: adding block $best_block would need " *
                             "$n_trial nodes (max_nodes = $max_nodes)"
            break
        end

        warm_coeffs = coeffs_final
        warm_mi = multi_indices
        levels = trial_levels
        n_refine += 1
    end

    if !converged && verbose
        @warn "Collocation solver did not converge after $max_iter iterations (||R|| = $residual_norm)"
    end
    if adaptive && !hit_target
        @warn "Adaptive refinement finished at $(size(nodes_unit, 1)) nodes " *
              "($n_refine refinements) with max Euler error $euler_err > euler_tol = " *
              "$euler_tol; raise `max_nodes`/`max_refinements` or relax `euler_tol`." maxlog=1
    end

    # Step 10: Package result
    level_matrix = if grid == :smolyak
        L = zeros(Int, length(levels), nx)
        for (i, l) in enumerate(levels)
            L[i, :] = l
        end
        L
    else
        zeros(Int, 0, 0)
    end

    return ProjectionSolution{T}(
        coeffs_final,
        state_bounds_T,
        grid,
        grid == :smolyak ? maximum(maximum.(levels)) : degree,
        Matrix{T}(nodes_unit),
        residual_norm,
        size(multi_indices, 1),
        multi_indices,
        quadrature,
        spec,
        ld,
        impact_mat,
        ss,
        state_idx,
        control_idx,
        converged,
        iter,
        :projection;
        euler_error=euler_err,
        smolyak_levels=level_matrix,
        refinements=n_refine
    )
end

# =============================================================================
# Policy Evaluation
# =============================================================================

# Evaluate the Chebyshev policy approximation at a physical state vector, given the raw
# coefficient/basis pieces. Shared by `evaluate_policy` and by the residual diagnostics that
# run before a `ProjectionSolution` exists (the adaptive-refinement loop).
function _proj_eval(coeffs::Matrix{T}, multi_indices::Matrix{Int},
                    state_bounds::Matrix{T}, steady_state::Vector{T},
                    x_level::AbstractVector) where {T}
    nx = size(state_bounds, 1)
    z = _scale_to_unit(Vector{T}(x_level), state_bounds)
    z = clamp.(z, T(-1), T(1))
    B = _chebyshev_basis_multi(reshape(z, 1, nx), multi_indices)
    return coeffs * Vector{T}(@view B[1, :]) .+ steady_state
end

"""
    evaluate_policy(sol::ProjectionSolution{T}, x_state::AbstractVector) -> Vector{T}

Evaluate the global policy function at a state vector.
`x_state` should be an nx-vector of state variable levels.
Returns n_vars-vector of all endogenous variable levels.
"""
function evaluate_policy(sol::ProjectionSolution{T}, x_state::AbstractVector) where {T}
    nx = nstates(sol)
    @assert length(x_state) == nx "x_state must have $nx elements"

    z = _scale_to_unit(Vector{T}(x_state), sol.state_bounds)

    if any(abs.(z) .> 1)
        @warn "State outside approximation domain — extrapolating" maxlog=1
    end

    return _proj_eval(sol.coefficients, sol.multi_indices, sol.state_bounds,
                      sol.steady_state, x_state)
end

"""
    evaluate_policy(sol::ProjectionSolution{T}, X_states::AbstractMatrix) -> Matrix{T}

Evaluate at multiple state points. X_states is n_points x nx.
Returns n_points x n_vars matrix of levels.
"""
function evaluate_policy(sol::ProjectionSolution{T}, X_states::AbstractMatrix) where {T}
    n_points = size(X_states, 1)
    n_vars = nvars(sol)
    Y = zeros(T, n_points, n_vars)
    for i in 1:n_points
        Y[i, :] = evaluate_policy(sol, X_states[i, :])
    end
    return Y
end

# =============================================================================
# Euler Error Diagnostic
# =============================================================================

# Absolute equilibrium-equation residuals of a Chebyshev policy approximation at one physical
# state point. The expectation is a genuine quadrature over the next-period shock (S-02 /
# #120): the first-order impact matrix moves the next-period state at every quadrature node.
# Shared by `max_euler_error` (random test points) and by the adaptive-refinement error
# indicator (candidate nodes), so refinement is driven by exactly the metric that is reported.
function _euler_residuals_at(spec::ModelSpec{T}, coeffs::Matrix{T},
                             multi_indices::Matrix{Int}, state_bounds::Matrix{T},
                             state_idx::Vector{Int}, steady_state::Vector{T},
                             quad_nodes::Matrix{T}, quad_weights::Vector{T},
                             impact::Matrix{T}, x_level::AbstractVector{T}) where {T}
    n_eq = spec.n_endog
    n_eps = spec.n_exog
    nx = length(state_idx)
    θ = spec.param_values

    y_t = _proj_eval(coeffs, multi_indices, state_bounds, steady_state, x_level)

    y_lag = copy(steady_state)
    for (ii, si) in enumerate(state_idx)
        y_lag[si] = x_level[ii]
    end

    y_lead_exp = zeros(T, n_eq)
    for q in 1:size(quad_nodes, 1)
        iszero(quad_weights[q]) && continue   # center node contributes 0 (S-19 / #224)
        x_next_level = Vector{T}(y_t[state_idx])
        for (ii, si) in enumerate(state_idx)
            for k in 1:n_eps
                x_next_level[ii] += impact[si, k] * quad_nodes[q, k]
            end
        end
        for d in 1:nx
            x_next_level[d] = clamp(x_next_level[d], state_bounds[d, 1], state_bounds[d, 2])
        end
        y_lead_exp .+= quad_weights[q] .*
            _proj_eval(coeffs, multi_indices, state_bounds, steady_state, x_next_level)
    end

    ε_zero = zeros(T, n_eps)
    out = zeros(T, n_eq)
    for i in 1:n_eq
        val = try
            abs(spec.residual_fns[i](y_t, y_lag, y_lead_exp, ε_zero, θ))
        catch e
            if e isa DomainError || e isa InexactError
                T(1e10)
            else
                rethrow(e)
            end
        end
        out[i] = isfinite(val) ? val : T(1e10)
    end
    return out
end

# Maximum absolute Euler residual over a supplied set of physical test points
# (`points` is n_points × nx).
function _max_euler_error_at(spec::ModelSpec{T}, coeffs::Matrix{T},
                             multi_indices::Matrix{Int}, state_bounds::Matrix{T},
                             state_idx::Vector{Int}, steady_state::Vector{T},
                             quad_nodes::Matrix{T}, quad_weights::Vector{T},
                             impact::Matrix{T}, points::AbstractMatrix{T}) where {T}
    max_err = zero(T)
    for j in 1:size(points, 1)
        r = _euler_residuals_at(spec, coeffs, multi_indices, state_bounds, state_idx,
                                steady_state, quad_nodes, quad_weights, impact,
                                Vector{T}(@view points[j, :]))
        max_err = max(max_err, maximum(r))
    end
    return max_err
end

# Uniform random test points inside the state domain (n_test × nx).
function _random_state_points(state_bounds::Matrix{T}, n_test::Int, rng) where {T}
    nx = size(state_bounds, 1)
    pts = zeros(T, n_test, nx)
    for j in 1:n_test, d in 1:nx
        lo = state_bounds[d, 1]
        hi = state_bounds[d, 2]
        pts[j, d] = lo + rand(rng, T) * (hi - lo)
    end
    return pts
end

# Quadrature rule matching a solution's `quadrature` symbol, as used by the Euler diagnostic.
function _euler_quadrature(::Type{T}, quadrature::Symbol, n_eps::Int) where {T}
    Sigma_e = Matrix{T}(I, n_eps, n_eps)
    qn, qw = quadrature == :gauss_hermite ? _gauss_hermite_scaled(5, Sigma_e) :
                                            _monomial_nodes_weights(n_eps)
    return Matrix{T}(qn), Vector{T}(qw)
end

"""
    max_euler_error(sol::ProjectionSolution{T}; n_test::Int=1000, rng=Random.default_rng()) -> T

Compute maximum Euler equation error on random test points within the state domain.
"""
function max_euler_error(sol::ProjectionSolution{T}; n_test::Int=1000,
                          rng=Random.default_rng()) where {T}
    quad_nodes, quad_weights = _euler_quadrature(T, sol.quadrature, nshocks(sol))

    # First-order shock impact so the Euler-error diagnostic integrates over next-period
    # shocks too (else it cannot detect the certainty-equivalent failure — S-02 / #120).
    impact_mat = sol.impact

    points = _random_state_points(sol.state_bounds, n_test, rng)
    return _max_euler_error_at(sol.spec, sol.coefficients, sol.multi_indices,
                               sol.state_bounds, sol.state_indices, sol.steady_state,
                               quad_nodes, quad_weights, impact_mat, points)
end
