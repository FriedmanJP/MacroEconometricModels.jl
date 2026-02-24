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
    for k in 2:degree
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
    for k in 1:n_basis
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
    _smolyak_grid(nx::Int, mu::Int) -> (nodes, multi_indices)

Smolyak sparse grid with exactness level mu.

Uses nested Chebyshev extrema (Clenshaw-Curtis) points.
Smolyak selection rule: |alpha|_1 <= mu + nx for multi-indices alpha.

Returns:
- `nodes`: n_nodes x nx grid points in [-1,1]
- `multi_indices`: n_basis x nx polynomial multi-indices
"""
function _smolyak_grid(nx::Int, mu::Int)
    # Level-to-number-of-points mapping (nested Clenshaw-Curtis)
    function _cc_points(level::Int)
        if level == 0
            return [0.0]
        else
            m = 2^level + 1
            return [cos(π * j / (m - 1)) for j in 0:(m - 1)]
        end
    end

    max_sum = mu + nx

    # Generate multi-indices with |alpha|_1 <= max_sum
    function _gen_multi_indices(ndim, max_s)
        if ndim == 1
            return reshape(collect(0:max_s), max_s + 1, 1)
        end
        sub = _gen_multi_indices(ndim - 1, max_s)
        result = Matrix{Int}(undef, 0, ndim)
        for i in 0:max_s
            valid = sub[vec(sum(sub; dims=2)) .<= max_s - i, :]
            if !isempty(valid)
                col_i = fill(i, size(valid, 1))
                result = vcat(result, hcat(col_i, valid))
            end
        end
        return result
    end

    mi = _gen_multi_indices(nx, max_sum)

    # Build Smolyak grid points using the combination technique
    all_points = Set{Vector{Float64}}()

    function _gen_level_combos(ndim, target_sum_max, min_level)
        if ndim == 1
            result = Matrix{Int}(undef, 0, 1)
            for s in min_level:target_sum_max
                result = vcat(result, reshape([s], 1, 1))
            end
            return result
        end
        result = Matrix{Int}(undef, 0, ndim)
        for s in min_level:target_sum_max
            sub = _gen_level_combos(ndim - 1, target_sum_max - s, min_level)
            if !isempty(sub)
                col_s = fill(s, size(sub, 1))
                result = vcat(result, hcat(col_s, sub))
            end
        end
        return result
    end

    level_combos = _gen_level_combos(nx, mu + nx, 1)

    for row in eachrow(level_combos)
        level_shifted = [max(r - 1, 0) for r in row]
        pts_per_dim = [_cc_points(l) for l in level_shifted]

        sizes = [length(p) for p in pts_per_dim]
        n_combo = prod(sizes)
        for idx in 0:(n_combo - 1)
            pt = zeros(nx)
            rem = idx
            for d in nx:-1:1
                j = rem % sizes[d]
                rem = div(rem, sizes[d])
                pt[d] = pts_per_dim[d][j + 1]
            end
            push!(all_points, round.(pt; digits=14))
        end
    end

    nodes_list = collect(all_points)
    sort!(nodes_list)
    n_nodes = length(nodes_list)
    nodes = zeros(n_nodes, nx)
    for (i, pt) in enumerate(nodes_list)
        nodes[i, :] = pt
    end

    # Filter multi-indices to match grid size
    mi_sums = vec(sum(mi; dims=2))
    perm = sortperm(mi_sums)
    mi_sorted = mi[perm, :]
    n_basis = min(n_nodes, size(mi_sorted, 1))
    mi_final = mi_sorted[1:n_basis, :]

    return nodes, mi_final
end

# =============================================================================
# State Bounds Computation
# =============================================================================

"""
    _compute_state_bounds(spec, linear, state_idx, scale) -> Matrix

Compute ergodic state bounds: SS_i +/- scale * sigma_i using first-order solution.
Returns nx x 2 matrix with [lower upper] per state.
"""
function _compute_state_bounds(spec::DSGESpec{T}, linear::LinearDSGE{T},
                                state_idx::Vector{Int}, scale::Real) where {T}
    nx = length(state_idx)
    result = gensys(linear.Gamma0, linear.Gamma1, linear.C, linear.Psi, linear.Pi)
    G1 = result.G1
    impact = result.impact

    # Unconditional variance via Lyapunov equation
    Var_y = solve_lyapunov(G1, impact)

    ss = spec.steady_state
    bounds = zeros(T, nx, 2)
    for (i, si) in enumerate(state_idx)
        sigma_i = sqrt(max(Var_y[si, si], zero(T)))
        bounds[i, 1] = ss[si] - T(scale) * sigma_i
        bounds[i, 2] = ss[si] + T(scale) * sigma_i
    end

    return bounds
end
