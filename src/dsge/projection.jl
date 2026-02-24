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
                                spec::DSGESpec{T},
                                quad_nodes::Matrix{T},
                                quad_weights::Vector{T},
                                state_bounds::Matrix{T},
                                multi_indices::Matrix{Int},
                                steady_state::Vector{T}) where {T}

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
            # Next-period states = current policy state components (deviation)
            x_next_dev = zeros(T, nx)
            for (ii, si) in enumerate(state_idx)
                x_next_dev[ii] = y_dev[si]
            end
            x_next_level = x_next_dev .+ steady_state[state_idx]

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

"""
    collocation_solver(spec::DSGESpec{T}; kwargs...) -> ProjectionSolution{T}

Solve DSGE model via Chebyshev collocation (projection method).

# Keyword Arguments
- `degree::Int=5`: Chebyshev polynomial degree
- `grid::Symbol=:auto`: `:tensor`, `:smolyak`, or `:auto`
- `smolyak_mu::Int=3`: Smolyak exactness level
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `scale::Real=3.0`: state bounds = SS +/- scale * sigma
- `tol::Real=1e-8`: Newton convergence tolerance
- `max_iter::Int=100`: maximum Newton iterations
- `verbose::Bool=false`: print iteration info
"""
function collocation_solver(spec::DSGESpec{T};
                            degree::Int=5,
                            grid::Symbol=:auto,
                            smolyak_mu::Int=3,
                            quadrature::Symbol=:auto,
                            n_quad::Int=5,
                            scale::Real=3.0,
                            tol::Real=1e-8,
                            max_iter::Int=100,
                            verbose::Bool=false) where {T<:AbstractFloat}

    n_eq = spec.n_endog
    n_eps = spec.n_exog
    ss = spec.steady_state

    # Step 1: Linearize to get state/control partition
    ld = linearize(spec)
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)

    nx > 0 || throw(ArgumentError("Model has no state variables — projection requires at least one"))

    # Auto-select grid type
    if grid == :auto
        grid = nx <= 4 ? :tensor : :smolyak
    end

    # Auto-select quadrature
    if quadrature == :auto
        quadrature = n_eps <= 2 ? :gauss_hermite : :monomial
    end

    if grid == :tensor && nx > 4
        @warn "Tensor grid with nx=$nx states is expensive. Consider grid=:smolyak." maxlog=1
    end

    # Step 2: Compute state bounds
    state_bounds = _compute_state_bounds(spec, ld, state_idx, scale)

    # Step 3: Build collocation grid
    if grid == :tensor
        nodes_unit, multi_indices = _tensor_grid(nx, degree)
    elseif grid == :smolyak
        nodes_unit, multi_indices = _smolyak_grid(nx, smolyak_mu)
    else
        throw(ArgumentError("grid must be :tensor, :smolyak, or :auto"))
    end

    n_nodes = size(nodes_unit, 1)
    n_basis = size(multi_indices, 1)
    n_vars = n_eq

    # Map nodes to physical coordinates
    nodes_phys = _scale_from_unit(nodes_unit, state_bounds)

    # Build basis matrix at collocation nodes
    basis_matrix = Matrix{T}(_chebyshev_basis_multi(nodes_unit, multi_indices))

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

    # Step 5: Initial guess from first-order perturbation
    result_1st = gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi)
    G1 = result_1st.G1

    coeffs = zeros(T, n_vars, n_basis)
    for v in 1:n_vars
        y_nodes = zeros(T, n_nodes)
        for j in 1:n_nodes
            x_dev = nodes_phys[j, :] .- ss[state_idx]
            y_nodes[j] = dot(G1[v, state_idx], x_dev)
        end
        coeffs[v, :] = basis_matrix \ y_nodes
    end

    # Step 6: Newton iteration
    coeffs_vec = vec(coeffs)
    converged = false
    iter = 0
    residual_norm = T(Inf)

    nodes_phys_T = Matrix{T}(nodes_phys)
    state_bounds_T = Matrix{T}(state_bounds)

    for k in 1:max_iter
        iter = k

        R = _collocation_residual(coeffs_vec, n_vars, n_basis,
                                   basis_matrix, nodes_phys_T,
                                   state_idx, control_idx, spec,
                                   quad_nodes, quad_weights,
                                   state_bounds_T, multi_indices, ss)

        residual_norm = norm(R)

        if verbose
            println("  Iteration $k: ||R|| = $(residual_norm)")
        end

        if residual_norm < tol
            converged = true
            break
        end

        # Jacobian via finite differences
        n_unknowns = length(coeffs_vec)
        n_residuals = length(R)
        J = zeros(T, n_residuals, n_unknowns)
        h_fd = max(T(1e-7), sqrt(eps(T)))

        for i in 1:n_unknowns
            c_plus = copy(coeffs_vec)
            c_plus[i] += h_fd
            R_plus = _collocation_residual(c_plus, n_vars, n_basis,
                                            basis_matrix, nodes_phys_T,
                                            state_idx, control_idx, spec,
                                            quad_nodes, quad_weights,
                                            state_bounds_T, multi_indices, ss)
            J[:, i] = (R_plus .- R) ./ h_fd
        end

        # Gauss-Newton step with line search
        JtJ = J' * J
        JtR = J' * R
        delta = -(robust_inv(JtJ) * JtR)

        # Line search
        alpha = one(T)
        best_norm = residual_norm
        best_alpha = zero(T)
        for _ in 1:8
            c_trial = coeffs_vec .+ alpha .* delta
            R_trial = _collocation_residual(c_trial, n_vars, n_basis,
                                             basis_matrix, nodes_phys_T,
                                             state_idx, control_idx, spec,
                                             quad_nodes, quad_weights,
                                             state_bounds_T, multi_indices, ss)
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
        end
    end

    if !converged && verbose
        @warn "Collocation solver did not converge after $max_iter iterations (||R|| = $residual_norm)"
    end

    # Step 7: Package result
    coeffs_final = reshape(coeffs_vec, n_vars, n_basis)

    return ProjectionSolution{T}(
        coeffs_final,
        state_bounds_T,
        grid,
        grid == :smolyak ? smolyak_mu : degree,
        Matrix{T}(nodes_unit),
        residual_norm,
        n_basis,
        multi_indices,
        quadrature,
        spec,
        ld,
        ss,
        state_idx,
        control_idx,
        converged,
        iter,
        :projection
    )
end

# =============================================================================
# Policy Evaluation
# =============================================================================

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
        z = clamp.(z, T(-1), T(1))
    end

    B = _chebyshev_basis_multi(reshape(z, 1, nx), sol.multi_indices)
    y_dev = sol.coefficients * B[1, :]

    return y_dev .+ sol.steady_state
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

"""
    max_euler_error(sol::ProjectionSolution{T}; n_test::Int=1000, rng=Random.default_rng()) -> T

Compute maximum Euler equation error on random test points within the state domain.
"""
function max_euler_error(sol::ProjectionSolution{T}; n_test::Int=1000,
                          rng=Random.default_rng()) where {T}
    nx = nstates(sol)
    n_eps = nshocks(sol)
    n_eq = nvars(sol)
    spec = sol.spec
    θ = spec.param_values
    ss = sol.steady_state

    Sigma_e = Matrix{T}(I, n_eps, n_eps)
    if sol.quadrature == :gauss_hermite
        quad_nodes, quad_weights = _gauss_hermite_scaled(5, Sigma_e)
    else
        quad_nodes, quad_weights = _monomial_nodes_weights(n_eps)
    end
    quad_nodes = Matrix{T}(quad_nodes)
    quad_weights = Vector{T}(quad_weights)

    max_err = zero(T)

    for _ in 1:n_test
        x_level = zeros(T, nx)
        for d in 1:nx
            lo = sol.state_bounds[d, 1]
            hi = sol.state_bounds[d, 2]
            x_level[d] = lo + rand(rng, T) * (hi - lo)
        end

        y_t = evaluate_policy(sol, x_level)

        y_lag = copy(ss)
        for (ii, si) in enumerate(sol.state_indices)
            y_lag[si] = x_level[ii]
        end

        y_lead_exp = zeros(T, n_eq)
        for q in 1:size(quad_nodes, 1)
            x_next_dev = y_t[sol.state_indices] .- ss[sol.state_indices]
            x_next_level = x_next_dev .+ ss[sol.state_indices]
            for d in 1:nx
                x_next_level[d] = clamp(x_next_level[d], sol.state_bounds[d, 1], sol.state_bounds[d, 2])
            end
            y_next = evaluate_policy(sol, x_next_level)
            y_lead_exp .+= quad_weights[q] .* y_next
        end

        ε_zero = zeros(T, n_eps)
        for i in 1:n_eq
            try
                err = abs(spec.residual_fns[i](y_t, y_lag, y_lead_exp, ε_zero, θ))
                if isfinite(err)
                    max_err = max(max_err, err)
                else
                    max_err = max(max_err, T(1e10))
                end
            catch e
                if e isa DomainError || e isa InexactError
                    max_err = max(max_err, T(1e10))
                else
                    rethrow(e)
                end
            end
        end
    end

    return max_err
end
