# MacroEconometricModels.jl — Policy Function Iteration (Time Iteration)
#
# References:
#   Coleman (1990), Solving the Stochastic Growth Model
#   Judd (1998), Numerical Methods in Economics, Ch. 11
#   Heer & Maussner (2009), Dynamic General Equilibrium Modeling

"""
    _pfi_euler_step(y_guess, y_lag, E_y_lead, spec) -> Vector{T}

Solve the Euler equation at one grid point via Newton iteration.

Given lagged state `y_lag` (levels) and expected next-period variables `E_y_lead` (levels),
find `y_t` (levels) such that `F(y_t, y_lag, E_y_lead, 0, θ) = 0`.
"""
function _pfi_euler_step(y_guess::Vector{T}, y_lag::Vector{T},
                          E_y_lead::Vector{T}, spec::DSGESpec{T};
                          newton_tol::Real=1e-10, newton_max::Int=50) where {T}
    n_eq = spec.n_endog
    n_eps = spec.n_exog
    θ = spec.param_values
    ε_zero = zeros(T, n_eps)

    y = copy(y_guess)
    h = max(T(1e-7), sqrt(eps(T)))

    for _ in 1:newton_max
        # Evaluate residuals
        R = zeros(T, n_eq)
        for i in 1:n_eq
            try
                R[i] = spec.residual_fns[i](y, y_lag, E_y_lead, ε_zero, θ)
            catch e
                if e isa DomainError || e isa InexactError
                    R[i] = T(1e10)
                else
                    rethrow(e)
                end
            end
        end

        if maximum(abs.(R)) < newton_tol
            return y
        end

        # Jacobian w.r.t. y_t (finite differences)
        J = zeros(T, n_eq, n_eq)
        for j in 1:n_eq
            y_plus = copy(y)
            y_plus[j] += h
            for i in 1:n_eq
                try
                    R_plus = spec.residual_fns[i](y_plus, y_lag, E_y_lead, ε_zero, θ)
                    J[i, j] = (R_plus - R[i]) / h
                catch e
                    if e isa DomainError || e isa InexactError
                        J[i, j] = T(1e10)
                    else
                        rethrow(e)
                    end
                end
            end
        end

        # Newton step
        if n_eq == 1
            abs(J[1, 1]) > eps(T) || break
            y[1] -= R[1] / J[1, 1]
        else
            delta = -(robust_inv(J) * R)
            y .+= delta
        end
    end

    return y
end

"""
    _pfi_compute_expectations(coeffs, n_vars, n_basis, state_idx, spec,
                               quad_nodes, quad_weights, state_bounds,
                               multi_indices, steady_state, y_current_nodes, impact)

Compute E[y_{t+1}] at all grid points using quadrature.

For each grid point j, expected next-period values are:
`E[y'] = Σ_i w_i · policy(x'(x_j, ε_i))`

where `x'` are next-period states derived from current policy + shock.
"""
function _pfi_compute_expectations(coeffs::Matrix{T}, n_vars::Int, n_basis::Int,
                                    state_idx::Vector{Int}, spec::DSGESpec{T},
                                    quad_nodes::Matrix{T}, quad_weights::Vector{T},
                                    state_bounds::Matrix{T}, multi_indices::Matrix{Int},
                                    steady_state::Vector{T},
                                    y_current_nodes::Matrix{T},
                                    impact::Matrix{T}) where {T}
    n_nodes = size(y_current_nodes, 1)
    n_eq = spec.n_endog
    n_eps = spec.n_exog
    nx = length(state_idx)
    n_quad = length(quad_weights)

    E_y_lead = zeros(T, n_nodes, n_eq)

    for j in 1:n_nodes
        for q in 1:n_quad
            # Next-period states from current policy
            x_next_level = zeros(T, nx)
            for (ii, si) in enumerate(state_idx)
                x_next_level[ii] = y_current_nodes[j, si]
            end

            # Add shock effect via linear impact matrix
            for (ii, si) in enumerate(state_idx)
                for k in 1:n_eps
                    x_next_level[ii] += impact[si, k] * quad_nodes[q, k]
                end
            end

            # Clamp to bounds
            for d in 1:nx
                x_next_level[d] = clamp(x_next_level[d], state_bounds[d, 1], state_bounds[d, 2])
            end

            # Evaluate policy at next-period states
            z_next = _scale_to_unit(x_next_level, state_bounds)
            z_next = clamp.(z_next, T(-1), T(1))
            B_next = _chebyshev_basis_multi(reshape(z_next, 1, nx), multi_indices)

            y_next = zeros(T, n_eq)
            for v in 1:n_vars
                y_next[v] = dot(@view(B_next[1, :]), @view(coeffs[v, :])) + steady_state[v]
            end

            E_y_lead[j, :] .+= quad_weights[q] .* y_next
        end
    end

    return E_y_lead
end

"""
    pfi_solver(spec::DSGESpec{T}; kwargs...) -> ProjectionSolution{T}

Solve DSGE model via Policy Function Iteration (Time Iteration).

At each iteration: (1) compute expected next-period values using quadrature,
(2) solve Euler equation at each grid point via Newton, (3) refit Chebyshev
coefficients via least squares.

# Keyword Arguments
- `degree::Int=5`: Chebyshev polynomial degree
- `grid::Symbol=:auto`: `:tensor`, `:smolyak`, or `:auto`
- `smolyak_mu::Int=3`: Smolyak exactness level
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `scale::Real=3.0`: state bounds = SS ± scale × σ
- `tol::Real=1e-8`: sup-norm convergence tolerance
- `max_iter::Int=500`: maximum PFI iterations
- `damping::Real=1.0`: policy mixing factor (1.0 = no damping)
- `verbose::Bool=false`: print iteration info
"""
function pfi_solver(spec::DSGESpec{T};
                    degree::Int=5,
                    grid::Symbol=:auto,
                    smolyak_mu::Int=3,
                    quadrature::Symbol=:auto,
                    n_quad::Int=5,
                    scale::Real=3.0,
                    tol::Real=1e-8,
                    max_iter::Int=500,
                    damping::Real=1.0,
                    verbose::Bool=false) where {T<:AbstractFloat}

    n_eq = spec.n_endog
    n_eps = spec.n_exog
    ss = spec.steady_state

    # Step 1: Setup (identical to collocation)
    ld = linearize(spec)
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)

    nx > 0 || throw(ArgumentError("Model has no state variables — PFI requires at least one"))

    if grid == :auto
        grid = nx <= 4 ? :tensor : :smolyak
    end
    if quadrature == :auto
        quadrature = n_eps <= 2 ? :gauss_hermite : :monomial
    end

    # State bounds
    state_bounds = _compute_state_bounds(spec, ld, state_idx, scale)

    # Grid
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

    nodes_phys = _scale_from_unit(nodes_unit, state_bounds)
    basis_matrix = Matrix{T}(_chebyshev_basis_multi(nodes_unit, multi_indices))

    # Quadrature
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

    # Step 2: Initial guess from first-order perturbation
    result_1st = gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi)
    G1 = result_1st.G1
    impact = result_1st.impact

    coeffs = zeros(T, n_vars, n_basis)
    for v in 1:n_vars
        y_nodes = zeros(T, n_nodes)
        for j in 1:n_nodes
            x_dev = nodes_phys[j, :] .- ss[state_idx]
            y_nodes[j] = dot(G1[v, state_idx], x_dev)
        end
        coeffs[v, :] = basis_matrix \ y_nodes
    end

    state_bounds_T = Matrix{T}(state_bounds)
    nodes_phys_T = Matrix{T}(nodes_phys)

    # Step 3: Time iteration loop
    converged = false
    iter = 0
    sup_norm = T(Inf)

    for k in 1:max_iter
        iter = k

        # (a) Evaluate current policy at all grid points (deviations → levels)
        y_current_nodes = zeros(T, n_nodes, n_eq)
        for j in 1:n_nodes
            for v in 1:n_vars
                y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs[v, :])) + ss[v]
            end
        end

        # (b) Compute expected next-period values via quadrature
        E_y_lead = _pfi_compute_expectations(coeffs, n_vars, n_basis,
                                              state_idx, spec,
                                              quad_nodes, quad_weights,
                                              state_bounds_T, multi_indices, ss,
                                              y_current_nodes, impact)

        # (c) Solve Euler equation at each grid point
        y_new_nodes = zeros(T, n_nodes, n_eq)
        for j in 1:n_nodes
            # y_lag: the grid point represents the lagged state
            y_lag = copy(ss)
            for (ii, si) in enumerate(state_idx)
                y_lag[si] = nodes_phys_T[j, ii]
            end

            # Solve for y_t given y_lag and E[y_{t+1}]
            y_new = _pfi_euler_step(y_current_nodes[j, :], y_lag, E_y_lead[j, :], spec)
            y_new_nodes[j, :] = y_new
        end

        # (d) Refit Chebyshev coefficients (deviations from SS)
        coeffs_new = zeros(T, n_vars, n_basis)
        for v in 1:n_vars
            y_dev_nodes = y_new_nodes[:, v] .- ss[v]
            coeffs_new[v, :] = basis_matrix \ y_dev_nodes
        end

        # (e) Apply damping
        if damping < one(T)
            coeffs_new = (one(T) - T(damping)) .* coeffs .+ T(damping) .* coeffs_new
        end

        # (f) Check convergence (sup-norm on policy change at grid points)
        y_updated_nodes = zeros(T, n_nodes, n_eq)
        for j in 1:n_nodes
            for v in 1:n_vars
                y_updated_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
            end
        end
        sup_norm = maximum(abs.(y_updated_nodes .- y_current_nodes))

        if verbose
            println("  PFI iteration $k: sup-norm = $(sup_norm)")
        end

        coeffs = coeffs_new

        if sup_norm < tol
            converged = true
            break
        end
    end

    if !converged && verbose
        @warn "PFI solver did not converge after $max_iter iterations (sup-norm = $sup_norm)"
    end

    # Step 4: Package result (reuse ProjectionSolution with method=:pfi)
    return ProjectionSolution{T}(
        coeffs,
        state_bounds_T,
        grid,
        grid == :smolyak ? smolyak_mu : degree,
        Matrix{T}(nodes_unit),
        sup_norm,
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
        :pfi
    )
end
