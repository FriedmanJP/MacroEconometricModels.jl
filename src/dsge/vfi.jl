# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Value Function Iteration (VFI)
#
# References:
#   Stokey, Lucas, Prescott (1989), Recursive Methods in Economic Dynamics
#   Howard (1960), Dynamic Programming and Markov Processes
#   Judd (1998), Numerical Methods in Economics, Ch. 12
#   Santos & Rust (2003), Convergence Properties of Policy Iteration

"""
    vfi_solver(spec::DSGESpec{T}; kwargs...) -> ProjectionSolution{T}

Solve DSGE model via Value Function Iteration.

Iterates on the Bellman operator using Euler equation residuals:
at each grid point, solve for the policy via Newton on `residual_fns`,
then update the value (Chebyshev coefficients) and check sup-norm
convergence on the policy function.

# Keyword Arguments
- `degree::Int=5`: Chebyshev polynomial degree
- `grid::Symbol=:auto`: `:tensor`, `:smolyak`, or `:auto`
- `smolyak_mu::Int=3`: Smolyak exactness level
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `scale::Real=3.0`: state bounds = SS ± scale × σ
- `tol::Real=1e-8`: sup-norm convergence tolerance on value function
- `max_iter::Int=1000`: maximum VFI iterations
- `damping::Real=1.0`: coefficient mixing factor (1.0 = no damping)
- `howard_steps::Int=0`: Howard improvement steps per iteration (0 = pure VFI)
- `anderson_m::Int=0`: Anderson acceleration depth (0 = disabled)
- `threaded::Bool=false`: enable multi-threaded grid evaluation
- `verbose::Bool=false`: print iteration info
- `initial_coeffs`: warm-start coefficients (n_vars × n_basis)
"""
function vfi_solver(spec::DSGESpec{T};
                    degree::Int=5,
                    grid::Symbol=:auto,
                    smolyak_mu::Int=3,
                    quadrature::Symbol=:auto,
                    n_quad::Int=5,
                    scale::Real=3.0,
                    tol::Real=1e-8,
                    max_iter::Int=1000,
                    damping::Real=1.0,
                    howard_steps::Int=0,
                    anderson_m::Int=0,
                    threaded::Bool=false,
                    verbose::Bool=false,
                    initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:AbstractFloat}

    n_eq = spec.n_endog
    n_eps = spec.n_exog
    ss = spec.steady_state

    # Step 1: Setup (identical to collocation/PFI)
    ld = linearize(spec)
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)

    nx > 0 || throw(ArgumentError("Model has no state variables — VFI requires at least one"))

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

    if initial_coeffs !== nothing && size(initial_coeffs) == (n_vars, n_basis)
        coeffs = Matrix{T}(initial_coeffs)
    else
        coeffs = zeros(T, n_vars, n_basis)
        for v in 1:n_vars
            y_nodes = zeros(T, n_nodes)
            for j in 1:n_nodes
                x_dev = nodes_phys[j, :] .- ss[state_idx]
                y_nodes[j] = dot(G1[v, state_idx], x_dev)
            end
            coeffs[v, :] = basis_matrix \ y_nodes
        end
    end

    state_bounds_T = Matrix{T}(state_bounds)
    nodes_phys_T = Matrix{T}(nodes_phys)

    # Pre-allocate buffers
    y_current_nodes = zeros(T, n_nodes, n_eq)
    y_new_nodes = zeros(T, n_nodes, n_eq)
    y_updated_nodes = zeros(T, n_nodes, n_eq)
    coeffs_new = zeros(T, n_vars, n_basis)

    # Anderson acceleration history
    anderson_history = anderson_m > 0 ? Vector{T}[] : nothing
    anderson_residuals = anderson_m > 0 ? Vector{T}[] : nothing

    # Step 3: VFI iteration loop
    converged = false
    iter = 0
    sup_norm = T(Inf)

    for k in 1:max_iter
        iter = k

        # (a) Evaluate current policy at all grid points (deviations -> levels)
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

        # (c) Solve Euler equation at each grid point (Bellman step)
        if threaded && Threads.nthreads() > 1
            Threads.@threads for j in 1:n_nodes
                y_lag = copy(ss)
                for (ii, si) in enumerate(state_idx)
                    y_lag[si] = nodes_phys_T[j, ii]
                end
                y_new = _pfi_euler_step(y_current_nodes[j, :], y_lag, E_y_lead[j, :], spec)
                y_new_nodes[j, :] = y_new
            end
        else
            for j in 1:n_nodes
                y_lag = copy(ss)
                for (ii, si) in enumerate(state_idx)
                    y_lag[si] = nodes_phys_T[j, ii]
                end
                y_new = _pfi_euler_step(y_current_nodes[j, :], y_lag, E_y_lead[j, :], spec)
                y_new_nodes[j, :] = y_new
            end
        end

        # (d) Refit Chebyshev coefficients (deviations from SS)
        for v in 1:n_vars
            coeffs_new[v, :] = basis_matrix \ (y_new_nodes[:, v] .- ss[v])
        end

        # (e) Apply damping
        if damping < one(T)
            coeffs_new .= (one(T) - T(damping)) .* coeffs .+ T(damping) .* coeffs_new
        end

        # (f) Howard improvement steps — hold policy fixed, re-evaluate
        for _h in 1:howard_steps
            for j in 1:n_nodes
                for v in 1:n_vars
                    y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
                end
            end

            E_y_lead_h = _pfi_compute_expectations(coeffs_new, n_vars, n_basis,
                                                     state_idx, spec,
                                                     quad_nodes, quad_weights,
                                                     state_bounds_T, multi_indices, ss,
                                                     y_current_nodes, impact)

            for j in 1:n_nodes
                y_lag = copy(ss)
                for (ii, si) in enumerate(state_idx)
                    y_lag[si] = nodes_phys_T[j, ii]
                end
                y_new = _pfi_euler_step(y_current_nodes[j, :], y_lag, E_y_lead_h[j, :], spec)
                y_new_nodes[j, :] = y_new
            end

            for v in 1:n_vars
                coeffs_new[v, :] = basis_matrix \ (y_new_nodes[:, v] .- ss[v])
            end
        end

        # (g) Anderson acceleration
        if anderson_m > 0
            coeffs_vec = vec(coeffs_new)
            coeffs_old_vec = vec(coeffs)
            residual_vec = coeffs_vec .- coeffs_old_vec

            push!(anderson_history, copy(coeffs_old_vec))
            push!(anderson_residuals, copy(residual_vec))

            if length(anderson_history) >= 2
                coeffs_mixed = _anderson_step(anderson_history, anderson_residuals, anderson_m)
                coeffs_new .= reshape(coeffs_mixed, n_vars, n_basis)
            end

            while length(anderson_history) > anderson_m + 1
                popfirst!(anderson_history)
                popfirst!(anderson_residuals)
            end
        end

        # (h) Check convergence (sup-norm on policy change at grid points)
        for j in 1:n_nodes
            for v in 1:n_vars
                y_updated_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
            end
        end

        # Recompute y_current_nodes with old coeffs for comparison
        for j in 1:n_nodes
            for v in 1:n_vars
                y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs[v, :])) + ss[v]
            end
        end
        sup_norm = maximum(abs.(y_updated_nodes .- y_current_nodes))

        if verbose
            println("  VFI iteration $k: sup-norm = $(sup_norm)")
        end

        coeffs .= coeffs_new

        if sup_norm < tol
            converged = true
            break
        end
    end

    if !converged && verbose
        @warn "VFI solver did not converge after $max_iter iterations (sup-norm = $sup_norm)"
    end

    # Step 4: Package result
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
        :vfi
    )
end
