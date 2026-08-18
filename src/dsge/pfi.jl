# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Policy Function Iteration (Time Iteration)
#
# References:
#   Coleman (1990), Solving the Stochastic Growth Model
#   Judd (1998), Numerical Methods in Economics, Ch. 11
#   Heer & Maussner (2009), Dynamic General Equilibrium Modeling

"""
    _pfi_euler_step(y_guess, y_lag, E_y_lead, spec; ε) -> Vector{T}

Solve the Euler equation at one grid point via Newton iteration.

Given lagged state `y_lag` (levels) and expected next-period variables `E_y_lead` (levels),
find `y_t` (levels) such that `F(y_t, y_lag, E_y_lead, ε, θ) = 0`.
`ε` defaults to zero (certainty-equivalent current period).
"""
function _pfi_euler_step(y_guess::Vector{T}, y_lag::Vector{T},
                          E_y_lead::Vector{T}, spec::ModelSpec{T};
                          ε::AbstractVector{T}=zeros(T, spec.n_exog),
                          newton_tol::Real=1e-10, newton_max::Int=50) where {T}
    n_eq = spec.n_endog
    θ = spec.param_values
    ε_use = Vector{T}(ε)

    y = copy(y_guess)
    h = max(T(1e-7), sqrt(eps(T)))

    for _ in 1:newton_max
        # Evaluate residuals
        R = zeros(T, n_eq)
        for i in 1:n_eq
            try
                R[i] = spec.residual_fns[i](y, y_lag, E_y_lead, ε_use, θ)
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
        @inbounds for j in 1:n_eq
            y_plus = copy(y)
            y_plus[j] += h
            for i in 1:n_eq
                try
                    R_plus = spec.residual_fns[i](y_plus, y_lag, E_y_lead, ε_use, θ)
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
    _pfi_next_state(y_current, y_lag, ε, state_idx, spec, impact, next_state) -> Vector

Next-period state levels at a grid point under one quadrature shock.

- `:linear` — current policy states plus first-order `impact * ε` (default)
- `:policy` — current policy states only (no linearized shock)
- `:nonlinear` — Newton-solve `F(y, y_lag, y_current, ε, θ) = 0` and take `y[state]`
"""
function _pfi_next_state(y_current::AbstractVector{T}, y_lag::AbstractVector{T},
                         ε::AbstractVector{T}, state_idx::Vector{Int},
                         spec::ModelSpec{T}, impact::AbstractMatrix{T},
                         next_state::Symbol) where {T}
    nx = length(state_idx)
    x_next = zeros(T, nx)
    if next_state === :nonlinear
        y_sh = _pfi_euler_step(Vector{T}(y_current), Vector{T}(y_lag),
                               Vector{T}(y_current), spec; ε=ε)
        for (ii, si) in enumerate(state_idx)
            x_next[ii] = y_sh[si]
        end
    else
        for (ii, si) in enumerate(state_idx)
            x_next[ii] = y_current[si]
        end
        if next_state === :linear
            n_eps = length(ε)
            for (ii, si) in enumerate(state_idx)
                for k in 1:n_eps
                    x_next[ii] += impact[si, k] * ε[k]
                end
            end
        elseif next_state !== :policy
            throw(ArgumentError("next_state must be :linear, :policy, or :nonlinear, got :$next_state"))
        end
    end
    return x_next
end

"""
    _pfi_compute_expectations(...) -> Matrix

`E[y'] = Σ_q w_q · policy(x'(x_j, ε_q))` at every grid point.
`next_state` selects the map `x'` (see [`_pfi_next_state`](@ref)).
"""
function _pfi_compute_expectations(coeffs::Matrix{T}, n_vars::Int, n_basis::Int,
                                    state_idx::Vector{Int}, spec::ModelSpec{T},
                                    quad_nodes::Matrix{T}, quad_weights::Vector{T},
                                    state_bounds::Matrix{T}, multi_indices::Matrix{Int},
                                    steady_state::Vector{T},
                                    y_current_nodes::Matrix{T},
                                    impact::Matrix{T},
                                    nodes_phys::Matrix{T};
                                    next_state::Symbol=:linear,
                                    threaded::Bool=false) where {T}
    n_nodes = size(y_current_nodes, 1)
    n_eq = spec.n_endog
    n_eps = spec.n_exog
    nx = length(state_idx)
    n_quad = length(quad_weights)

    E_y_lead = zeros(T, n_nodes, n_eq)
    ss = steady_state

    function one_node!(j::Int)
        y_lag = copy(ss)
        for (ii, si) in enumerate(state_idx)
            y_lag[si] = nodes_phys[j, ii]
        end
        acc = zeros(T, n_eq)
        @inbounds for q in 1:n_quad
            iszero(quad_weights[q]) && continue
            ε = n_eps == 0 ? T[] : vec(@view quad_nodes[q, :])
            x_next_level = _pfi_next_state(view(y_current_nodes, j, :), y_lag, ε,
                                           state_idx, spec, impact, next_state)
            for d in 1:nx
                x_next_level[d] = clamp(x_next_level[d], state_bounds[d, 1], state_bounds[d, 2])
            end
            z_next = _scale_to_unit(x_next_level, state_bounds)
            z_next = clamp.(z_next, T(-1), T(1))
            B_next = _chebyshev_basis_multi(reshape(z_next, 1, nx), multi_indices)
            for v in 1:n_vars
                acc[v] += quad_weights[q] *
                    (dot(@view(B_next[1, :]), @view(coeffs[v, :])) + steady_state[v])
            end
        end
        E_y_lead[j, :] = acc
        return nothing
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads for j in 1:n_nodes
            one_node!(j)
        end
    else
        for j in 1:n_nodes
            one_node!(j)
        end
    end

    return E_y_lead
end

"""
    pfi_solver(spec::ModelSpec{T}; kwargs...) -> ProjectionSolution{T}

Solve DSGE model via Policy Function Iteration (Time Iteration).

At each iteration: (1) compute expected next-period values using quadrature,
(2) solve Euler equation at each grid point via Newton, (3) refit Chebyshev
coefficients via least squares.

# Keyword Arguments
- `degree::Int=5`: Chebyshev polynomial degree
- `grid::Symbol=:auto`: `:tensor`, `:smolyak`, or `:auto`
- `smolyak_mu=3`: Smolyak exactness level. A scalar gives the isotropic rule `|l|₁ ≤ μ`;
  an `nx`-vector `(μ_1,…,μ_nx)` gives the anisotropic rule `Σ_k l_k/μ_k ≤ 1`.
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `scale::Real=3.0`: state bounds = SS ± scale × σ
- `tol::Real=1e-8`: sup-norm convergence tolerance
- `max_iter::Int=500`: maximum PFI iterations
- `damping::Real=1.0`: policy mixing factor (1.0 = no damping)
- `anderson_m::Int=0`: Anderson acceleration depth (0 = disabled)
- `howard_steps::Int=0`: extra Euler re-solves per iteration with the policy held
  (Coleman time-iteration analogue of Howard; 0 = one Euler pass per outer step)
- `next_state::Symbol=:linear`: map from current policy + shock to `x'`
  (`:linear` = policy states + gensys `impact`; `:policy` = policy states only;
  `:nonlinear` = Newton-solve the residuals at the quadrature shock)
- `threaded::Bool=false`: multi-thread the per-node Euler solve **and** the
  expectation quadrature
- `verbose::Bool=false`: print iteration info
- `initial_coeffs`: optional `n_vars × n_basis` warm-start coefficients
"""
function pfi_solver(spec::ModelSpec{T};
                    degree::Int=5,
                    grid::Symbol=:auto,
                    smolyak_mu::Union{Integer,AbstractVector{<:Integer}}=3,
                    quadrature::Symbol=:auto,
                    n_quad::Int=5,
                    scale::Real=3.0,
                    tol::Real=1e-8,
                    max_iter::Int=500,
                    damping::Real=1.0,
                    anderson_m::Int=0,
                    howard_steps::Int=0,
                    next_state::Symbol=:linear,
                    threaded::Bool=false,
                    verbose::Bool=false,
                    initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:AbstractFloat}
    next_state in (:linear, :policy, :nonlinear) || throw(ArgumentError(
        "next_state must be :linear, :policy, or :nonlinear, got :$next_state"))
    howard_steps >= 0 || throw(ArgumentError("howard_steps must be ≥ 0"))

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

    # Grid. `smolyak_mu` may be a scalar (isotropic) or an nx-vector (anisotropic).
    smolyak_level_set = Vector{Vector{Int}}()
    if grid == :tensor
        nodes_unit, multi_indices = _tensor_grid(nx, degree)
    elseif grid == :smolyak
        smolyak_level_set = _smolyak_admissible_levels(_smolyak_level_vector(nx, smolyak_mu))
        nodes_unit, multi_indices = _smolyak_grid_from_levels(smolyak_level_set)
    else
        throw(ArgumentError("grid must be :tensor, :smolyak, or :auto"))
    end
    smolyak_level_matrix = if grid == :smolyak
        L = zeros(Int, length(smolyak_level_set), nx)
        for (i, l) in enumerate(smolyak_level_set)
            L[i, :] = l
        end
        L
    else
        zeros(Int, 0, 0)
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
    result_1st = _gensys_qz(spec, ld)
    G1 = result_1st.G
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

    # Step 3: Time iteration loop
    converged = false
    iter = 0
    sup_norm = T(Inf)

    # Anderson acceleration history
    anderson_history = anderson_m > 0 ? Vector{T}[] : nothing
    anderson_residuals = anderson_m > 0 ? Vector{T}[] : nothing

    # Pre-allocate buffers for iteration loop
    y_current_nodes = zeros(T, n_nodes, n_eq)
    y_new_nodes = zeros(T, n_nodes, n_eq)
    y_updated_nodes = zeros(T, n_nodes, n_eq)
    coeffs_new = zeros(T, n_vars, n_basis)

    for k in 1:max_iter
        iter = k

        # (a) Evaluate current policy at all grid points (deviations → levels)
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
                                              y_current_nodes, impact,
                                              nodes_phys_T;
                                              next_state=next_state,
                                              threaded=threaded)

        # (c) Solve Euler equation at each grid point
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
        fill!(coeffs_new, zero(T))
        for v in 1:n_vars
            y_dev_nodes = y_new_nodes[:, v] .- ss[v]
            coeffs_new[v, :] = basis_matrix \ y_dev_nodes
        end

        # (e) Apply damping
        if damping < one(T)
            coeffs_new .= (one(T) - T(damping)) .* coeffs .+ T(damping) .* coeffs_new
        end

        # Extra Euler re-solves with the policy held (time-iteration Howard)
        for _h in 1:howard_steps
            for j in 1:n_nodes
                for v in 1:n_vars
                    y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
                end
            end
            E_h = _pfi_compute_expectations(coeffs_new, n_vars, n_basis,
                                            state_idx, spec,
                                            quad_nodes, quad_weights,
                                            state_bounds_T, multi_indices, ss,
                                            y_current_nodes, impact,
                                            nodes_phys_T;
                                            next_state=next_state,
                                            threaded=threaded)
            if threaded && Threads.nthreads() > 1
                Threads.@threads for j in 1:n_nodes
                    y_lag = copy(ss)
                    for (ii, si) in enumerate(state_idx)
                        y_lag[si] = nodes_phys_T[j, ii]
                    end
                    y_new_nodes[j, :] = _pfi_euler_step(y_current_nodes[j, :], y_lag,
                                                        E_h[j, :], spec)
                end
            else
                for j in 1:n_nodes
                    y_lag = copy(ss)
                    for (ii, si) in enumerate(state_idx)
                        y_lag[si] = nodes_phys_T[j, ii]
                    end
                    y_new_nodes[j, :] = _pfi_euler_step(y_current_nodes[j, :], y_lag,
                                                        E_h[j, :], spec)
                end
            end
            for v in 1:n_vars
                coeffs_new[v, :] = basis_matrix \ (y_new_nodes[:, v] .- ss[v])
            end
        end

        # Anderson acceleration
        if anderson_m > 0
            coeffs_vec_new = vec(coeffs_new)
            coeffs_vec_old = vec(coeffs)
            residual_vec = coeffs_vec_new .- coeffs_vec_old

            push!(anderson_history, copy(coeffs_vec_old))
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

        # (f) Check convergence (sup-norm on policy change at grid points)
        for j in 1:n_nodes
            for v in 1:n_vars
                y_updated_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
            end
        end
        sup_norm = maximum(abs.(y_updated_nodes .- y_current_nodes))

        if verbose
            @info "PFI iteration $k: sup-norm = $(sup_norm)"
        else
            @debug "PFI iteration $k: sup-norm = $(sup_norm)"
        end

        coeffs .= coeffs_new

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
        grid == :smolyak ? maximum(maximum.(smolyak_level_set)) : degree,
        Matrix{T}(nodes_unit),
        sup_norm,
        n_basis,
        multi_indices,
        quadrature,
        spec,
        ld,
        impact,
        ss,
        state_idx,
        control_idx,
        converged,
        iter,
        :pfi;
        smolyak_levels=smolyak_level_matrix
    )
end
