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

const _VFI_MISSING_MSG =
    "vfi_solver performs Bellman value-function iteration and requires " *
    "`utility` and `beta` (from `@dsge utility:` / `beta:` or as keywords). " *
    "For Euler-equation time iteration use pfi_solver."

"""
    vfi_solver(spec::ModelSpec{T}; utility, beta, kwargs...) -> ProjectionSolution{T}

Solve a representative-agent DSGE via **value-function iteration**: at each state node
maximize ``u + \\beta E[V]``, then apply Howard (1960) policy evaluation of ``V``.
This is not Euler time iteration; that algorithm is [`pfi_solver`](@ref).

# Required (spec or keywords)
- `utility`: `u(c)` if `consumption` is set, otherwise `u(y, y_lag, ε, θ)`
- `beta`: discount factor as a `Real` or a `Symbol` in `spec.param_values`

# Optional keyword arguments
- `transition`: `(x, a, ε, θ) → x′`. When omitted, inferred from `next_state`
- `control_bounds`: `(x, θ) → (a_lo, a_hi)`. When omitted, inferred from
  `constraints` / `variable_bound` or an SS collar of width `scale`
- `next_state::Symbol=:linear`: how to infer `transition` when it is omitted
  (`:linear` = `G1`/`impact` after packing `y[states]=x`, `y[controls]=a`;
  `:residual` = drop residuals with `defines ∈ controls` — or a named `euler:` —
  and Newton-solve the leftover)
- `constraints`: `VariableBound`s used when inferring `control_bounds`
- `consumption::Union{Nothing,Symbol}=nothing`: if set, `utility` is `u(c::Real)`
- `controls`: choice-variable names; default is the non-state endogenous variables
- `outcome`: `(x, a, θ) → y` full endogenous vector; default fills states + controls
- `degree::Int=5`: Chebyshev degree used to export the policy
- `n_grid::Int=12`: uniform tensor nodes per state for the Bellman grid
- `grid::Symbol=:tensor`: only `:tensor` (or `:auto` → tensor) is supported
- `quadrature::Symbol=:auto`: `:gauss_hermite`, `:monomial`, or `:auto`
- `n_quad::Int=5`: quadrature nodes per shock dimension
- `n_choice::Int=41`: line-search points on the control box
- `scale::Real=3.0`: state bounds = SS ± scale × σ (capital is widened further)
- `tol::Real=1e-8`: sup-norm tolerance on ``V``
- `max_iter::Int=500`: maximum VFI iterations
- `damping::Real=1.0`: mixing factor on ``V`` (1 = no damping)
- `howard_steps::Int=20`: Howard policy-evaluation steps per iteration
- `threaded::Bool=false`: multi-thread the per-node maximization
- `verbose::Bool=false`: print iteration info
"""
function vfi_solver(spec::ModelSpec{T};
                    utility=nothing,
                    beta=nothing,
                    transition=nothing,
                    control_bounds=nothing,
                    next_state::Symbol=:linear,
                    constraints::Vector=Any[],
                    consumption::Union{Nothing,Symbol}=nothing,
                    controls::Union{Nothing,AbstractVector{Symbol}}=nothing,
                    outcome=nothing,
                    degree::Int=5,
                    n_grid::Int=12,
                    grid::Symbol=:auto,
                    smolyak_mu::Union{Integer,AbstractVector{<:Integer}}=3,
                    quadrature::Symbol=:auto,
                    n_quad::Int=5,
                    n_choice::Int=41,
                    scale::Real=3.0,
                    tol::Real=1e-8,
                    max_iter::Int=500,
                    damping::Real=1.0,
                    howard_steps::Int=20,
                    threaded::Bool=false,
                    verbose::Bool=false,
                    initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:AbstractFloat}

    utility === nothing && spec.bellman_utility !== nothing &&
        (utility = spec.bellman_utility)
    beta === nothing && spec.bellman_beta !== nothing &&
        (beta = spec.bellman_beta)
    consumption === nothing && spec.bellman_consumption !== nothing &&
        (consumption = spec.bellman_consumption)
    controls === nothing && !isempty(spec.bellman_controls) &&
        (controls = spec.bellman_controls)

    (utility === nothing || beta === nothing) &&
        throw(ArgumentError(_VFI_MISSING_MSG))
    next_state in (:linear, :residual) || throw(ArgumentError(
        "next_state must be :linear or :residual, got :$next_state"))

    n_choice >= 3 || throw(ArgumentError("n_choice must be ≥ 3, got $n_choice"))
    n_grid >= 3 || throw(ArgumentError("n_grid must be ≥ 3, got $n_grid"))
    howard_steps >= 0 || throw(ArgumentError("howard_steps must be ≥ 0"))

    if grid === :auto
        grid = :tensor
    end
    grid === :tensor || throw(ArgumentError(
        "vfi_solver supports grid=:tensor only (got :$grid). " *
        "Smolyak value-function iteration is not implemented; use pfi_solver " *
        "for Smolyak Euler time iteration."))
    _ = smolyak_mu

    n_eq = spec.n_endog
    n_eps = spec.n_exog
    ss = spec.steady_state
    isempty(ss) && throw(ArgumentError(
        "vfi_solver requires a computed steady state; call compute_steady_state first."))

    ld = linearize(spec)
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)
    nx > 0 || throw(ArgumentError("Model has no state variables — VFI requires at least one"))

    ctrl_names = controls === nothing ? spec.endog[control_idx] : collect(controls)
    n_ctrl = length(ctrl_names)
    n_ctrl == 1 || throw(ArgumentError(
        "vfi_solver supports one continuous control (got $(n_ctrl): $ctrl_names). " *
        "Pass controls=[:c] (or the single choice variable) and recover the rest via outcome."))
    ctrl_idx = [findfirst(==(nm), spec.endog) for nm in ctrl_names]
    any(isnothing, ctrl_idx) && throw(ArgumentError(
        "controls $ctrl_names are not all in spec.endog = $(spec.endog)"))
    ctrl_idx = Int[ctrl_idx...]

    if quadrature === :auto
        quadrature = n_eps <= 2 ? :gauss_hermite : :monomial
    end

    β = _vfi_resolve_beta(spec, beta)
    θ = spec.param_values
    ε_zero = zeros(T, n_eps)

    state_bounds = _vfi_state_bounds(spec, ld, state_idx, scale)
    state_bounds_T = Matrix{T}(state_bounds)
    grids, nodes_phys = _vfi_uniform_tensor(state_bounds_T, n_grid)
    n_nodes = size(nodes_phys, 1)
    V_shape = ntuple(_ -> n_grid, nx)

    # Chebyshev basis used only to export the policy / a Chebyshev copy of V
    nodes_unit_cheb, multi_indices = _tensor_grid(nx, degree)
    n_basis = size(multi_indices, 1)
    basis_cheb = Matrix{T}(_chebyshev_basis_multi(nodes_unit_cheb, multi_indices))
    nodes_phys_cheb = Matrix{T}(_scale_from_unit(nodes_unit_cheb, state_bounds_T))

    Sigma_e = Matrix{T}(I, max(n_eps, 1), max(n_eps, 1))
    if n_eps == 0
        quad_nodes = zeros(T, 1, 0)
        quad_weights = T[one(T)]
        quadrature = :none
    elseif quadrature === :gauss_hermite
        qn, qw = _gauss_hermite_scaled(n_quad, Sigma_e[1:n_eps, 1:n_eps])
        quad_nodes = Matrix{T}(qn)
        quad_weights = Vector{T}(qw)
    elseif quadrature === :monomial
        qn, qw = _monomial_nodes_weights(n_eps)
        quad_nodes = Matrix{T}(qn)
        quad_weights = Vector{T}(qw)
    else
        throw(ArgumentError("quadrature must be :gauss_hermite, :monomial, or :auto"))
    end

    result_1st = _gensys_qz(spec, ld)
    G1 = result_1st.G
    impact = result_1st.impact

    if transition === nothing
        transition = _vfi_infer_transition(spec, state_idx, ctrl_idx, G1, impact, next_state)
    end
    if control_bounds === nothing
        control_bounds = _vfi_infer_control_bounds(spec, ctrl_idx, scale, constraints,
                                                   G1, impact)
    end

    V = fill(_vfi_init_value(spec, utility, consumption, β), V_shape)
    V_new = similar(V)
    a_pol = zeros(T, n_nodes, n_ctrl)
    for j in 1:n_nodes
        x_dev = nodes_phys[j, :] .- ss[state_idx]
        y_lin = ss + G1[:, state_idx] * x_dev
        a_pol[j, 1] = y_lin[ctrl_idx[1]]
    end

    cart = CartesianIndices(V_shape)
    tol_T = T(tol)
    damp = T(damping)
    converged = false
    iter = 0
    sup_norm = T(Inf)

    for k in 1:max_iter
        iter = k

        if threaded && Threads.nthreads() > 1
            Threads.@threads for j in 1:n_nodes
                V_new[cart[j]], a_pol[j, :] = _vfi_maximize(
                    nodes_phys[j, :], a_pol[j, :], spec, utility, consumption,
                    transition, control_bounds, outcome, ctrl_idx, state_idx,
                    β, θ, ε_zero, V, grids, state_bounds_T,
                    quad_nodes, quad_weights, n_choice)
            end
        else
            for j in 1:n_nodes
                V_new[cart[j]], a_pol[j, :] = _vfi_maximize(
                    nodes_phys[j, :], a_pol[j, :], spec, utility, consumption,
                    transition, control_bounds, outcome, ctrl_idx, state_idx,
                    β, θ, ε_zero, V, grids, state_bounds_T,
                    quad_nodes, quad_weights, n_choice)
            end
        end

        for _h in 1:howard_steps
            Vh = copy(V_new)
            for j in 1:n_nodes
                V_new[cart[j]] = _vfi_eval_action(
                    nodes_phys[j, :], a_pol[j, :], spec, utility, consumption,
                    transition, outcome, ctrl_idx, state_idx, β, θ, ε_zero,
                    Vh, grids, state_bounds_T, quad_nodes, quad_weights)
            end
        end

        if damp < one(T)
            @. V_new = (one(T) - damp) * V + damp * V_new
        end

        sup_norm = zero(T)
        @inbounds for i in eachindex(V)
            d = abs(V_new[i] - V[i])
            if isfinite(d) && d > sup_norm
                sup_norm = d
            end
        end
        copyto!(V, V_new)

        if verbose
            @info "VFI iteration $k: ||ΔV||_∞ = $sup_norm"
        else
            @debug "VFI iteration $k: ||ΔV||_∞ = $sup_norm"
        end

        if sup_norm < tol_T
            converged = true
            break
        end
    end

    if !converged && verbose
        @warn "VFI solver did not converge after $max_iter iterations (||ΔV||_∞ = $sup_norm)"
    end

    # Export a Chebyshev policy (and a Chebyshev copy of V) on the degree-grid
    y_cheb = zeros(T, size(nodes_phys_cheb, 1), n_eq)
    V_cheb = zeros(T, size(nodes_phys_cheb, 1))
    for j in 1:size(nodes_phys_cheb, 1)
        x = nodes_phys_cheb[j, :]
        a = T[_vfi_multilinear_scalar(grids, reshape(a_pol[:, 1], V_shape), x,
                                      state_bounds_T)]
        y_cheb[j, :] = _vfi_pack_y(x, a, spec, outcome, ctrl_idx, state_idx,
                                   transition, θ, ε_zero)
        V_cheb[j] = _vfi_interp_V(V, grids, state_bounds_T, x)
    end
    coeffs = zeros(T, n_eq, n_basis)
    if initial_coeffs !== nothing && size(initial_coeffs) == (n_eq, n_basis)
        coeffs = Matrix{T}(initial_coeffs)
    end
    for v in 1:n_eq
        coeffs[v, :] = basis_cheb \ (y_cheb[:, v] .- ss[v])
    end
    V_coeffs = basis_cheb \ V_cheb

    nodes_unit = _scale_to_unit(nodes_phys, state_bounds_T)

    return ProjectionSolution{T}(
        coeffs,
        state_bounds_T,
        :tensor,
        degree,
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
        :vfi;
        value_fn=reshape(vec(V), n_nodes, 1),
        value_coefficients=V_coeffs,
    )
end

function _vfi_resolve_beta(spec::ModelSpec{T}, beta) where {T}
    if beta isa Symbol
        haskey(spec.param_values, beta) || throw(ArgumentError(
            "beta = :$beta is not in spec.param_values $(collect(keys(spec.param_values)))"))
        b = T(spec.param_values[beta])
    else
        b = T(beta)
    end
    zero(T) < b < one(T) || throw(ArgumentError("beta must lie in (0, 1), got $b"))
    return b
end

function _vfi_init_value(spec::ModelSpec{T}, utility, consumption, β::T) where {T}
    ss = spec.steady_state
    c0 = if consumption !== nothing
        i = findfirst(==(consumption), spec.endog)
        i === nothing ? ss[1] : ss[i]
    else
        ss[1]
    end
    c0 = max(c0, T(1e-8))
    u0 = try
        T(utility(c0))
    catch
        T(utility(ss, ss, zeros(T, spec.n_exog), spec.param_values))
    end
    return u0 / (one(T) - β)
end

"""
State box for Bellman VFI.

`_compute_state_bounds` uses a 10%–of–SS floor, a thin collar around the
deterministic SS. Clamping k′ into that collar makes “consume everything”
costless. Expand positive-SS states so k can fall to 20% of SS and rise to 2× SS.
"""
function _vfi_state_bounds(spec::ModelSpec{T}, ld, state_idx, scale) where {T}
    bounds = _compute_state_bounds(spec, ld, state_idx, scale)
    ss = spec.steady_state
    for (i, si) in enumerate(state_idx)
        if ss[si] > T(0.5)
            bounds[i, 1] = min(bounds[i, 1], T(0.2) * ss[si])
            bounds[i, 2] = max(bounds[i, 2], T(2.0) * ss[si])
        end
    end
    return bounds
end

function _vfi_uniform_tensor(bounds::AbstractMatrix{T}, n_grid::Int) where {T}
    nx = size(bounds, 1)
    grids = [collect(range(bounds[d, 1], bounds[d, 2]; length=n_grid)) for d in 1:nx]
    n_nodes = n_grid^nx
    nodes = zeros(T, n_nodes, nx)
    # First dimension is fastest (matches vec(reshape(..., n_grid, n_grid, ...)))
    for idx in 0:(n_nodes - 1)
        rem = idx
        for d in 1:nx
            j = rem % n_grid
            rem = div(rem, n_grid)
            nodes[idx + 1, d] = grids[d][j + 1]
        end
    end
    return grids, nodes
end

"""
    evaluate_value(sol::ProjectionSolution, x_state) -> T

Evaluate the stored Bellman value function at a state vector of levels.
Requires `sol.method === :vfi` with a nonempty `value_fn`.
"""
function evaluate_value(sol::ProjectionSolution{T}, x_state::AbstractVector) where {T}
    isempty(sol.value_fn) && throw(ArgumentError(
        "evaluate_value: no value function stored on this solution " *
        "(method = $(sol.method)). Solve with vfi_solver / method=:vfi."))
    length(x_state) == nstates(sol) || throw(ArgumentError(
        "x_state must have $(nstates(sol)) elements"))
    nx = nstates(sol)
    n_nodes = size(sol.value_fn, 1)
    n_grid = round(Int, n_nodes^(1 / nx))
    n_grid^nx == n_nodes || throw(ArgumentError(
        "evaluate_value: value_fn length $n_nodes is not a tensor power of $nx states"))
    grids = [collect(range(sol.state_bounds[d, 1], sol.state_bounds[d, 2];
                           length=n_grid)) for d in 1:nx]
    V = reshape(sol.value_fn, ntuple(_ -> n_grid, nx)...)
    return _vfi_interp_V(V, grids, sol.state_bounds, Vector{T}(x_state))
end

function _vfi_interp_V(V::AbstractArray{T}, grids::Vector{Vector{T}},
                       state_bounds::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    return _vfi_multilinear_scalar(grids, V, x, state_bounds)
end

function _vfi_multilinear_scalar(grids::Vector{Vector{T}}, V::AbstractArray{T},
                                 x::AbstractVector{T},
                                 state_bounds::AbstractMatrix{T}) where {T}
    nx = length(grids)
    idx0 = Vector{Int}(undef, nx)
    wgt = Vector{T}(undef, nx)
    pen = zero(T)
    @inbounds for d in 1:nx
        g = grids[d]
        xd = x[d]
        if xd < g[1]
            span = g[end] - g[1]
            pen += T(50) * (g[1] - xd) / max(span, eps(T))
            idx0[d] = 1
            wgt[d] = zero(T)
        elseif xd > g[end]
            span = g[end] - g[1]
            pen += T(50) * (xd - g[end]) / max(span, eps(T))
            idx0[d] = length(g) - 1
            wgt[d] = one(T)
        else
            i = searchsortedlast(g, xd)
            i = clamp(i, 1, length(g) - 1)
            idx0[d] = i
            den = g[i + 1] - g[i]
            wgt[d] = den > zero(T) ? (xd - g[i]) / den : zero(T)
        end
    end
    val = zero(T)
    ncorn = 1 << nx
    @inbounds for mask in 0:(ncorn - 1)
        wt = one(T)
        I = Vector{Int}(undef, nx)
        for d in 1:nx
            bit = (mask >> (d - 1)) & 1
            I[d] = idx0[d] + bit
            wt *= bit == 1 ? wgt[d] : (one(T) - wgt[d])
        end
        val += wt * V[I...]
    end
    return val - pen
end

function _vfi_expected_V(x::AbstractVector{T}, a::AbstractVector{T},
                         transition, θ, V, grids, state_bounds,
                         quad_nodes::AbstractMatrix{T},
                         quad_weights::AbstractVector{T}) where {T}
    ev = zero(T)
    nq = length(quad_weights)
    n_eps = size(quad_nodes, 2)
    for q in 1:nq
        w = quad_weights[q]
        iszero(w) && continue
        ε = n_eps == 0 ? T[] : vec(@view quad_nodes[q, :])
        xp = Vector{T}(transition(x, a, ε, θ))
        ev += w * _vfi_interp_V(V, grids, state_bounds, xp)
    end
    return ev
end

function _vfi_reward(x, a, spec::ModelSpec{T}, utility, consumption,
                     outcome, ctrl_idx, state_idx, transition, θ,
                     ε::AbstractVector{T}) where {T}
    if consumption !== nothing
        c = if length(a) == 1 && spec.endog[ctrl_idx[1]] === consumption
            a[1]
        else
            y = _vfi_pack_y(x, a, spec, outcome, ctrl_idx, state_idx,
                            transition, θ, ε)
            i = findfirst(==(consumption), spec.endog)
            i === nothing ? a[1] : y[i]
        end
        c = max(T(c), T(1e-12))
        return T(utility(c))
    end
    y = _vfi_pack_y(x, a, spec, outcome, ctrl_idx, state_idx, transition, θ, ε)
    y_lag = copy(spec.steady_state)
    y_lag[state_idx] = x
    return T(utility(y, y_lag, ε, θ))
end

function _vfi_pack_y(x, a, spec::ModelSpec{T}, outcome, ctrl_idx, state_idx,
                     transition, θ, ε::AbstractVector{T}) where {T}
    if outcome !== nothing
        return Vector{T}(outcome(x, a, θ))
    end
    y = copy(spec.steady_state)
    for (i, ci) in enumerate(ctrl_idx)
        i <= length(a) && (y[ci] = a[i])
    end
    xp = transition(x, a, ε, θ)
    for (i, si) in enumerate(state_idx)
        i <= length(xp) && (y[si] = xp[i])
    end
    return y
end

function _vfi_eval_action(x, a, spec, utility, consumption, transition, outcome,
                          ctrl_idx, state_idx, β, θ, ε_zero, V, grids,
                          state_bounds, quad_nodes, quad_weights)
    u = _vfi_reward(x, a, spec, utility, consumption, outcome, ctrl_idx,
                    state_idx, transition, θ, ε_zero)
    ev = _vfi_expected_V(x, a, transition, θ, V, grids, state_bounds,
                         quad_nodes, quad_weights)
    return u + β * ev
end

function _vfi_maximize(x, a_guess, spec::ModelSpec{T}, utility, consumption,
                       transition, control_bounds, outcome, ctrl_idx, state_idx,
                       β, θ, ε_zero, V, grids, state_bounds,
                       quad_nodes, quad_weights, n_choice) where {T}
    lo_hi = control_bounds(x, θ)
    lo = T(_vfi_bound_scalar(lo_hi[1]))
    hi = T(_vfi_bound_scalar(lo_hi[2]))
    hi < lo && ((lo, hi) = (hi, lo))
    hi == lo && (hi = lo + T(1e-8))

    best_val = T(-Inf)
    best_a = T[clamp(length(a_guess) >= 1 ? a_guess[1] : (lo + hi) / 2, lo, hi)]

    @inbounds for i in 1:n_choice
        t = n_choice == 1 ? T(0.5) : T(i - 1) / T(n_choice - 1)
        a = T[lo + t * (hi - lo)]
        val = _vfi_eval_action(x, a, spec, utility, consumption, transition,
                               outcome, ctrl_idx, state_idx, β, θ, ε_zero,
                               V, grids, state_bounds, quad_nodes, quad_weights)
        if isfinite(val) && val > best_val
            best_val = val
            best_a = a
        end
    end

    if !isfinite(best_val)
        best_a = T[(lo + hi) / 2]
        best_val = _vfi_eval_action(x, best_a, spec, utility, consumption,
                                    transition, outcome, ctrl_idx, state_idx,
                                    β, θ, ε_zero, V, grids, state_bounds,
                                    quad_nodes, quad_weights)
        return best_val, best_a
    end

    step = (hi - lo) / T(max(n_choice - 1, 1))
    glo = max(lo, best_a[1] - step)
    ghi = min(hi, best_a[1] + step)
    φ = T(0.5) * (sqrt(T(5)) - one(T))
    a1 = ghi - φ * (ghi - glo)
    a2 = glo + φ * (ghi - glo)
    f1 = _vfi_eval_action(x, T[a1], spec, utility, consumption, transition,
                          outcome, ctrl_idx, state_idx, β, θ, ε_zero,
                          V, grids, state_bounds, quad_nodes, quad_weights)
    f2 = _vfi_eval_action(x, T[a2], spec, utility, consumption, transition,
                          outcome, ctrl_idx, state_idx, β, θ, ε_zero,
                          V, grids, state_bounds, quad_nodes, quad_weights)
    for _ in 1:20
        (ghi - glo) < T(1e-8) * max(one(T), abs(best_a[1])) && break
        if f1 < f2
            glo = a1
            a1 = a2
            f1 = f2
            a2 = glo + φ * (ghi - glo)
            f2 = _vfi_eval_action(x, T[a2], spec, utility, consumption,
                                  transition, outcome, ctrl_idx, state_idx,
                                  β, θ, ε_zero, V, grids, state_bounds,
                                  quad_nodes, quad_weights)
        else
            ghi = a2
            a2 = a1
            f2 = f1
            a1 = ghi - φ * (ghi - glo)
            f1 = _vfi_eval_action(x, T[a1], spec, utility, consumption,
                                  transition, outcome, ctrl_idx, state_idx,
                                  β, θ, ε_zero, V, grids, state_bounds,
                                  quad_nodes, quad_weights)
        end
    end
    if f1 >= f2 && f1 > best_val
        best_val = f1
        best_a = T[a1]
    elseif f2 > best_val
        best_val = f2
        best_a = T[a2]
    end
    return best_val, best_a
end

_vfi_bound_scalar(x::Number) = x
_vfi_bound_scalar(x::AbstractArray) = x[1]

# =============================================================================
# Infer transition / control_bounds from ModelSpec (#658)
# =============================================================================

"""Indices of FOC residuals to drop for `next_state=:residual`."""
function _vfi_foc_indices(spec::ModelSpec, ctrl_names::Vector{Symbol})
    foc = Int[]
    missing = Symbol[]
    for c in ctrl_names
        idxs = findall(eq -> eq.defines === c, spec.equations)
        if length(idxs) == 1
            push!(foc, idxs[1])
        elseif length(idxs) > 1
            throw(ArgumentError(
                "next_state=:residual: multiple equations define control :$c"))
        else
            push!(missing, c)
        end
    end
    if !isempty(missing)
        euler_idxs = findall(eq -> eq.name === :euler, spec.equations)
        if length(missing) == 1 && length(euler_idxs) == 1 && !(euler_idxs[1] in foc)
            push!(foc, euler_idxs[1])
        else
            throw(ArgumentError(
                "next_state=:residual requires a defining equation for each " *
                "control (defines ∈ $ctrl_names) or a named euler:; missing $missing"))
        end
    end
    return sort!(unique!(foc))
end

"""
Newton-solve leftover residuals, holding `y[fixed]` and updating `y[free]`.
`y_lead` is the current `y` (certainty-equivalent, same as PFI `:nonlinear`).
"""
function _vfi_newton_keep(y::Vector{T}, y_lag::Vector{T}, spec::ModelSpec{T},
                          ε::AbstractVector{T}, keep::Vector{Int},
                          free::Vector{Int};
                          newton_tol::Real=1e-10, newton_max::Int=50) where {T}
    θ = spec.param_values
    n_k = length(keep)
    n_f = length(free)
    h = max(T(1e-7), sqrt(eps(T)))
    y = copy(y)
    R = zeros(T, n_k)
    J = zeros(T, n_k, n_f)
    for _ in 1:newton_max
        for (i, ieq) in enumerate(keep)
            try
                R[i] = spec.residual_fns[ieq](y, y_lag, y, ε, θ)
            catch e
                (e isa DomainError || e isa InexactError) || rethrow(e)
                R[i] = T(1e10)
            end
        end
        maximum(abs, R) < newton_tol && return y
        @inbounds for j in 1:n_f
            yj = y[free[j]]
            y[free[j]] = yj + h
            for (i, ieq) in enumerate(keep)
                try
                    R_plus = spec.residual_fns[ieq](y, y_lag, y, ε, θ)
                    J[i, j] = (R_plus - R[i]) / h
                catch e
                    (e isa DomainError || e isa InexactError) || rethrow(e)
                    J[i, j] = T(1e10)
                end
            end
            y[free[j]] = yj
        end
        if n_f == 1
            abs(J[1, 1]) > eps(T) || break
            y[free[1]] -= R[1] / J[1, 1]
        else
            y[free] .+= -(robust_inv(J) * R)
        end
    end
    return y
end

function _vfi_infer_transition(spec::ModelSpec{T}, state_idx::Vector{Int},
                               ctrl_idx::Vector{Int}, G1::AbstractMatrix{T},
                               impact::AbstractMatrix{T},
                               next_state::Symbol) where {T}
    ss = spec.steady_state
    nx = length(state_idx)
    if next_state === :linear
        return function (x, a, ε, _)
            y = copy(ss)
            @inbounds for (i, si) in enumerate(state_idx)
                i <= length(x) && (y[si] = x[i])
            end
            @inbounds for (i, ci) in enumerate(ctrl_idx)
                i <= length(a) && (y[ci] = a[i])
            end
            x′ = zeros(T, nx)
            @inbounds for (i, si) in enumerate(state_idx)
                val = ss[si]
                for j in eachindex(y)
                    val += G1[si, j] * (y[j] - ss[j])
                end
                for k in eachindex(ε)
                    val += impact[si, k] * ε[k]
                end
                x′[i] = val
            end
            return x′
        end
    elseif next_state === :residual
        ctrl_names = spec.endog[ctrl_idx]
        foc = _vfi_foc_indices(spec, collect(Symbol, ctrl_names))
        keep = setdiff(collect(1:spec.n_endog), foc)
        free = setdiff(collect(1:spec.n_endog), ctrl_idx)
        length(keep) == length(free) || throw(ArgumentError(
            "next_state=:residual: leftover residuals ($(length(keep))) ≠ " *
            "free variables ($(length(free))) after dropping control FOCs"))
        ε_buf_n = spec.n_exog
        return function (x, a, ε, _)
            y_lag = copy(ss)
            y = copy(ss)
            @inbounds for (i, si) in enumerate(state_idx)
                if i <= length(x)
                    y_lag[si] = x[i]
                    y[si] = x[i]
                end
            end
            @inbounds for (i, ci) in enumerate(ctrl_idx)
                i <= length(a) && (y[ci] = a[i])
            end
            ε_use = length(ε) == ε_buf_n ? Vector{T}(ε) : T[ε...; zeros(T, ε_buf_n - length(ε))]
            y = _vfi_newton_keep(y, y_lag, spec, ε_use, keep, free)
            return T[y[si] for si in state_idx]
        end
    else
        throw(ArgumentError("next_state must be :linear or :residual, got :$next_state"))
    end
end

"""
Infer a box on the single VFI control: `variable_bound` / `constraints` when
present, otherwise an SS collar of half-width `scale × σ` (Lyapunov, with floor).
"""
function _vfi_infer_control_bounds(spec::ModelSpec{T}, ctrl_idx::Vector{Int},
                                   scale::Real, constraints::Vector,
                                   G1::AbstractMatrix{T},
                                   impact::AbstractMatrix{T}) where {T}
    ci = ctrl_idx[1]
    ss_c = spec.steady_state[ci]
    lo = T(-Inf)
    hi = T(Inf)
    cname = spec.endog[ci]
    for c in constraints
        if c isa VariableBound && c.var_name === cname
            c.lower !== nothing && (lo = T(c.lower))
            c.upper !== nothing && (hi = T(c.upper))
        end
    end
    Var_y = try
        solve_lyapunov(G1, impact)
    catch
        zeros(T, spec.n_endog, spec.n_endog)
    end
    sigma_c = sqrt(max(Var_y[ci, ci], zero(T)))
    half = max(T(scale) * sigma_c, T(0.5) * abs(ss_c), T(0.1))
    if !isfinite(lo)
        lo = ss_c > zero(T) ? max(T(1e-8), ss_c - half) : ss_c - half
    end
    if !isfinite(hi)
        hi = ss_c + half
        ss_c > zero(T) && (hi = max(hi, T(2) * ss_c))
    end
    hi <= lo && (hi = lo + T(1e-8))
    lo_v = T[lo]
    hi_v = T[hi]
    return (x, θ) -> (lo_v, hi_v)
end
