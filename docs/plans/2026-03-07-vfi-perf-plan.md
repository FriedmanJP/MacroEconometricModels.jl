# VFI Solver + Performance Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Value Function Iteration (VFI) solver for DSGE models and optimize all three global nonlinear solvers (VFI, PFI, collocation) for performance.

**Architecture:** VFI reuses the existing Chebyshev grid/basis/quadrature infrastructure from `projection.jl`. A shared Anderson acceleration utility serves both VFI and PFI. Threading is opt-in via `threaded::Bool=false`. All solvers return `ProjectionSolution{T}`.

**Tech Stack:** Julia, LinearAlgebra, existing `DSGESpec`/`ProjectionSolution` types, Chebyshev polynomials, Gauss-Hermite/monomial quadrature.

---

### Task 1: Anderson Acceleration Utility

**Files:**
- Create: `src/dsge/anderson.jl`
- Modify: `src/MacroEconometricModels.jl:191` (add include after `pfi.jl`)
- Test: `test/dsge/test_dsge.jl` (append)

**Step 1: Write the failing test**

Append to `test/dsge/test_dsge.jl`, inside a new `@testset "Anderson Acceleration"` block after the PFI section (after line ~4509):

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section: Anderson Acceleration
# ─────────────────────────────────────────────────────────────────────────────

@testset "Anderson Acceleration" begin

@testset "Anderson step on simple fixed point" begin
    # Fixed-point problem: x = 0.5*x + 1 → x* = 2
    # Store history of iterates and residuals
    T_fp = Float64
    history = Vector{T_fp}[]
    residuals = Vector{T_fp}[]

    x = [0.0]  # initial guess
    for i in 1:5
        g_x = [0.5 * x[1] + 1.0]  # g(x)
        r = g_x .- x               # residual = g(x) - x
        push!(history, copy(x))
        push!(residuals, copy(r))
        x = g_x  # simple iteration
    end

    # Anderson step with m=3 should produce a valid iterate
    x_anderson = MacroEconometricModels._anderson_step(history, residuals, 3)
    @test length(x_anderson) == 1
    @test isfinite(x_anderson[1])
    # Should be closer to fixed point x*=2 than last simple iterate
    @test abs(x_anderson[1] - 2.0) < abs(x[1] - 2.0)
end

@testset "Anderson step m=1 is valid" begin
    history = [[1.0], [1.5]]
    residuals = [[0.5], [0.25]]
    x_a = MacroEconometricModels._anderson_step(history, residuals, 1)
    @test length(x_a) == 1
    @test isfinite(x_a[1])
end

@testset "Anderson step with multidimensional vectors" begin
    history = [[1.0, 2.0], [1.5, 1.8], [1.7, 1.9]]
    residuals = [[0.5, -0.2], [0.2, 0.1], [0.1, 0.05]]
    x_a = MacroEconometricModels._anderson_step(history, residuals, 2)
    @test length(x_a) == 2
    @test all(isfinite.(x_a))
end

end # Anderson Acceleration
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e 'using MacroEconometricModels; MacroEconometricModels._anderson_step([[1.0]], [[0.5]], 1)'`
Expected: ERROR — `_anderson_step` not defined

**Step 3: Write the implementation**

Create `src/dsge/anderson.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Anderson Acceleration for Fixed-Point Iterations
#
# References:
#   Walker & Ni (2011), Anderson Acceleration for Fixed-Point Iterations,
#     SIAM J. Numer. Anal. 49(4): 1715–1735

"""
    _anderson_step(history, residuals, m) -> Vector{T}

Compute an Anderson-accelerated iterate from the last `m` iterates.

Given iterates `x_k` and residuals `r_k = g(x_k) - x_k`, solve:

    min ‖Σ αᵢ rᵢ‖²  s.t. Σ αᵢ = 1

and return the mixed iterate `x_new = Σ αᵢ (xᵢ + rᵢ)`.

# Arguments
- `history::Vector{Vector{T}}`: previous iterates x_k
- `residuals::Vector{Vector{T}}`: corresponding residuals r_k = g(x_k) - x_k
- `m::Int`: mixing depth (use last m entries)
"""
function _anderson_step(history::Vector{Vector{T}}, residuals::Vector{Vector{T}},
                         m::Int) where {T<:AbstractFloat}
    n_hist = length(history)
    m_eff = min(m, n_hist)  # effective depth

    if m_eff <= 1
        # Fallback: just return last g(x) = x + r
        return history[end] .+ residuals[end]
    end

    # Use last m_eff entries
    start_idx = n_hist - m_eff + 1
    R = hcat(residuals[start_idx:end]...)  # n_dim × m_eff

    # Solve constrained least squares: min ||R*α||^2 s.t. 1'α = 1
    # Equivalent: min ||R*(α - α₀) + R*α₀||^2 where α₀ = e_m/m_eff
    # Use the unconstrained approach with difference matrix:
    # ΔR[:,j] = R[:,j+1] - R[:,j], then solve min ||ΔR*γ - r_last||^2
    # and recover α from γ.

    r_last = residuals[end]

    if m_eff == 2
        # Simple case: one ΔR column
        dr = R[:, 2] .- R[:, 1]
        dr_norm_sq = dot(dr, dr)
        if dr_norm_sq < eps(T)
            return history[end] .+ r_last
        end
        gamma1 = dot(dr, r_last) / dr_norm_sq
        alpha = [gamma1, one(T) - gamma1]
    else
        # General case: m_eff-1 columns in ΔR
        n_col = m_eff - 1
        n_dim = length(r_last)
        DR = zeros(T, n_dim, n_col)
        for j in 1:n_col
            DR[:, j] = R[:, j + 1] .- R[:, j]
        end

        # Solve min ||DR*γ - r_last||^2
        # Normal equations: DR'DR * γ = DR' * r_last
        DRtDR = DR' * DR
        DRtr = DR' * r_last

        # Regularize for numerical stability
        reg = max(eps(T) * T(1e4), eps(T) * norm(DRtDR))
        for i in 1:n_col
            DRtDR[i, i] += reg
        end

        gamma = DRtDR \ DRtr

        # Recover α: α_j = γ_j - γ_{j-1} (with γ_0=0), α_m = 1 - γ_{m-1}
        alpha = zeros(T, m_eff)
        alpha[1] = gamma[1]
        for j in 2:n_col
            alpha[j] = gamma[j] - gamma[j - 1]
        end
        alpha[m_eff] = one(T) - gamma[n_col]
    end

    # Mixed iterate: x_new = Σ αᵢ (xᵢ + rᵢ)
    x_new = zeros(T, length(r_last))
    for j in 1:m_eff
        idx = start_idx + j - 1
        x_new .+= alpha[j] .* (history[idx] .+ residuals[idx])
    end

    return x_new
end
```

**Step 4: Add include to `src/MacroEconometricModels.jl`**

After line 191 (`include("dsge/pfi.jl")`), add:

```julia
include("dsge/anderson.jl")
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e 'using MacroEconometricModels, Test; h=[[0.0],[0.5],[0.75],[0.875]]; r=[[0.5],[0.25],[0.125],[0.0625]]; x=MacroEconometricModels._anderson_step(h,r,3); @test isfinite(x[1]); println("PASS: anderson_step works")'`
Expected: PASS

Then run the full Anderson tests:
`cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e 'using MacroEconometricModels, Test, Random, LinearAlgebra; include("test/dsge/test_dsge.jl")' 2>&1 | tail -5`

Note: This runs the full DSGE test file — if too slow, extract just the Anderson testset into a temp file.

**Step 6: Commit**

```bash
git add src/dsge/anderson.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add Anderson acceleration utility for fixed-point iterations"
```

---

### Task 2: VFI Solver — Core Implementation

**Files:**
- Create: `src/dsge/vfi.jl`
- Modify: `src/MacroEconometricModels.jl:192` (add include after `anderson.jl`)
- Modify: `src/dsge/gensys.jl:173-175` (add `:vfi` dispatch)
- Test: `test/dsge/test_dsge.jl` (append)

**Step 1: Write the failing tests**

Append to `test/dsge/test_dsge.jl`, after the Anderson section:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section: Value Function Iteration (VFI)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Value Function Iteration" begin

@testset "Linear AR(1) VFI" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:vfi, degree=5, verbose=false)

    @test sol isa MacroEconometricModels.ProjectionSolution
    @test sol.converged
    @test sol.method == :vfi
    @test sol.residual_norm < 1e-6

    # evaluate_policy at steady state should return steady state
    y_ss = evaluate_policy(sol, [0.0])
    @test length(y_ss) == 1
    @test abs(y_ss[1]) < 1e-6

    # Linear model: VFI should recover linear policy
    pert_sol = solve(spec; method=:gensys)
    for x_val in [-0.02, -0.01, 0.0, 0.01, 0.02]
        y_vfi = evaluate_policy(sol, [x_val])
        y_pert = pert_sol.G1[1, 1] * x_val
        @test abs(y_vfi[1] - y_pert) < 1e-4
    end

    # Euler error
    euler_err = max_euler_error(sol; n_test=100, rng=Random.MersenneTwister(42))
    @test euler_err < 1e-6
end

@testset "VFI vs PFI vs projection agreement" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_vfi = solve(spec; method=:vfi, degree=5, verbose=false)
    sol_pfi = solve(spec; method=:pfi, degree=5, verbose=false)
    sol_proj = solve(spec; method=:projection, degree=5, verbose=false)

    @test sol_vfi.converged
    @test sol_pfi.converged
    @test sol_proj.converged

    for x_val in [-0.02, 0.0, 0.02]
        y_vfi = evaluate_policy(sol_vfi, [x_val])
        y_pfi = evaluate_policy(sol_pfi, [x_val])
        y_proj = evaluate_policy(sol_proj, [x_val])
        @test abs(y_vfi[1] - y_pfi[1]) < 1e-4
        @test abs(y_vfi[1] - y_proj[1]) < 1e-4
    end
end

@testset "VFI damping" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol1 = solve(spec; method=:vfi, degree=5, damping=1.0, verbose=false)
    @test sol1.converged

    sol05 = solve(spec; method=:vfi, degree=5, damping=0.5, verbose=false)
    @test sol05.converged

    y1 = evaluate_policy(sol1, [0.01])
    y05 = evaluate_policy(sol05, [0.01])
    @test abs(y1[1] - y05[1]) < 1e-4
end

@testset "VFI dispatch through solve()" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:vfi, degree=3, verbose=false)
    @test sol isa ProjectionSolution
    @test sol.method == :vfi
end

@testset "VFI show()" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:vfi, degree=3, verbose=false)
    io = IOBuffer()
    show(io, sol)
    output = String(take!(io))
    @test occursin("Converged", output)
end

@testset "VFI simulate and irf" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:vfi, degree=5, verbose=false)

    Random.seed!(42)
    Y_sim = simulate(sol, 100)
    @test size(Y_sim) == (100, 1)
    @test all(abs.(Y_sim) .< 1.0)

    irfs = irf(sol, 20; n_sim=200)
    @test irfs isa ImpulseResponse
    @test size(irfs.values) == (20, 1, 1)
    @test abs(irfs.values[1, 1, 1] - 0.01) < 0.005
end

end # VFI
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e 'using MacroEconometricModels; solve(MacroEconometricModels.DSGESpec{Float64}; method=:vfi)'`
Expected: ERROR — `:vfi` not in dispatch

**Step 3: Write the VFI solver**

Create `src/dsge/vfi.jl`:

```julia
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
            y_dev_nodes = @view(y_new_nodes[:, v]) .- ss[v]
            coeffs_new[v, :] = basis_matrix \ y_dev_nodes
        end

        # (e) Apply damping
        if damping < one(T)
            coeffs_new .= (one(T) - T(damping)) .* coeffs .+ T(damping) .* coeffs_new
        end

        # (f) Howard improvement steps — hold policy fixed, re-evaluate
        for _h in 1:howard_steps
            # Re-evaluate current policy with new coefficients
            for j in 1:n_nodes
                for v in 1:n_vars
                    y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs_new[v, :])) + ss[v]
                end
            end

            # Re-compute expectations
            E_y_lead_h = _pfi_compute_expectations(coeffs_new, n_vars, n_basis,
                                                     state_idx, spec,
                                                     quad_nodes, quad_weights,
                                                     state_bounds_T, multi_indices, ss,
                                                     y_current_nodes, impact)

            # Re-solve Euler at each grid point
            for j in 1:n_nodes
                y_lag = copy(ss)
                for (ii, si) in enumerate(state_idx)
                    y_lag[si] = nodes_phys_T[j, ii]
                end
                y_new = _pfi_euler_step(y_current_nodes[j, :], y_lag, E_y_lead_h[j, :], spec)
                y_new_nodes[j, :] = y_new
            end

            # Refit
            for v in 1:n_vars
                y_dev_nodes = @view(y_new_nodes[:, v]) .- ss[v]
                coeffs_new[v, :] = basis_matrix \ y_dev_nodes
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

            # Keep history bounded
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
```

**Step 4: Add include and dispatch**

In `src/MacroEconometricModels.jl`, after the `anderson.jl` include (added in Task 1), add:

```julia
include("dsge/vfi.jl")
```

In `src/dsge/gensys.jl`, modify lines 172-175. After the `:pfi` branch (line 173), add the `:vfi` branch:

```julia
    elseif method == :vfi
        return vfi_solver(spec; kwargs...)
```

Update the error message on line 175 to include `:vfi`:

```julia
        throw(ArgumentError("method must be :gensys, :blanchard_kahn, :klein, :perturbation, :projection, :pfi, :vfi, or :perfect_foresight"))
```

Also add to the docstring on line 138, after the `:pfi` line:

```julia
- `:vfi` -- Value Function Iteration (Stokey-Lucas-Prescott 1989); pass `degree=5`, `howard_steps=0`
```

In `src/MacroEconometricModels.jl`, find the export line with `evaluate_policy, max_euler_error` (line 393) and add `vfi_solver` to the exports. Find the nearby export block and add:

```julia
export vfi_solver
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test, Random, LinearAlgebra
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
sol = solve(spec; method=:vfi, degree=5, verbose=true)
@test sol.converged
@test sol.method == :vfi
println("VFI basic test PASSED")
'`
Expected: VFI iteration messages, then "PASSED"

**Step 6: Commit**

```bash
git add src/dsge/vfi.jl src/dsge/gensys.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add VFI solver with Howard steps and Anderson acceleration (#78)"
```

---

### Task 3: VFI Howard Steps + Anderson Tests

**Files:**
- Test: `test/dsge/test_dsge.jl` (append to VFI section)

**Step 1: Write the tests**

Append inside the `@testset "Value Function Iteration"` block:

```julia
@testset "Howard improvement steps" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_pure = solve(spec; method=:vfi, degree=5, howard_steps=0, verbose=false)
    sol_howard = solve(spec; method=:vfi, degree=5, howard_steps=5, verbose=false)

    @test sol_pure.converged
    @test sol_howard.converged

    # Howard steps should reduce iteration count
    @test sol_howard.iterations <= sol_pure.iterations

    # Both should give same policy
    for x_val in [-0.02, 0.0, 0.02]
        y_pure = evaluate_policy(sol_pure, [x_val])
        y_howard = evaluate_policy(sol_howard, [x_val])
        @test abs(y_pure[1] - y_howard[1]) < 1e-4
    end
end

@testset "VFI Anderson acceleration" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_plain = solve(spec; method=:vfi, degree=5, anderson_m=0, verbose=false)
    sol_anderson = solve(spec; method=:vfi, degree=5, anderson_m=3, verbose=false)

    @test sol_plain.converged
    @test sol_anderson.converged

    # Anderson should converge in same or fewer iterations
    @test sol_anderson.iterations <= sol_plain.iterations + 5  # allow small margin

    # Same policy
    for x_val in [-0.02, 0.0, 0.02]
        y_plain = evaluate_policy(sol_plain, [x_val])
        y_anderson = evaluate_policy(sol_anderson, [x_val])
        @test abs(y_plain[1] - y_anderson[1]) < 1e-3
    end
end

@testset "VFI Smolyak grid" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:vfi, degree=5, grid=:smolyak, smolyak_mu=2, verbose=false)
    @test sol.converged
    @test sol.grid_type == :smolyak
end

@testset "VFI threaded matches sequential" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_seq = solve(spec; method=:vfi, degree=5, threaded=false, verbose=false)
    sol_par = solve(spec; method=:vfi, degree=5, threaded=true, verbose=false)

    @test sol_seq.converged
    @test sol_par.converged

    for x_val in [-0.02, 0.0, 0.02]
        y_seq = evaluate_policy(sol_seq, [x_val])
        y_par = evaluate_policy(sol_par, [x_val])
        @test abs(y_seq[1] - y_par[1]) < 1e-6
    end
end
```

**Step 2: Run tests**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test, Random, LinearAlgebra
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
sol_pure = solve(spec; method=:vfi, degree=5, howard_steps=0, verbose=false)
sol_howard = solve(spec; method=:vfi, degree=5, howard_steps=5, verbose=false)
@test sol_pure.converged && sol_howard.converged
@test sol_howard.iterations <= sol_pure.iterations
println("Howard: pure=$(sol_pure.iterations) howard=$(sol_howard.iterations)")
sol_anderson = solve(spec; method=:vfi, degree=5, anderson_m=3, verbose=false)
@test sol_anderson.converged
println("Anderson: iters=$(sol_anderson.iterations)")
println("All Howard/Anderson tests PASSED")
'`
Expected: PASS with Howard iterations ≤ pure iterations

**Step 3: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): add VFI Howard steps, Anderson, threading, Smolyak tests"
```

---

### Task 4: PFI Performance — Anderson + Threading

**Files:**
- Modify: `src/dsge/pfi.jl:169-180` (add kwargs), `src/dsge/pfi.jl:258-321` (add threading + Anderson to loop)
- Test: `test/dsge/test_dsge.jl` (append to PFI section)

**Step 1: Write the failing tests**

Append inside the `@testset "Policy Function Iteration"` block (before its `end`):

```julia
@testset "PFI Anderson acceleration" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_plain = solve(spec; method=:pfi, degree=5, anderson_m=0, verbose=false)
    sol_anderson = solve(spec; method=:pfi, degree=5, anderson_m=3, verbose=false)

    @test sol_plain.converged
    @test sol_anderson.converged

    for x_val in [-0.02, 0.0, 0.02]
        y_plain = evaluate_policy(sol_plain, [x_val])
        y_anderson = evaluate_policy(sol_anderson, [x_val])
        @test abs(y_plain[1] - y_anderson[1]) < 1e-3
    end
end

@testset "PFI threaded matches sequential" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_seq = solve(spec; method=:pfi, degree=5, threaded=false, verbose=false)
    sol_par = solve(spec; method=:pfi, degree=5, threaded=true, verbose=false)

    @test sol_seq.converged
    @test sol_par.converged

    for x_val in [-0.02, 0.0, 0.02]
        y_seq = evaluate_policy(sol_seq, [x_val])
        y_par = evaluate_policy(sol_par, [x_val])
        @test abs(y_seq[1] - y_par[1]) < 1e-6
    end
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e 'using MacroEconometricModels; spec = @dsge begin; parameters: ρ=0.9, σ=0.01; endogenous: y; exogenous: ε; y[t]=ρ*y[t-1]+σ*ε[t]; steady_state: [0.0]; end; spec = compute_steady_state(spec); solve(spec; method=:pfi, anderson_m=3)'`
Expected: ERROR — `anderson_m` is not a valid keyword

**Step 3: Modify `pfi_solver` to accept new kwargs and use them**

In `src/dsge/pfi.jl`, modify the function signature (lines 169-180) to add three new kwargs after `verbose`:

```julia
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
                    anderson_m::Int=0,
                    threaded::Bool=false,
                    verbose::Bool=false,
                    initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:AbstractFloat}
```

After the pre-existing line `sup_norm = T(Inf)` (line 256), add Anderson history initialization:

```julia
    # Anderson acceleration history
    anderson_history = anderson_m > 0 ? Vector{T}[] : nothing
    anderson_residuals = anderson_m > 0 ? Vector{T}[] : nothing
```

In the iteration loop, replace the Euler solve block (lines 278-288) with a threaded version:

```julia
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
```

After the damping block (line ~300), before convergence check, add Anderson acceleration:

```julia
        # Anderson acceleration
        if anderson_m > 0
            coeffs_vec_new = vec(coeffs_new)
            coeffs_vec_old = vec(coeffs)
            residual_vec = coeffs_vec_new .- coeffs_vec_old

            push!(anderson_history, copy(coeffs_vec_old))
            push!(anderson_residuals, copy(residual_vec))

            if length(anderson_history) >= 2
                coeffs_mixed = _anderson_step(anderson_history, anderson_residuals, anderson_m)
                coeffs_new = reshape(coeffs_mixed, n_vars, n_basis)
            end

            while length(anderson_history) > anderson_m + 1
                popfirst!(anderson_history)
                popfirst!(anderson_residuals)
            end
        end
```

Also update the docstring for `pfi_solver` (lines 148-167) to add:

```
- `anderson_m::Int=0`: Anderson acceleration depth (0 = disabled)
- `threaded::Bool=false`: enable multi-threaded grid evaluation
```

**Step 4: Run tests**

Run the quick smoke test:
`cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
sol = solve(spec; method=:pfi, degree=5, anderson_m=3, threaded=true, verbose=false)
@test sol.converged
println("PFI anderson+threaded PASSED")
'`

**Step 5: Commit**

```bash
git add src/dsge/pfi.jl test/dsge/test_dsge.jl
git commit -m "perf(dsge): add Anderson acceleration and threading to PFI solver (#79)"
```

---

### Task 5: Collocation Performance — Threading + Pre-allocation

**Files:**
- Modify: `src/dsge/projection.jl:421-431` (add `threaded` kwarg)
- Modify: `src/dsge/projection.jl:309-398` (add threading to residual eval)
- Test: `test/dsge/test_dsge.jl` (append to Projection section)

**Step 1: Write the failing test**

Append inside the `@testset "API integration"` block (within Projection Methods):

```julia
    @testset "Projection threaded matches sequential" begin
        sol_seq = solve(spec; method=:projection, degree=3, threaded=false, verbose=false)
        sol_par = solve(spec; method=:projection, degree=3, threaded=true, verbose=false)

        @test sol_seq.converged
        @test sol_par.converged

        for x_val in [-0.02, 0.0, 0.02]
            y_seq = evaluate_policy(sol_seq, [x_val])
            y_par = evaluate_policy(sol_par, [x_val])
            @test abs(y_seq[1] - y_par[1]) < 1e-6
        end
    end
```

**Step 2: Modify `collocation_solver` signature**

Add `threaded::Bool=false` to the kwargs in `src/dsge/projection.jl` (line ~431, before `initial_coeffs`):

```julia
function collocation_solver(spec::DSGESpec{T};
                            degree::Int=5,
                            grid::Symbol=:auto,
                            smolyak_mu::Int=3,
                            quadrature::Symbol=:auto,
                            n_quad::Int=5,
                            scale::Real=3.0,
                            tol::Real=1e-8,
                            max_iter::Int=100,
                            threaded::Bool=false,
                            verbose::Bool=false,
                            initial_coeffs::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:AbstractFloat}
```

**Step 3: Thread the Jacobian computation**

The hot path in collocation is the finite-difference Jacobian (lines 545-554). Replace with:

```julia
        # Jacobian via finite differences
        n_unknowns = length(coeffs_vec)
        n_residuals = length(R)
        J = zeros(T, n_residuals, n_unknowns)
        h_fd = max(T(1e-7), sqrt(eps(T)))

        if threaded && Threads.nthreads() > 1
            Threads.@threads for i in 1:n_unknowns
                c_plus = copy(coeffs_vec)
                c_plus[i] += h_fd
                R_plus = _collocation_residual(c_plus, n_vars, n_basis,
                                                basis_matrix, nodes_phys_T,
                                                state_idx, control_idx, spec,
                                                quad_nodes, quad_weights,
                                                state_bounds_T, multi_indices, ss)
                J[:, i] = (R_plus .- R) ./ h_fd
            end
        else
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
        end
```

Also update the docstring (lines 404-419) to add:
```
- `threaded::Bool=false`: enable multi-threaded Jacobian evaluation
```

**Step 4: Run tests**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
sol = solve(spec; method=:projection, degree=3, threaded=true, verbose=false)
@test sol.converged
println("Projection threaded PASSED")
'`

**Step 5: Commit**

```bash
git add src/dsge/projection.jl test/dsge/test_dsge.jl
git commit -m "perf(dsge): add threading to collocation Jacobian computation (#79)"
```

---

### Task 6: Pre-allocation + In-place Optimizations

**Files:**
- Modify: `src/dsge/pfi.jl` (pre-allocate loop buffers)
- Modify: `src/dsge/projection.jl` (pre-allocate in `_collocation_residual`)

**Step 1: Pre-allocate PFI loop buffers**

In `src/dsge/pfi.jl`, before the iteration loop (line ~254), add buffer pre-allocation:

```julia
    # Pre-allocate buffers for iteration loop
    y_current_nodes = zeros(T, n_nodes, n_eq)
    y_new_nodes = zeros(T, n_nodes, n_eq)
    y_updated_nodes = zeros(T, n_nodes, n_eq)
    coeffs_new = zeros(T, n_vars, n_basis)
```

Then in the loop, replace `y_current_nodes = zeros(T, n_nodes, n_eq)` (line 262) and similar with in-place fills:

```julia
        # (a) Evaluate current policy at all grid points
        fill!(y_current_nodes, zero(T))
        for j in 1:n_nodes
            for v in 1:n_vars
                y_current_nodes[j, v] = dot(@view(basis_matrix[j, :]), @view(coeffs[v, :])) + ss[v]
            end
        end
```

Similarly replace `y_new_nodes = zeros(...)` (line 277) with `fill!(y_new_nodes, zero(T))`, `coeffs_new = zeros(...)` (line 291) with `fill!(coeffs_new, zero(T))`, and `y_updated_nodes = zeros(...)` (line 303) with `fill!(y_updated_nodes, zero(T))`.

**Step 2: Run existing PFI tests to verify no regression**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test, Random
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
sol = solve(spec; method=:pfi, degree=5, verbose=false)
@test sol.converged
@test sol.method == :pfi
pert_sol = solve(spec; method=:gensys)
for x_val in [-0.02, -0.01, 0.0, 0.01, 0.02]
    y_pfi = evaluate_policy(sol, [x_val])
    y_pert = pert_sol.G1[1, 1] * x_val
    @test abs(y_pfi[1] - y_pert) < 1e-4
end
println("PFI pre-alloc regression test PASSED")
'`

**Step 3: Commit**

```bash
git add src/dsge/pfi.jl src/dsge/projection.jl
git commit -m "perf(dsge): pre-allocate loop buffers in PFI and collocation (#79)"
```

---

### Task 7: Type Stability Audit + @inbounds

**Files:**
- Modify: `src/dsge/projection.jl` (inner loops)
- Modify: `src/dsge/pfi.jl` (inner loops)
- Modify: `src/dsge/vfi.jl` (inner loops)

**Step 1: Audit type stability**

Run `@code_warntype` on the hot inner functions:

```bash
cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels
# Check _chebyshev_eval
@code_warntype MacroEconometricModels._chebyshev_eval(0.5, 5)
'
```

Do the same for `_chebyshev_basis_multi`, `_scale_to_unit`, `_scale_from_unit`.

**Step 2: Add `@inbounds` to verified-safe inner loops**

In `src/dsge/projection.jl`, add `@inbounds` to the Chebyshev evaluation recurrence (lines 34-44):

```julia
function _chebyshev_eval(x::Real, degree::Int)
    vals = zeros(degree + 1)
    vals[1] = 1.0
    if degree >= 1
        vals[2] = Float64(x)
    end
    @inbounds for k in 2:degree
        vals[k + 1] = 2.0 * Float64(x) * vals[k] - vals[k - 1]
    end
    return vals
end
```

In `_chebyshev_basis_multi` (lines 53-77), add `@inbounds` to the tensor product loop:

```julia
    @inbounds for k in 1:n_basis
        for d in 1:nx
            deg = multi_indices[k, d]
            B[:, k] .*= T_vals[d][:, deg + 1]
        end
    end
```

In `_pfi_compute_expectations` (pfi.jl lines 111-142), add `@inbounds` to the quadrature loops:

```julia
    @inbounds for j in 1:n_nodes
        for q in 1:n_quad
            ...
        end
    end
```

Similarly in `_pfi_euler_step` Jacobian loop (pfi.jl lines 54-68):

```julia
        @inbounds for j in 1:n_eq
            ...
        end
```

**Step 3: Run all existing tests to verify no regression**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels, Test, Random
spec = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)
for method in [:projection, :pfi, :vfi]
    sol = solve(spec; method=method, degree=5, verbose=false)
    @test sol.converged
    println("$method: converged in $(sol.iterations) iterations")
end
println("All solvers pass after @inbounds")
'`

**Step 4: Commit**

```bash
git add src/dsge/projection.jl src/dsge/pfi.jl src/dsge/vfi.jl
git commit -m "perf(dsge): add @inbounds to verified-safe inner loops (#79)"
```

---

### Task 8: Benchmark Script

**Files:**
- Create: `benchmarks/bench_nonlinear.jl`

**Step 1: Create the benchmark script**

```bash
ls /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon/benchmarks/ 2>/dev/null || mkdir -p /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon/benchmarks
```

Create `benchmarks/bench_nonlinear.jl`:

```julia
# Benchmarks for DSGE nonlinear solvers
# Run: julia --project=. benchmarks/bench_nonlinear.jl
# Optional: julia --project=. -t 4 benchmarks/bench_nonlinear.jl  (for threading)

using MacroEconometricModels
using Random
using Printf

println("Julia threads: ", Threads.nthreads())
println()

# ─── Model Setup ───

# Simple AR(1)
spec_ar1 = @dsge begin
    parameters: ρ = 0.9, σ = 0.01
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec_ar1 = compute_steady_state(spec_ar1)

# Neoclassical growth model (2 variables, 1 shock)
spec_growth = @dsge begin
    parameters: α = 0.36, β = 0.99, δ = 0.025, γ = 2.0, σ_e = 0.01
    endogenous: k, c
    exogenous: ε
    c[t]^(-γ) - β * c[t+1]^(-γ) * (α * k[t]^(α - 1) + 1 - δ) = 0
    k[t] - k[t-1]^α - (1 - δ) * k[t-1] + c[t] - σ_e * ε[t] = 0
    steady_state = begin
        k_ss = (α / (1/β - 1 + δ))^(1 / (1 - α))
        c_ss = k_ss^α - δ * k_ss
        [k_ss, c_ss]
    end
end
spec_growth = compute_steady_state(spec_growth)

# ─── Benchmarks ───

function bench(name, f; n_runs=3)
    # Warmup
    f()
    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end
    median_t = sort(times)[div(n_runs, 2) + 1]
    @printf("  %-45s %8.3f ms (median of %d)\n", name, median_t * 1000, n_runs)
    return median_t
end

println("═══ AR(1) model (1 var, 1 shock) ═══")
bench("Projection (degree=5)") do
    solve(spec_ar1; method=:projection, degree=5, verbose=false)
end
bench("PFI (degree=5)") do
    solve(spec_ar1; method=:pfi, degree=5, verbose=false)
end
bench("VFI (degree=5)") do
    solve(spec_ar1; method=:vfi, degree=5, verbose=false)
end
bench("VFI + Howard(5)") do
    solve(spec_ar1; method=:vfi, degree=5, howard_steps=5, verbose=false)
end
bench("VFI + Anderson(3)") do
    solve(spec_ar1; method=:vfi, degree=5, anderson_m=3, verbose=false)
end
bench("VFI + Howard(5) + Anderson(3)") do
    solve(spec_ar1; method=:vfi, degree=5, howard_steps=5, anderson_m=3, verbose=false)
end

println()
println("═══ Growth model (2 vars, 1 shock) ═══")
bench("Projection (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:projection, degree=5, verbose=false, tol=1e-3)
end
bench("PFI (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:pfi, degree=5, verbose=false, tol=1e-3)
end
bench("VFI (degree=5, tol=1e-3)") do
    solve(spec_growth; method=:vfi, degree=5, verbose=false, tol=1e-3)
end
bench("VFI + Howard(10, tol=1e-3)") do
    solve(spec_growth; method=:vfi, degree=5, howard_steps=10, verbose=false, tol=1e-3)
end

if Threads.nthreads() > 1
    println()
    println("═══ Threading comparison (growth model) ═══")
    bench("PFI sequential") do
        solve(spec_growth; method=:pfi, degree=5, threaded=false, verbose=false, tol=1e-3)
    end
    bench("PFI threaded") do
        solve(spec_growth; method=:pfi, degree=5, threaded=true, verbose=false, tol=1e-3)
    end
    bench("VFI sequential") do
        solve(spec_growth; method=:vfi, degree=5, threaded=false, verbose=false, tol=1e-3)
    end
    bench("VFI threaded") do
        solve(spec_growth; method=:vfi, degree=5, threaded=true, verbose=false, tol=1e-3)
    end
end

println()
println("Done.")
```

**Step 2: Run to verify it works**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. benchmarks/bench_nonlinear.jl`
Expected: Timing output for each solver variant

**Step 3: Commit**

```bash
git add -f benchmarks/bench_nonlinear.jl
git commit -m "bench(dsge): add nonlinear solver benchmark script (#79)"
```

---

### Task 9: Full Test Suite Verification

**Files:** None (read-only verification)

**Step 1: Run the full DSGE test suite**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using Test
@testset "DSGE" begin
    include("test/dsge/test_dsge.jl")
end
' 2>&1 | tail -20`

Expected: All tests pass (existing + new VFI/Anderson/perf tests).

**Step 2: Run the full test suite (multiprocess)**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl 2>&1 | tail -20`

Expected: All ~12400+ tests pass, no regressions.

**Step 3: If any failures, fix and re-run**

If tests fail, debug and fix. Common issues:
- `y_dev_nodes` may need explicit `Vector` conversion for `basis_matrix \` on a view
- Anderson history vectors may be empty on first iteration — guard with `length >= 2`
- Threading with `Threads.@threads` requires all writes to be to distinct array indices

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(dsge): address test regressions from VFI/perf changes"
```

---

### Task 10: refs() Entry + Version Bump

**Files:**
- Modify: `src/dsge/vfi.jl` (ensure refs are in header comments)
- Check: refs system picks up VFI references

**Step 1: Verify refs() works for VFI**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/playful-chasing-dragon && julia --project=. -e '
using MacroEconometricModels
r = refs(:vfi_solver)
println(r)
'`

If refs() dispatches on function names, it should pick up the references from the file header comments. If not, check how refs() works and add the entry.

**Step 2: Commit if needed**

Only commit if refs() required changes.

---

## Summary of Commits

1. `feat(dsge): add Anderson acceleration utility for fixed-point iterations`
2. `feat(dsge): add VFI solver with Howard steps and Anderson acceleration (#78)`
3. `test(dsge): add VFI Howard steps, Anderson, threading, Smolyak tests`
4. `perf(dsge): add Anderson acceleration and threading to PFI solver (#79)`
5. `perf(dsge): add threading to collocation Jacobian computation (#79)`
6. `perf(dsge): pre-allocate loop buffers in PFI and collocation (#79)`
7. `perf(dsge): add @inbounds to verified-safe inner loops (#79)`
8. `bench(dsge): add nonlinear solver benchmark script (#79)`
9. (fix commit if needed)
10. (refs/version if needed)
