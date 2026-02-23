# Analytical Steady State + Moment Conditions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add analytical steady-state computation via `@dsge` macro `steady_state` block and analytical moment conditions (Lyapunov equation) for GMM estimation without simulation.

**Architecture:** Add `ss_fn` field to `DSGESpec` (with backward-compatible default `nothing`). Parse `steady_state:` block in `@dsge` macro into a function `θ -> Vector{T}`. New `src/dsge/analytical.jl` contains `solve_lyapunov` (discrete Lyapunov via Kronecker vectorization) and `analytical_moments` (matching `autocovariance_moments` output format). New `:analytical_gmm` method in `estimate_dsge` uses analytical moments instead of simulation.

**Tech Stack:** Julia 1.10+, LinearAlgebra (kron, vec), Optim.jl (existing), existing GMM infrastructure.

---

## Critical Context

### Adding `ss_fn` to DSGESpec — backward compatibility

`DSGESpec{T}` is constructed with a 9-argument inner constructor at **22 call sites** (7 in `src/`, 16 in `test/`). To avoid breaking all of them, the inner constructor must accept `ss_fn` as an **optional 10th argument** with default `nothing`.

**Affected call sites in `src/`:**
- `src/dsge/types.jl:64` — inner constructor definition
- `src/dsge/steady_state.jl:90` — `_update_steady_state`
- `src/dsge/estimation.jl:127,185,267,316,344` — 5 call sites in estimation
- `src/dsge/parser.jl:127` — `@dsge` macro output

**Affected call sites in `test/`:**
- `test/dsge/test_dsge.jl` — 16 direct `DSGESpec{Float64}(...)` calls (lines 41, 62, 69, 78, 94, 106, 125, 144, 163, 199, 224, 240, 256, 286, 312, 342)

### autocovariance_moments format

For k variables and L lags, `autocovariance_moments` returns a vector with:
1. Upper-triangle of variance-covariance: k*(k+1)/2 elements (in row-major order: `[var(1), cov(1,2), var(2), ...]`)
2. Diagonal autocovariances at each lag: k elements per lag

`analytical_moments` must produce the same vector format.

### Lyapunov equation

For solution `y_t = G1 * y_{t-1} + impact * ε_t` with `E[ε_t ε_t'] = I`:
- Unconditional covariance: `Σ_y = G1 * Σ_y * G1' + impact * impact'`
- Vectorized: `vec(Σ_y) = (I - kron(G1, G1))⁻¹ * vec(impact * impact')`
- Autocovariance at lag h: `Γ_h = G1^h * Σ_y`

---

### Task 1: Add `ss_fn` field to DSGESpec

**Files:**
- Modify: `src/dsge/types.jl:49-76` — add field + update inner constructor
- Modify: `src/dsge/steady_state.jl:36-95` — check `spec.ss_fn` + propagate in `_update_steady_state`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write failing tests**

Add at the end of the "Steady state" section in `test/dsge/test_dsge.jl` (after line 492):

```julia
@testset "Steady state: ss_fn field on DSGESpec" begin
    # DSGESpec with ss_fn=nothing (backward compat)
    spec = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[]
    )
    @test spec.ss_fn === nothing

    # DSGESpec with explicit ss_fn
    my_ss = (θ) -> [0.0]
    spec2 = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[], my_ss
    )
    @test spec2.ss_fn === my_ss
    @test spec2.ss_fn(spec2.param_values) == [0.0]
end

@testset "Steady state: auto-detect ss_fn on spec" begin
    spec = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[], (θ) -> [0.0]
    )
    spec2 = compute_steady_state(spec)
    @test spec2.steady_state[1] ≈ 0.0
    @test spec2.ss_fn !== nothing  # ss_fn propagated
end
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `ss_fn` field does not exist on DSGESpec

**Step 3: Implement — modify DSGESpec inner constructor**

In `src/dsge/types.jl`, add `ss_fn` field to the struct and update the inner constructor to accept it as optional 10th argument:

```julia
struct DSGESpec{T<:AbstractFloat}
    endog::Vector{Symbol}
    exog::Vector{Symbol}
    params::Vector{Symbol}
    param_values::Dict{Symbol,T}
    equations::Vector{Expr}
    residual_fns::Vector{Function}
    n_endog::Int
    n_exog::Int
    n_params::Int
    n_expect::Int
    forward_indices::Vector{Int}
    steady_state::Vector{T}
    varnames::Vector{String}
    ss_fn::Union{Nothing, Function}

    function DSGESpec{T}(endog, exog, params, param_values, equations, residual_fns,
                         n_expect, forward_indices, steady_state,
                         ss_fn::Union{Nothing, Function}=nothing) where {T<:AbstractFloat}
        n_endog = length(endog)
        n_exog = length(exog)
        n_params = length(params)
        @assert length(equations) == n_endog "Need as many equations as endogenous variables"
        @assert length(residual_fns) == n_endog
        @assert length(forward_indices) == n_expect
        varnames = [string(s) for s in endog]
        new{T}(endog, exog, params, param_values, equations, residual_fns,
               n_endog, n_exog, n_params, n_expect, forward_indices, steady_state, varnames, ss_fn)
    end
end
```

**Step 4: Implement — update `_update_steady_state` to propagate `ss_fn`**

In `src/dsge/steady_state.jl`, change `_update_steady_state` to pass through `spec.ss_fn`:

```julia
function _update_steady_state(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    DSGESpec{T}(
        spec.endog, spec.exog, spec.params, spec.param_values,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, y_ss, spec.ss_fn
    )
end
```

Also update `compute_steady_state` to auto-detect `spec.ss_fn`: after line 40 (`θ = spec.param_values`), add:

```julia
    # Auto-detect: if spec has an analytical ss_fn, use it
    if method == :auto && spec.ss_fn !== nothing
        y_ss = T.(spec.ss_fn(θ))
        @assert length(y_ss) == n "ss_fn must return vector of length $n"
        return _update_steady_state(spec, y_ss)
    end
```

**Step 5: Implement — update estimation.jl call sites to propagate `ss_fn`**

In `src/dsge/estimation.jl`, every `DSGESpec{T}(...)` constructor call that creates a `new_spec` or `final_spec` must pass `spec.ss_fn` as the 10th argument. There are 5 call sites at lines 127, 185, 267, 316, 344.

For each, add `spec.ss_fn` after the `T[]` (empty steady state) argument:

```julia
# Line 127 (irf_matching new_spec):
new_spec = DSGESpec{T}(
    spec.endog, spec.exog, spec.params, new_pv,
    spec.equations, spec.residual_fns,
    spec.n_expect, spec.forward_indices, T[], spec.ss_fn
)

# Same pattern for lines 185, 267, 316, 344
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: ALL tests pass (including 234+ existing + 5 new)

**Step 7: Commit**

```bash
git add src/dsge/types.jl src/dsge/steady_state.jl src/dsge/estimation.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add ss_fn field to DSGESpec for analytical steady state"
```

---

### Task 2: Parse `steady_state` block in `@dsge` macro

**Files:**
- Modify: `src/dsge/parser.jl:51-137` — detect and parse `steady_state:` block
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write failing tests**

Add after the existing parser tests (after "Parser: equation count mismatch" testset, around line 446):

```julia
@testset "Parser: steady_state block" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state:
            return [0.0]
    end
    @test spec isa DSGESpec{Float64}
    @test spec.ss_fn !== nothing
    @test spec.ss_fn(spec.param_values) == [0.0]
    @test spec.n_endog == 1
    @test spec.n_expect == 0
end

@testset "Parser: steady_state with param access" begin
    spec = @dsge begin
        parameters: α = 0.33, δ = 0.025
        endogenous: y, k
        exogenous: ε
        y[t] = k[t-1]^α + ε[t]
        k[t] = y[t] - δ * k[t-1]
        steady_state:
            k_ss = (1.0 / δ)^(1 / (1 - α))
            y_ss = k_ss^α
            return [y_ss, k_ss]
    end
    @test spec.ss_fn !== nothing
    ss = spec.ss_fn(spec.param_values)
    @test length(ss) == 2
    @test ss[2] ≈ (1.0 / 0.025)^(1 / (1 - 0.33)) atol=1e-6
end

@testset "Parser: steady_state auto-detected by compute_steady_state" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state:
            return [0.0]
    end
    spec2 = compute_steady_state(spec)
    @test spec2.steady_state[1] ≈ 0.0
end
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `steady_state:` treated as unrecognized statement or equation

**Step 3: Implement — modify `_dsge_impl` in parser.jl**

The key changes in `_dsge_impl` (starting at line 51):

1. Add a variable to hold the steady-state expression:

```julia
function _dsge_impl(block::Expr)
    params = Symbol[]
    param_defaults = Dict{Symbol,Any}()
    endog = Symbol[]
    exog = Symbol[]
    raw_equations = Expr[]
    ss_body = nothing  # NEW: steady_state block body
```

2. In the statement loop (line 60), add detection for `steady_state:`:

```julia
    for stmt in stmts
        label = _detect_declaration(stmt)
        if label === :parameters
            _extract_parameters!(stmt, params, param_defaults)
        elseif label === :endogenous
            append!(endog, _extract_names(stmt))
        elseif label === :exogenous
            append!(exog, _extract_names(stmt))
        elseif label === :steady_state
            ss_body = _extract_ss_body(stmt)
        elseif label === nothing
            if stmt isa Expr && stmt.head == :(=)
                push!(raw_equations, stmt)
            else
                error("@dsge: unrecognized statement: $stmt")
            end
        end
    end
```

3. Build the `ss_fn` expression and pass to constructor (modify the `result` quote block around line 126):

```julia
    # Build ss_fn expression if steady_state block was provided
    ss_fn_expr = if ss_body !== nothing
        # Build: (θ) -> begin <param unpacking>; <ss_body> end
        param_unpack = [:($(p) = _ss_θ_[$(QuoteNode(p))]) for p in params]
        body = Expr(:block, param_unpack..., ss_body)
        Expr(:->, :_ss_θ_, body)
    else
        :nothing
    end

    result = quote
        DSGESpec{Float64}(
            $endog_expr, $exog_expr, $params_expr,
            $param_vals_expr,
            $eq_vec_expr,
            $fn_vec_expr,
            $n_expect, $fwd_expr, Float64[], $ss_fn_expr
        )
    end

    return esc(result)
```

4. Add the `_extract_ss_body` helper function:

```julia
"""
    _extract_ss_body(stmt) → Expr

Extract the body from a `steady_state: ...` declaration.

The parser sees `steady_state:` as the start of a labeled section.
The body contains arbitrary Julia code ending with `return [...]`.
"""
function _extract_ss_body(stmt::Expr)
    # steady_state: <body> is parsed as (= (call : steady_state <first_expr>) <rest>)
    # or potentially other AST shapes depending on the body.
    # We need to extract the RHS as a block of Julia code.
    if stmt.head == :(=)
        rhs = stmt.args[2]
        # Unwrap block wrapper
        if rhs isa Expr && rhs.head == :block
            inner = filter(a -> !(a isa LineNumberNode), rhs.args)
            if length(inner) == 1
                return inner[1]
            else
                return Expr(:block, inner...)
            end
        end
        return rhs
    end
    error("@dsge: cannot parse steady_state block")
end
```

**Important AST note:** The `steady_state:` label followed by multiple lines creates a complex AST. The implementer must test the actual AST shape Julia produces and adjust `_extract_ss_body` and possibly `_detect_declaration` accordingly. The `_detect_declaration` function already handles `label: name = value` patterns — `steady_state:` followed by assignments will hit this path. The key insight is that `steady_state:` is detected as a label by `_detect_declaration`, and then `_extract_ss_body` must extract the multi-line body.

**Testing note:** The exact AST parsing may need iteration. If the first approach doesn't work with multi-line bodies, an alternative is to use `equations:` and `steady_state:` as section delimiters that partition the block's statements, rather than relying on Julia's AST to group them.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: ALL tests pass (existing + 8 new)

**Step 5: Commit**

```bash
git add src/dsge/parser.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): parse steady_state block in @dsge macro"
```

---

### Task 3: Implement `solve_lyapunov` and `analytical_moments`

**Files:**
- Create: `src/dsge/analytical.jl`
- Modify: `src/MacroEconometricModels.jl` — add include + exports
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write failing tests**

Add a new Section 13 at the end of `test/dsge/test_dsge.jl` (before the final `end`):

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 13: Analytical Moments
# ─────────────────────────────────────────────────────────────────────────────

@testset "solve_lyapunov: 1D AR(1)" begin
    # y_t = ρ * y_{t-1} + σ * ε_t → Σ = σ² / (1 - ρ²)
    ρ = 0.9
    σ = 1.0
    G1 = fill(ρ, 1, 1)
    impact = fill(σ, 1, 1)
    Σ = solve_lyapunov(G1, impact)
    @test size(Σ) == (1, 1)
    @test Σ[1, 1] ≈ σ^2 / (1 - ρ^2) atol=1e-10
end

@testset "solve_lyapunov: 2D VAR(1)" begin
    G1 = [0.8 0.1; 0.0 0.5]
    impact = [1.0 0.0; 0.0 1.0]
    Σ = solve_lyapunov(G1, impact)
    @test size(Σ) == (2, 2)
    @test issymmetric(Σ)  # covariance must be symmetric
    @test all(eigvals(Σ) .> 0)  # positive definite

    # Verify Σ = G1 * Σ * G1' + impact * impact'
    residual = Σ - G1 * Σ * G1' - impact * impact'
    @test norm(residual) < 1e-10
end

@testset "solve_lyapunov: unstable system errors" begin
    G1 = fill(1.5, 1, 1)  # explosive
    impact = fill(1.0, 1, 1)
    @test_throws ArgumentError solve_lyapunov(G1, impact)
end

@testset "analytical_moments: AR(1) matches simulation" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    m_analytical = analytical_moments(sol; lags=1)
    # For 1D: [var(y), autocov(y, 1)] = [σ²/(1-ρ²), ρ*σ²/(1-ρ²)]
    expected_var = 1.0 / (1 - 0.81)
    expected_autocov = 0.9 * expected_var
    @test length(m_analytical) == 2
    @test m_analytical[1] ≈ expected_var atol=1e-10
    @test m_analytical[2] ≈ expected_autocov atol=1e-10
end

@testset "analytical_moments: 2D matches simulation" begin
    spec = @dsge begin
        parameters: ρ = 0.8, σ = 1.0
        endogenous: y, x
        exogenous: ε_y, ε_x
        y[t] = ρ * y[t-1] + σ * ε_y[t]
        x[t] = 0.5 * y[t-1] + 0.5 * x[t-1] + ε_x[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    m_analytical = analytical_moments(sol; lags=2)
    # k=2, lags=2: k*(k+1)/2 + k*lags = 3 + 4 = 7 moments
    @test length(m_analytical) == 7

    # Cross-check with long simulation
    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, 100_000; rng=rng)
    m_simulated = autocovariance_moments(sim_data; lags=2)
    # Simulation moments should converge to analytical
    for i in eachindex(m_analytical)
        @test m_analytical[i] ≈ m_simulated[i] rtol=0.05
    end
end

@testset "analytical_moments: lags=0 returns only variances" begin
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    m = analytical_moments(sol; lags=0)
    @test length(m) == 1  # just var(y)
    @test m[1] ≈ 1.0 / (1 - 0.25) atol=1e-10
end
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `solve_lyapunov` and `analytical_moments` not defined

**Step 3: Create `src/dsge/analytical.jl`**

```julia
# MacroEconometricModels.jl — GPL-3.0-or-later
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>

"""
Analytical moment computation for linear DSGE models via discrete Lyapunov equation.

References:
- Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. Ch. 10.
"""

using LinearAlgebra

"""
    solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) -> Matrix{T}

Solve the discrete Lyapunov equation: `Σ = G1 * Σ * G1' + impact * impact'`.

Uses Kronecker vectorization: `vec(Σ) = (I - G1 ⊗ G1)⁻¹ * vec(impact * impact')`.

Returns the unconditional covariance matrix `Σ` (n × n, symmetric positive semi-definite).

Throws `ArgumentError` if G1 is not stable (max |eigenvalue| >= 1).
"""
function solve_lyapunov(G1::AbstractMatrix{T}, impact::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(G1, 1)
    @assert size(G1) == (n, n) "G1 must be square"
    @assert size(impact, 1) == n "impact must have n rows"

    # Check stability
    max_eig = maximum(abs.(eigvals(G1)))
    max_eig >= one(T) && throw(ArgumentError(
        "G1 is not stable (max |eigenvalue| = $(max_eig)). Lyapunov equation has no solution."))

    Q = impact * impact'
    # Vectorize: vec(Σ) = (I_n² - G1 ⊗ G1)⁻¹ * vec(Q)
    n2 = n * n
    A = Matrix{T}(I, n2, n2) - kron(G1, G1)
    sigma_vec = A \ vec(Q)
    Sigma = reshape(sigma_vec, n, n)

    # Enforce exact symmetry
    Sigma = (Sigma + Sigma') / 2
    return Sigma
end

# Float64 fallback
solve_lyapunov(G1::AbstractMatrix{<:Real}, impact::AbstractMatrix{<:Real}) =
    solve_lyapunov(Float64.(G1), Float64.(impact))

"""
    analytical_moments(sol::DSGESolution{T}; lags::Int=1) -> Vector{T}

Compute analytical moment vector from a solved DSGE model.

Uses the discrete Lyapunov equation to compute the unconditional covariance,
then extracts the same moment format as `autocovariance_moments`:

1. Upper-triangle of variance-covariance matrix: k*(k+1)/2 elements
2. Diagonal autocovariances at each lag: k elements per lag

# Arguments
- `sol` — solved DSGE model (must be stable/determined)
- `lags` — number of autocovariance lags (default: 1)
"""
function analytical_moments(sol::DSGESolution{T}; lags::Int=1) where {T<:AbstractFloat}
    k = nvars(sol)
    Sigma = solve_lyapunov(sol.G1, sol.impact)

    moments = T[]

    # Upper triangle of variance-covariance matrix (matching autocovariance_moments)
    for i in 1:k
        for j in i:k
            push!(moments, Sigma[i, j])
        end
    end

    # Autocovariances at each lag: Γ_h = G1^h * Σ, extract diagonal
    G1_power = copy(sol.G1)
    for lag in 1:lags
        Gamma_h = G1_power * Sigma
        for i in 1:k
            push!(moments, Gamma_h[i, i])
        end
        G1_power = G1_power * sol.G1
    end

    moments
end
```

**Step 4: Add include and exports in `src/MacroEconometricModels.jl`**

Add the include after the existing `include("dsge/simulation.jl")` line (around line 248):

```julia
include("dsge/analytical.jl")
```

Add exports near the existing DSGE exports (around line 330):

```julia
export solve_lyapunov, analytical_moments
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: ALL tests pass

**Step 6: Commit**

```bash
git add src/dsge/analytical.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add solve_lyapunov and analytical_moments"
```

---

### Task 4: Add `:analytical_gmm` method to `estimate_dsge`

**Files:**
- Modify: `src/dsge/estimation.jl` — add `:analytical_gmm` branch + helper
- Modify: `src/dsge/types.jl:279` — add `:analytical_gmm` to assertion
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write failing tests**

Add a new Section 14 at the end of `test/dsge/test_dsge.jl` (before the final `end`):

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 14: Analytical GMM Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "DSGE Analytical GMM Estimation" begin
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: rho = 0.7, sigma = 1.0
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
    end
    spec = compute_steady_state(spec)

    # Simulate data
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 500; rng=rng)

    # Estimate rho via analytical GMM
    bounds = ParameterTransform([-0.99], [0.99])
    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:analytical_gmm,
                        bounds=bounds)

    @test est isa DSGEEstimation{Float64}
    @test est.method == :analytical_gmm
    @test est.converged
    @test abs(est.theta[1] - 0.7) < 0.2  # reasonable recovery
    @test is_determined(est.solution)
end

@testset "DSGE Analytical GMM: lags kwarg" begin
    spec = @dsge begin
        parameters: rho = 0.8
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 300; rng=Random.MersenneTwister(99))

    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:analytical_gmm, lags=2)
    @test est isa DSGEEstimation{Float64}
    @test est.method == :analytical_gmm
end

@testset "DSGE Analytical GMM: invalid method includes analytical_gmm" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    Y = randn(100, 1)
    @test_throws ArgumentError estimate_dsge(spec, Y, [:rho]; method=:invalid)
end
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `:analytical_gmm` not recognized

**Step 3: Implement — update types.jl assertion**

In `src/dsge/types.jl` line 279, update:

```julia
@assert method ∈ (:irf_matching, :euler_gmm, :smm, :analytical_gmm)
```

**Step 4: Implement — add `lags` kwarg and `:analytical_gmm` branch in estimation.jl**

Add `lags::Int=1` to the `estimate_dsge` signature (line 51-61):

```julia
function estimate_dsge(spec::DSGESpec{T}, data::AbstractMatrix,
                        param_names::Vector{Symbol};
                        method::Symbol=:irf_matching,
                        target_irfs::Union{Nothing,ImpulseResponse}=nothing,
                        var_lags::Int=4, irf_horizon::Int=20,
                        weighting::Symbol=:two_step,
                        n_lags_instruments::Int=4,
                        sim_ratio::Int=5, burn::Int=100,
                        moments_fn::Function=d -> autocovariance_moments(d; lags=1),
                        bounds::Union{Nothing,ParameterTransform}=nothing,
                        lags::Int=1,
                        rng=Random.default_rng()) where {T<:AbstractFloat}
```

Add the new branch after the `:smm` branch (around line 78):

```julia
    elseif method == :analytical_gmm
        return _estimate_dsge_analytical_gmm(spec, data_T, param_names;
                                              lags=lags, weighting=weighting,
                                              moments_fn=d -> autocovariance_moments(d; lags=lags),
                                              bounds=bounds)
```

Update the error message:

```julia
    else
        throw(ArgumentError("method must be :irf_matching, :euler_gmm, :smm, or :analytical_gmm"))
    end
```

**Step 5: Implement — add `_estimate_dsge_analytical_gmm` function**

Add at the end of `src/dsge/estimation.jl`:

```julia
# =============================================================================
# Analytical GMM Estimation (Lyapunov equation moments)
# =============================================================================

"""
    _estimate_dsge_analytical_gmm(spec, data, param_names; ...) -> DSGEEstimation

Internal: Analytical GMM estimation using Lyapunov equation moments.

Matches model-implied analytical moments (from `analytical_moments`) to data moments
(from `autocovariance_moments`). No simulation required — uses the discrete Lyapunov
equation to compute unconditional covariances exactly.
"""
function _estimate_dsge_analytical_gmm(spec::DSGESpec{T}, data::Matrix{T},
                                         param_names::Vector{Symbol};
                                         lags=1, weighting=:two_step,
                                         moments_fn=d -> autocovariance_moments(d; lags=1),
                                         bounds=nothing) where {T}
    theta0 = T[spec.param_values[p] for p in param_names]

    # Data moments
    m_data = moments_fn(data)
    n_moments = length(m_data)

    # GMM moment function: returns T_obs × n_moments matrix
    # For analytical GMM, each "observation" contributes the same moment discrepancy
    # (the analytical moments are non-stochastic given θ).
    # We construct per-observation moment contributions as deviations from model moments.
    T_obs = size(data, 1)

    function analytical_moment_fn(theta, _data)
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[], spec.ss_fn
        )
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol) || !is_stable(sol)
                # Return large discrepancy for infeasible parameters
                return fill(T(1e6), 1, n_moments)
            end
            m_model = analytical_moments(sol; lags=lags)
            # GMM expects T_obs × n_moments; replicate the discrepancy
            g = (m_data .- m_model)'  # 1 × n_moments
            return g
        catch
            return fill(T(1e6), 1, n_moments)
        end
    end

    gmm_result = estimate_gmm(analytical_moment_fn, theta0, data;
                                weighting=weighting, bounds=bounds)

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = gmm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :analytical_gmm, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Test; include("test/dsge/test_dsge.jl")'`
Expected: ALL tests pass

**Step 7: Commit**

```bash
git add src/dsge/estimation.jl src/dsge/types.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add :analytical_gmm estimation method"
```

---

### Task 5: Full integration + full test suite

**Files:**
- Modify: `src/summary_refs.jl` — add Hamilton 1994 reference for Lyapunov
- Test: Full test suite run

**Step 1: Add reference**

Add to `_REFERENCES` in `src/summary_refs.jl`:

```julia
:hamilton1994 => (
    authors = "Hamilton, J. D.",
    year = 1994,
    title = "Time Series Analysis",
    journal = "Princeton University Press",
    bibtex = """@book{hamilton1994,
      author = {Hamilton, James D.},
      title = {Time Series Analysis},
      year = {1994},
      publisher = {Princeton University Press}}"""
),
```

Add to `_TYPE_REFS`:

```julia
:analytical_gmm => [:hamilton1994, :hansen1982],
```

**Step 2: Run the full test suite**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: ~7660+ pass, 2 broken (pre-existing)

**Step 3: Commit**

```bash
git add src/summary_refs.jl
git commit -m "feat(dsge): add analytical GMM references"
```

---

## Summary

| Task | Description | New tests |
|------|-------------|-----------|
| 1 | `ss_fn` field on DSGESpec + steady_state integration | ~5 |
| 2 | Parse `steady_state:` block in `@dsge` macro | ~8 |
| 3 | `solve_lyapunov` + `analytical_moments` | ~12 |
| 4 | `:analytical_gmm` method in `estimate_dsge` | ~5 |
| 5 | References + full test suite verification | 0 |

Total: ~30 new tests across 5 tasks.
