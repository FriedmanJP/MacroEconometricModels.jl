# DSGE Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add linear DSGE model support — `@dsge` macro, gensys/BK solvers, perfect foresight, GMM estimation, and simulation with IRF/FEVD bridging to existing types.

**Architecture:** 9 source files in `src/dsge/`, 4 test files in `test/dsge/`. Types first, then parser, then numerical pipeline (SS → linearize → solve), then simulation/estimation. IRF/FEVD reuse existing `ImpulseResponse{T}` and `FEVD{T}` types so `plot_result()` works automatically.

**Tech Stack:** Julia metaprogramming (macros, Expr walking), `LinearAlgebra` (QZ/Schur), `Optim` (steady state), `SparseArrays` (perfect foresight), existing `estimate_gmm`, `ImpulseResponse{T}`, `FEVD{T}`.

**Design doc:** `docs/plans/2026-02-24-dsge-module-design.md`

---

### Task 1: Types and Abstract Hierarchy

**Files:**
- Create: `src/dsge/types.jl`
- Modify: `src/core/types.jl` (add `AbstractDSGEModel`)
- Test: `test/dsge/test_dsge.jl`

**Step 1: Add abstract type to core/types.jl**

In `src/core/types.jl`, after the `AbstractNowcastModel` line (line 94), add:

```julia
"""Abstract supertype for DSGE models."""
abstract type AbstractDSGEModel <: StatsAPI.StatisticalModel end
```

**Step 2: Create src/dsge/types.jl**

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Type definitions for DSGE models — specification, linearized form, solution, and estimation.
"""

using LinearAlgebra

# =============================================================================
# DSGESpec — parsed model specification from @dsge macro
# =============================================================================

"""
    DSGESpec{T}

Parsed DSGE model specification. Created by the `@dsge` macro.

Fields:
- `endog::Vector{Symbol}` — endogenous variable names
- `exog::Vector{Symbol}` — exogenous shock names
- `params::Vector{Symbol}` — parameter names
- `param_values::Dict{Symbol,T}` — calibrated parameter values
- `equations::Vector{Expr}` — raw Julia equation expressions
- `residual_fns::Vector{Function}` — callable `f(y_t, y_lag, y_lead, ε, θ) → scalar`
- `n_endog::Int` — number of endogenous variables
- `n_exog::Int` — number of exogenous shocks
- `n_params::Int` — number of parameters
- `n_expect::Int` — number of expectation errors (forward-looking variables)
- `forward_indices::Vector{Int}` — indices of equations with `[t+1]` terms
- `steady_state::Vector{T}` — steady state values
- `varnames::Vector{String}` — display names
"""
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

    function DSGESpec{T}(endog, exog, params, param_values, equations, residual_fns,
                         n_expect, forward_indices, steady_state) where {T<:AbstractFloat}
        n_endog = length(endog)
        n_exog = length(exog)
        n_params = length(params)
        @assert length(equations) == n_endog "Need as many equations as endogenous variables"
        @assert length(residual_fns) == n_endog
        @assert length(forward_indices) == n_expect
        varnames = [string(s) for s in endog]
        new{T}(endog, exog, params, param_values, equations, residual_fns,
               n_endog, n_exog, n_params, n_expect, forward_indices, steady_state, varnames)
    end
end

function Base.show(io::IO, spec::DSGESpec{T}) where {T}
    spec_data = Any[
        "Endogenous"    spec.n_endog;
        "Exogenous"     spec.n_exog;
        "Parameters"    spec.n_params;
        "Equations"     length(spec.equations);
        "Forward-looking" spec.n_expect;
        "Steady state"  isempty(spec.steady_state) ? "Not computed" : "Computed"
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Model Specification",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# LinearDSGE — canonical form Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
# =============================================================================

"""
    LinearDSGE{T}

Linearized DSGE in Sims canonical form: `Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t`.

Fields:
- `Gamma0::Matrix{T}` — n × n coefficient on y_t
- `Gamma1::Matrix{T}` — n × n coefficient on y_{t-1}
- `C::Vector{T}` — n × 1 constants
- `Psi::Matrix{T}` — n × n_shocks shock loading
- `Pi::Matrix{T}` — n × n_expect expectation error selection
- `spec::DSGESpec{T}` — back-reference to specification
"""
struct LinearDSGE{T<:AbstractFloat}
    Gamma0::Matrix{T}
    Gamma1::Matrix{T}
    C::Vector{T}
    Psi::Matrix{T}
    Pi::Matrix{T}
    spec::DSGESpec{T}

    function LinearDSGE{T}(Gamma0, Gamma1, C, Psi, Pi, spec) where {T<:AbstractFloat}
        n = spec.n_endog
        @assert size(Gamma0) == (n, n) "Gamma0 must be n×n"
        @assert size(Gamma1) == (n, n) "Gamma1 must be n×n"
        @assert length(C) == n "C must be length n"
        @assert size(Psi, 1) == n "Psi must have n rows"
        @assert size(Pi, 1) == n "Pi must have n rows"
        new{T}(Gamma0, Gamma1, C, Psi, Pi, spec)
    end
end

function Base.show(io::IO, ld::LinearDSGE{T}) where {T}
    n = ld.spec.n_endog
    spec_data = Any[
        "State dimension"   n;
        "Shocks"            size(ld.Psi, 2);
        "Expectation errors" size(ld.Pi, 2);
        "rank(Γ₀)"         rank(ld.Gamma0);
    ]
    _pretty_table(io, spec_data;
        title = "Linearized DSGE — Canonical Form",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# DSGESolution — RE solution y_t = G1·y_{t-1} + impact·ε_t + C_sol
# =============================================================================

"""
    DSGESolution{T}

Rational expectations solution: `y_t = G1·y_{t-1} + impact·ε_t + C_sol`.

Fields:
- `G1::Matrix{T}` — n × n state transition matrix
- `impact::Matrix{T}` — n × n_shocks impact matrix
- `C_sol::Vector{T}` — n × 1 constants
- `eu::Vector{Int}` — [existence, uniqueness]: 1=yes, 0=no, -1=indeterminate
- `method::Symbol` — `:gensys` or `:blanchard_kahn`
- `eigenvalues::Vector{ComplexF64}` — generalized eigenvalues from QZ
- `spec::DSGESpec{T}` — model specification
- `linear::LinearDSGE{T}` — linearized form
"""
struct DSGESolution{T<:AbstractFloat}
    G1::Matrix{T}
    impact::Matrix{T}
    C_sol::Vector{T}
    eu::Vector{Int}
    method::Symbol
    eigenvalues::Vector{ComplexF64}
    spec::DSGESpec{T}
    linear::LinearDSGE{T}
end

# Accessors
nvars(sol::DSGESolution) = sol.spec.n_endog
nshocks(sol::DSGESolution) = sol.spec.n_exog
is_determined(sol::DSGESolution) = sol.eu[1] == 1 && sol.eu[2] == 1
is_stable(sol::DSGESolution) = maximum(abs.(eigvals(sol.G1))) < 1.0

function Base.show(io::IO, sol::DSGESolution{T}) where {T}
    n = nvars(sol)
    n_stable = count(x -> abs(x) < 1.0, sol.eigenvalues)
    n_unstable = length(sol.eigenvalues) - n_stable
    exist_str = sol.eu[1] == 1 ? "Yes" : "No"
    unique_str = sol.eu[2] == 1 ? "Yes" : "No"
    max_eig = maximum(abs.(eigvals(sol.G1)))

    spec_data = Any[
        "Variables"        n;
        "Shocks"           nshocks(sol);
        "Method"           string(sol.method);
        "Existence"        exist_str;
        "Uniqueness"       unique_str;
        "Stable eigenvalues"   n_stable;
        "Unstable eigenvalues" n_unstable;
        "Max |eigenvalue(G1)|" _fmt(max_eig);
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# PerfectForesightPath
# =============================================================================

"""
    PerfectForesightPath{T}

Deterministic perfect foresight path.

Fields:
- `path::Matrix{T}` — T_periods × n_endog level values
- `deviations::Matrix{T}` — T_periods × n_endog deviations from SS
- `converged::Bool` — Newton convergence flag
- `iterations::Int` — Newton iterations used
- `spec::DSGESpec{T}` — model specification
"""
struct PerfectForesightPath{T<:AbstractFloat}
    path::Matrix{T}
    deviations::Matrix{T}
    converged::Bool
    iterations::Int
    spec::DSGESpec{T}
end

function Base.show(io::IO, pf::PerfectForesightPath{T}) where {T}
    spec_data = Any[
        "Variables"   pf.spec.n_endog;
        "Periods"     size(pf.path, 1);
        "Converged"   pf.converged ? "Yes" : "No";
        "Iterations"  pf.iterations;
    ]
    _pretty_table(io, spec_data;
        title = "Perfect Foresight Path",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# DSGEEstimation — GMM estimation result
# =============================================================================

"""
    DSGEEstimation{T} <: AbstractDSGEModel

DSGE model estimated via GMM (IRF matching or Euler equation moments).

Fields:
- `theta::Vector{T}` — estimated deep parameters
- `vcov::Matrix{T}` — asymptotic covariance matrix
- `param_names::Vector{Symbol}` — names of estimated parameters
- `method::Symbol` — `:irf_matching` or `:euler_gmm`
- `J_stat::T` — Hansen J-test statistic
- `J_pvalue::T` — J-test p-value
- `solution::DSGESolution{T}` — solution at estimated parameters
- `converged::Bool` — optimization convergence
- `spec::DSGESpec{T}` — model specification
"""
struct DSGEEstimation{T<:AbstractFloat} <: AbstractDSGEModel
    theta::Vector{T}
    vcov::Matrix{T}
    param_names::Vector{Symbol}
    method::Symbol
    J_stat::T
    J_pvalue::T
    solution::DSGESolution{T}
    converged::Bool
    spec::DSGESpec{T}

    function DSGEEstimation{T}(theta, vcov, param_names, method, J_stat, J_pvalue,
                                solution, converged, spec) where {T<:AbstractFloat}
        @assert length(theta) == length(param_names)
        @assert size(vcov) == (length(theta), length(theta))
        @assert method ∈ (:irf_matching, :euler_gmm)
        new{T}(theta, vcov, param_names, method, J_stat, J_pvalue, solution, converged, spec)
    end
end

# StatsAPI interface
StatsAPI.coef(m::DSGEEstimation) = m.theta
StatsAPI.vcov(m::DSGEEstimation) = m.vcov
StatsAPI.dof(m::DSGEEstimation) = length(m.theta)
StatsAPI.islinear(::DSGEEstimation) = false
StatsAPI.stderror(m::DSGEEstimation) = sqrt.(max.(diag(m.vcov), zero(eltype(m.theta))))

function Base.show(io::IO, est::DSGEEstimation{T}) where {T}
    spec_data = Any[
        "Parameters"    length(est.theta);
        "Method"        string(est.method);
        "J-statistic"   _fmt(est.J_stat);
        "J p-value"     _format_pvalue(est.J_pvalue);
        "Converged"     est.converged ? "Yes" : "No";
        "Determined"    is_determined(est.solution) ? "Yes" : "No";
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Estimation — GMM",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # Coefficient table
    se = stderror(est)
    pnames = [string(s) for s in est.param_names]
    _coef_table(io, "Estimated Parameters", pnames, est.theta, se; dist=:z)
    _sig_legend(io)
end

function report(est::DSGEEstimation{T}) where {T}
    show(stdout, est)
    println(stdout)
    show(stdout, est.solution)
end
```

**Step 3: Write the failing test**

Create `test/dsge/test_dsge.jl`:

```julia
# MacroEconometricModels.jl — DSGE Module Tests
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
# (GPL-3.0 license header)

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "DSGE Module" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Types
# ─────────────────────────────────────────────────────────────────────────────

@testset "Type hierarchy" begin
    @test AbstractDSGEModel <: StatsAPI.StatisticalModel
end

@testset "DSGESpec construction" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α, :β],
        Dict(:α => 0.33, :β => 0.99),
        [:(C[t] + K[t])], [identity],
        0, Int[], Float64[]
    )
    @test spec.n_endog == 2
    @test spec.n_exog == 1
    @test spec.n_params == 2
    @test spec.n_expect == 0
    @test length(spec.varnames) == 2
end

@testset "DSGESpec show" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α, :β],
        Dict(:α => 0.33, :β => 0.99),
        [:(C[t] + K[t])], [identity],
        0, Int[], Float64[]
    )
    io = IOBuffer()
    show(io, spec)
    s = String(take!(io))
    @test occursin("DSGE Model Specification", s)
    @test occursin("Endogenous", s)
end

@testset "LinearDSGE construction" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    @test size(ld.Gamma0) == (2, 2)
    @test size(ld.Psi) == (2, 1)
end

@testset "DSGESolution construction and accessors" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    @test nvars(sol) == 2
    @test nshocks(sol) == 1
    @test is_determined(sol)
    @test is_stable(sol)
end

@testset "DSGESolution show" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    io = IOBuffer()
    show(io, sol)
    s = String(take!(io))
    @test occursin("DSGE Solution", s)
    @test occursin("Existence", s)
end

end # top-level @testset
```

**Step 4: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/fixtures.jl"); include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `DSGESpec` not defined

**Step 5: Wire up includes and exports in MacroEconometricModels.jl**

In `src/MacroEconometricModels.jl`, add the DSGE includes after nowcast/forecast.jl (line 157) and before GMM (line 160):

```julia
# DSGE models
include("dsge/types.jl")
```

Add exports after the Nowcast section (around line 293):

```julia
# =============================================================================
# Exports - DSGE Models
# =============================================================================

# Abstract type
export AbstractDSGEModel

# Types
export DSGESpec, LinearDSGE, DSGESolution, PerfectForesightPath, DSGEEstimation

# Accessors
export nshocks, is_determined, is_stable
```

**Step 6: Run test to verify it passes**

Run: `julia --project=. -e 'using MacroEconometricModels; include("test/fixtures.jl"); include("test/dsge/test_dsge.jl")'`
Expected: PASS

**Step 7: Commit**

```bash
git add src/core/types.jl src/dsge/types.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add type hierarchy — DSGESpec, LinearDSGE, DSGESolution, DSGEEstimation"
```

---

### Task 2: Parser — `@dsge` Macro

**Files:**
- Create: `src/dsge/parser.jl`
- Modify: `src/MacroEconometricModels.jl` (add include + export)
- Test: `test/dsge/test_dsge.jl` (append parser tests)

**Step 1: Write the failing test**

Append to `test/dsge/test_dsge.jl` inside the top-level `@testset`:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Parser
# ─────────────────────────────────────────────────────────────────────────────

@testset "Parser: @dsge macro basic" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    @test spec isa DSGESpec{Float64}
    @test spec.endog == [:y]
    @test spec.exog == [:ε]
    @test spec.params == [:ρ, :σ]
    @test spec.param_values[:ρ] ≈ 0.9
    @test spec.n_endog == 1
    @test spec.n_exog == 1
    @test spec.n_expect == 0
end

@testset "Parser: multi-variable model" begin
    spec = @dsge begin
        parameters: α = 0.33, β = 0.99, δ = 0.025, ρ = 0.9
        endogenous: C, K, A
        exogenous: ε_A
        C[t] + K[t] = (1-δ)*K[t-1] + K[t-1]^α
        C[t] = β * E[t](C[t+1]) * (α * K[t]^(α-1) + 1-δ)
        A[t] = ρ * A[t-1] + ε_A[t]
    end
    @test spec.n_endog == 3
    @test spec.n_exog == 1
    @test spec.n_expect >= 1  # C[t+1] is forward-looking
    @test 1 ∈ spec.forward_indices || 2 ∈ spec.forward_indices
end

@testset "Parser: E[t] operator detection" begin
    spec = @dsge begin
        parameters: β = 0.99
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    @test spec.n_expect == 1
end

@testset "Parser: implicit forward-looking" begin
    # x[t+1] without E[t]() should still be treated as forward-looking
    spec = @dsge begin
        parameters: β = 0.99
        endogenous: x
        exogenous: ε
        x[t] = β * x[t+1] + ε[t]
    end
    @test spec.n_expect == 1
end

@testset "Parser: residual functions evaluate correctly" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    # residual_fns[1] should be: y_t - ρ*y_{t-1} - σ*ε_t
    # f(y_t, y_lag, y_lead, ε, θ) where θ = param_values
    fn = spec.residual_fns[1]
    # At steady state y=0, shock=0: residual = 0
    @test fn([0.0], [0.0], [0.0], [0.0], spec.param_values) ≈ 0.0 atol=1e-12
    # y_t=1, y_lag=0, ε=0: residual = 1 - 0.9*0 - 1.0*0 = 1
    @test fn([1.0], [0.0], [0.0], [0.0], spec.param_values) ≈ 1.0 atol=1e-12
    # y_t=0.9, y_lag=1.0, ε=0: residual = 0.9 - 0.9*1.0 = 0
    @test fn([0.9], [1.0], [0.0], [0.0], spec.param_values) ≈ 0.0 atol=1e-12
end

@testset "Parser: error on mismatch" begin
    @test_throws ArgumentError (@dsge begin
        parameters: ρ = 0.9
        endogenous: x, y
        exogenous: ε
        # Only 1 equation for 2 endogenous variables
        x[t] = ρ * x[t-1] + ε[t]
    end)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using MacroEconometricModels; include("test/fixtures.jl"); include("test/dsge/test_dsge.jl")'`
Expected: FAIL — `@dsge` not defined

**Step 3: Create src/dsge/parser.jl**

This is the most complex file. The macro needs to:

1. Parse `parameters:`, `endogenous:`, `exogenous:` declarations
2. Walk equation expressions finding `var[t]`, `var[t-1]`, `var[t+1]`, `E[t](...)`
3. Transform each equation LHS = RHS into residual function LHS - RHS
4. Build callable residual functions using `@eval`-free closures
5. Emit a `DSGESpec{Float64}(...)` constructor

Key internal functions:
- `_parse_dsge_block(block::Expr)` — main entry, returns named tuple of (params, endog, exog, equations)
- `_parse_declaration(line::Expr, keyword::Symbol)` — extract `keyword: a, b, c` or `keyword: a=1, b=2`
- `_extract_time_refs(eq::Expr, endog::Vector{Symbol})` — find all `var[t±k]` → `Dict{Symbol, Set{Int}}`
- `_has_forward_looking(eq::Expr, endog)` — does equation contain `[t+1]` terms?
- `_strip_expectation_operator(eq::Expr)` — replace `E[t](expr)` with just `expr`
- `_equation_to_residual(eq::Expr)` — transform `LHS = RHS` into `LHS - (RHS)`
- `_build_residual_fn(eq::Expr, endog, exog, params)` — build `(y_t, y_lag, y_lead, ε, θ) → scalar`
- `_substitute_time_vars(eq::Expr, endog, exog, ...)` — replace `C[t]` with `y_t[i]`, `C[t-1]` with `y_lag[i]`, etc.

The macro should be hygienic and produce a `DSGESpec{Float64}(...)` at call site.

```julia
# (GPL license header)

"""
@dsge macro and parser for DSGE model specification.

Parses a begin...end block with parameter, variable, and equation declarations
into a DSGESpec for downstream steady-state computation, linearization, and solving.
"""

# =============================================================================
# AST Walking Utilities
# =============================================================================

"""Check if an Expr is a `parameters:`, `endogenous:`, or `exogenous:` declaration."""
function _is_declaration(ex::Expr, keyword::Symbol)
    # Matches patterns like :(parameters: α = 0.33, β = 0.99)
    # which Julia parses as various nested forms
    ex.head == :call && ex.args[1] == keyword && return true
    # Handle the : operator form: :(keyword : stuff)
    if ex.head == :call && ex.args[1] == :(:) && length(ex.args) >= 2
        return ex.args[2] == keyword
    end
    false
end
_is_declaration(::Any, ::Symbol) = false

"""
    _parse_parameters(args) → (names::Vector{Symbol}, defaults::Dict{Symbol,Float64})

Parse parameter declarations like `α = 0.33, β = 0.99`.
"""
function _parse_parameters(args)
    names = Symbol[]
    defaults = Dict{Symbol,Float64}()
    for arg in args
        if arg isa Expr && arg.head == :(=) || (arg isa Expr && arg.head == :kw)
            name = arg.args[1]
            val = arg.args[2]
            push!(names, name)
            defaults[name] = Float64(eval(val))
        elseif arg isa Symbol
            push!(names, arg)
        end
    end
    names, defaults
end

"""
    _parse_varlist(args) → Vector{Symbol}

Parse variable declarations like `C, K, Y` or `ε_A`.
"""
function _parse_varlist(args)
    syms = Symbol[]
    for arg in args
        if arg isa Symbol
            push!(syms, arg)
        elseif arg isa Expr && arg.head == :tuple
            append!(syms, [a for a in arg.args if a isa Symbol])
        end
    end
    syms
end

"""
    _extract_time_refs(eq::Expr, endog::Vector{Symbol}, exog::Vector{Symbol})

Walk AST finding `var[t]`, `var[t-1]`, `var[t+1]` references.
Returns Dict{Symbol, Set{Int}} mapping variable name to set of time offsets.
"""
function _extract_time_refs(eq, endog::Vector{Symbol}, exog::Vector{Symbol})
    refs = Dict{Symbol, Set{Int}}()
    all_vars = vcat(endog, exog)
    _walk_time_refs!(refs, eq, all_vars)
    refs
end

function _walk_time_refs!(refs, ex::Expr, all_vars)
    # Match var[t], var[t-1], var[t+1]
    if ex.head == :ref && length(ex.args) >= 2 && ex.args[1] isa Symbol
        varname = ex.args[1]
        if varname ∈ all_vars
            offset = _parse_time_index(ex.args[2])
            if offset !== nothing
                if !haskey(refs, varname)
                    refs[varname] = Set{Int}()
                end
                push!(refs[varname], offset)
            end
        end
        # Also check E[t] pattern — E is not a variable
        if varname == :E
            # Walk into E[t](expr) — the call part
            # Already handled by recursive walk
        end
    end
    # Recurse into children
    for arg in ex.args
        _walk_time_refs!(refs, arg, all_vars)
    end
end
_walk_time_refs!(refs, ::Any, all_vars) = nothing

"""Parse time index expression: `t` → 0, `t-1` → -1, `t+1` → 1."""
function _parse_time_index(ex)
    ex == :t && return 0
    if ex isa Expr && ex.head == :call
        if ex.args[1] == :(-) && length(ex.args) == 3 && ex.args[2] == :t
            return -Int(ex.args[3])
        elseif ex.args[1] == :(+) && length(ex.args) == 3 && ex.args[2] == :t
            return Int(ex.args[3])
        end
    end
    nothing
end

"""Check if equation has forward-looking terms (t+1)."""
function _has_forward_looking(eq, endog, exog)
    refs = _extract_time_refs(eq, endog, exog)
    for (var, offsets) in refs
        if var ∈ endog && any(o > 0 for o in offsets)
            return true
        end
    end
    false
end

"""Transform `LHS = RHS` into `(LHS) - (RHS)` residual expression."""
function _equation_to_residual(eq::Expr)
    if eq.head == :(=) && length(eq.args) == 2
        return :( $(eq.args[1]) - $(eq.args[2]) )
    end
    error("Equation must be in LHS = RHS form, got: $eq")
end

"""
Replace `E[t](expr)` with just `expr` (under RE linearization they're equivalent).
"""
function _strip_expectation_operator(ex::Expr)
    # E[t](expr) parses as: call(ref(E, t), expr)
    if ex.head == :call && length(ex.args) >= 2
        callee = ex.args[1]
        if callee isa Expr && callee.head == :ref && callee.args[1] == :E
            # This is E[t](expr) — return the inner expression
            return length(ex.args) == 2 ? _strip_expectation_operator(ex.args[2]) :
                   Expr(:tuple, [_strip_expectation_operator(a) for a in ex.args[2:end]]...)
        end
    end
    # Recurse into children
    new_args = Any[]
    for arg in ex.args
        if arg isa Expr
            push!(new_args, _strip_expectation_operator(arg))
        else
            push!(new_args, arg)
        end
    end
    Expr(ex.head, new_args...)
end
_strip_expectation_operator(x) = x

"""
Build a residual function from an equation expression.

Returns a closure: `(y_t::Vector, y_lag::Vector, y_lead::Vector, ε::Vector, θ::Dict) → Float64`

The function substitutes:
- `var[t]`   → `y_t[index]`
- `var[t-1]` → `y_lag[index]`
- `var[t+1]` → `y_lead[index]`
- `shock[t]` → `ε[index]`
- parameter names → `θ[name]`
"""
function _build_residual_fn(resid_expr::Expr, endog::Vector{Symbol},
                             exog::Vector{Symbol}, params::Vector{Symbol})
    # Substitute time-indexed variables with array accesses
    substituted = _substitute_vars(resid_expr, endog, exog, params)

    # Build the function
    # Use eval to create the closure (macro context handles hygiene)
    fn_expr = quote
        function(y_t::AbstractVector, y_lag::AbstractVector,
                 y_lead::AbstractVector, ε::AbstractVector,
                 θ::Dict{Symbol})
            $substituted
        end
    end
    eval(fn_expr)
end

"""
Recursively substitute `var[t]` → `y_t[i]`, `var[t-1]` → `y_lag[i]`, etc.
Also substitute parameter names → `θ[:name]`.
"""
function _substitute_vars(ex::Expr, endog, exog, params)
    # Handle var[t±k] references
    if ex.head == :ref && length(ex.args) >= 2 && ex.args[1] isa Symbol
        varname = ex.args[1]
        offset = _parse_time_index(ex.args[2])

        if varname ∈ endog && offset !== nothing
            idx = findfirst(==(varname), endog)
            if offset == 0
                return :(y_t[$idx])
            elseif offset < 0
                return :(y_lag[$idx])
            else
                return :(y_lead[$idx])
            end
        elseif varname ∈ exog && offset !== nothing
            idx = findfirst(==(varname), exog)
            return :(ε[$idx])
        end
        # E[t] — leave as-is (handled by _strip_expectation_operator earlier)
    end

    # Handle E[t](expr) calls — strip them
    if ex.head == :call && length(ex.args) >= 2
        callee = ex.args[1]
        if callee isa Expr && callee.head == :ref && callee.args[1] == :E
            # E[t](expr) → just process the inner expression
            inner = length(ex.args) == 2 ? ex.args[2] : Expr(:tuple, ex.args[2:end]...)
            return _substitute_vars(inner, endog, exog, params)
        end
    end

    # Substitute bare parameter symbols
    new_args = Any[]
    for arg in ex.args
        if arg isa Symbol && arg ∈ params
            push!(new_args, :(θ[$(QuoteNode(arg))]))
        elseif arg isa Expr
            push!(new_args, _substitute_vars(arg, endog, exog, params))
        else
            push!(new_args, arg)
        end
    end
    Expr(ex.head, new_args...)
end
_substitute_vars(x::Symbol, endog, exog, params) =
    x ∈ params ? :(θ[$(QuoteNode(x))]) : x
_substitute_vars(x, endog, exog, params) = x

# =============================================================================
# @dsge Macro
# =============================================================================

"""
    @dsge begin ... end

Specify a DSGE model using natural Julia syntax.

# Syntax

```julia
spec = @dsge begin
    parameters: α = 0.33, β = 0.99, δ = 0.025, ρ = 0.9, σ_A = 0.01

    endogenous: C, K, Y, A
    exogenous: ε_A

    1/C[t] = β * E[t](1/C[t+1]) * (α * K[t]^(α-1) + 1-δ)
    C[t] + K[t] = K[t-1]^α + (1-δ)*K[t-1]
    Y[t] = exp(A[t]) * K[t-1]^α
    A[t] = ρ * A[t-1] + σ_A * ε_A[t]
end
```

The `E[t](·)` expectation operator is optional — any `var[t+1]` is treated as
forward-looking regardless.

Returns a `DSGESpec{Float64}` ready for `compute_steady_state`, `linearize`, and `solve`.
"""
macro dsge(block)
    _dsge_impl(block)
end

function _dsge_impl(block::Expr)
    @assert block.head == :block "Expected begin...end block"

    # Filter out LineNumberNode
    lines = filter(x -> !(x isa LineNumberNode), block.args)

    # Phase 1: Extract declarations
    param_names = Symbol[]
    param_defaults = Dict{Symbol,Float64}()
    endog = Symbol[]
    exog = Symbol[]
    eq_lines = Expr[]

    for line in lines
        if !(line isa Expr)
            continue
        end

        # Try to match `keyword: args...` patterns
        parsed = _try_parse_declaration(line)
        if parsed !== nothing
            keyword, args = parsed
            if keyword == :parameters
                pn, pd = _parse_parameters(args)
                append!(param_names, pn)
                merge!(param_defaults, pd)
            elseif keyword == :endogenous
                append!(endog, _parse_varlist(args))
            elseif keyword == :exogenous
                append!(exog, _parse_varlist(args))
            end
        elseif line.head == :(=)
            push!(eq_lines, line)
        end
    end

    # Validate
    n_eq = length(eq_lines)
    n_endog = length(endog)
    if n_eq != n_endog
        return :(throw(ArgumentError(
            "Number of equations ($($n_eq)) must equal number of endogenous variables ($($n_endog))"
        )))
    end

    # Phase 2: Process equations
    # Detect forward-looking indices and build residual expressions
    forward_indices = Int[]
    residual_exprs = Expr[]

    for (i, eq) in enumerate(eq_lines)
        stripped = _strip_expectation_operator(eq)
        if _has_forward_looking(stripped, endog, exog)
            push!(forward_indices, i)
        end
        resid = _equation_to_residual(stripped)
        push!(residual_exprs, resid)
    end

    n_expect = length(forward_indices)

    # Phase 3: Build residual functions
    # We generate the code that builds them at runtime
    fn_exprs = []
    for resid in residual_exprs
        substituted = _substitute_vars(resid, endog, exog, param_names)
        fn_ex = quote
            function(y_t::AbstractVector, y_lag::AbstractVector,
                     y_lead::AbstractVector, ε::AbstractVector,
                     θ::Dict{Symbol})
                $substituted
            end
        end
        push!(fn_exprs, fn_ex)
    end

    # Phase 4: Emit DSGESpec constructor
    quote
        DSGESpec{Float64}(
            $(QuoteNode(endog)),
            $(QuoteNode(exog)),
            $(QuoteNode(param_names)),
            Dict{Symbol,Float64}($([:( $(QuoteNode(k)) => $(v) ) for (k,v) in param_defaults]...)),
            Expr[$(QuoteNode.(eq_lines)...)],
            Function[$(fn_exprs...)],
            $(n_expect),
            Int[$(forward_indices...)],
            Float64[]
        )
    end |> esc
end

"""Try to parse a line as `keyword: args...`. Returns `(keyword, args)` or `nothing`."""
function _try_parse_declaration(ex::Expr)
    # Julia parses `parameters: α = 0.33, β = 0.99` in various forms depending on
    # whether there are assignments. Handle the common forms:

    # Form 1: :(parameters: α = 0.33, β = 0.99)
    # Julia may parse this as a :tuple or :call with : operator

    # The simplest approach: check if first symbol matches a keyword
    str = string(ex)
    for keyword in (:parameters, :endogenous, :exogenous)
        kw_str = string(keyword)
        if startswith(str, kw_str)
            # This is a declaration line — extract the arguments after the keyword
            return _extract_declaration_args(ex, keyword)
        end
    end
    nothing
end

function _extract_declaration_args(ex::Expr, keyword::Symbol)
    # Walk the expression to find arguments after the keyword
    args = Any[]
    _collect_declaration_args!(args, ex, keyword, false)
    isempty(args) ? nothing : (keyword, args)
end

function _collect_declaration_args!(args, ex::Expr, keyword::Symbol, past_keyword::Bool)
    if past_keyword
        # Everything after keyword is arguments
        push!(args, ex)
        return
    end
    for (i, arg) in enumerate(ex.args)
        if arg == keyword
            # Collect remaining args
            for j in (i+1):length(ex.args)
                push!(args, ex.args[j])
            end
            return
        elseif arg isa Expr
            _collect_declaration_args!(args, arg, keyword, false)
            if !isempty(args)
                return
            end
        end
    end
end
_collect_declaration_args!(args, ::Any, keyword, past_keyword) = nothing
```

**Step 4: Wire up include and export**

In `src/MacroEconometricModels.jl`, after `include("dsge/types.jl")`:
```julia
include("dsge/parser.jl")
```

Add to exports:
```julia
export @dsge
```

**Step 5: Run test to verify it passes**

Run: `julia --project=. -e 'using MacroEconometricModels; include("test/fixtures.jl"); include("test/dsge/test_dsge.jl")'`
Expected: PASS

**Step 6: Commit**

```bash
git add src/dsge/parser.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add @dsge macro parser with E[t] operator support"
```

**Implementation note:** The parser is the most delicate part. Julia's macro system parses `parameters: α = 0.33, β = 0.99` in surprising ways. The implementation may need iteration — test each form interactively with `Meta.show_sexpr(:(parameters: α = 0.33))` to understand how Julia parses the syntax. Be prepared to adjust `_try_parse_declaration` based on actual parse trees. If the colon syntax proves too fragile, fall back to keyword function syntax: `parameters(α = 0.33, β = 0.99)`.

---

### Task 3: Steady State Solver

**Files:**
- Create: `src/dsge/steady_state.jl`
- Modify: `src/MacroEconometricModels.jl` (add include + export)
- Test: `test/dsge/test_dsge.jl` (append SS tests)

**Step 1: Write the failing test**

Append to `test/dsge/test_dsge.jl`:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Steady State
# ─────────────────────────────────────────────────────────────────────────────

@testset "Steady state: AR(1)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec2 = compute_steady_state(spec)
    @test spec2 isa DSGESpec
    @test length(spec2.steady_state) == 1
    @test spec2.steady_state[1] ≈ 0.0 atol=1e-6  # AR(1) SS = 0
end

@testset "Steady state: simple production" begin
    # y = A * k^α, k = s*y, A=1, s=0.3, α=0.33
    # SS: y = k^α, k = 0.3*y → y = (0.3*y)^0.33 → y^(1-0.33) = 0.3^0.33
    spec = @dsge begin
        parameters: α = 0.33, s = 0.3
        endogenous: y, k
        exogenous: ε
        y[t] = k[t-1]^α + ε[t]
        k[t] = s * y[t]
    end
    spec2 = compute_steady_state(spec; initial_guess=[1.0, 0.3])
    @test length(spec2.steady_state) == 2
    # Check SS satisfies equations
    y_ss, k_ss = spec2.steady_state
    @test y_ss ≈ k_ss^0.33 atol=1e-4
    @test k_ss ≈ 0.3 * y_ss atol=1e-4
end

@testset "Steady state: analytical" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    ss_fn = (θ) -> [0.0]  # Known: y_ss = 0 for zero-mean AR
    spec2 = compute_steady_state(spec; method=:analytical, ss_fn=ss_fn)
    @test spec2.steady_state[1] ≈ 0.0
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `compute_steady_state` not defined

**Step 3: Create src/dsge/steady_state.jl**

```julia
# (GPL license header)

"""
Numerical steady-state computation for DSGE models via Optim.jl.
"""

"""
    compute_steady_state(spec::DSGESpec{T}; initial_guess=nothing, method=:auto,
                          ss_fn=nothing) → DSGESpec{T}

Compute the deterministic steady state: f(y_ss, y_ss, y_ss, 0, θ) = 0.

Returns a new `DSGESpec` with the `steady_state` field filled.

# Keywords
- `initial_guess::Vector{T}` — starting point (default: ones)
- `method::Symbol` — `:auto` (NelderMead → LBFGS), `:analytical`
- `ss_fn::Function` — for `:analytical`, a function `θ → y_ss::Vector`
"""
function compute_steady_state(spec::DSGESpec{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        method::Symbol=:auto,
        ss_fn::Union{Nothing,Function}=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    θ = spec.param_values

    if method == :analytical
        ss_fn === nothing && throw(ArgumentError("method=:analytical requires ss_fn"))
        y_ss = T.(ss_fn(θ))
        @assert length(y_ss) == n "ss_fn must return vector of length $n"
        return _update_steady_state(spec, y_ss)
    end

    # Numerical: minimize sum of squared residuals
    y0 = initial_guess !== nothing ? T.(initial_guess) : ones(T, n)
    @assert length(y0) == n "initial_guess must have length $n"

    ε_zero = zeros(T, spec.n_exog)

    # Objective: sum of squared residuals at SS (y_t = y_{t-1} = y_{t+1} = y)
    function ss_objective(y)
        total = zero(T)
        for fn in spec.residual_fns
            r = fn(y, y, y, ε_zero, θ)
            total += r^2
        end
        total
    end

    # Phase 1: Nelder-Mead (derivative-free, robust to bad starting point)
    result = Optim.optimize(ss_objective, y0, Optim.NelderMead(),
                            Optim.Options(iterations=5000, f_reltol=T(1e-12)))
    y_ss = Optim.minimizer(result)

    # Phase 2: Refine with LBFGS if gradient is available
    if Optim.minimum(result) > T(1e-10)
        result2 = Optim.optimize(ss_objective, y_ss, Optim.LBFGS(),
                                 Optim.Options(iterations=2000, f_reltol=T(1e-14)))
        if Optim.minimum(result2) < Optim.minimum(result)
            y_ss = Optim.minimizer(result2)
        end
    end

    # Verify convergence
    final_resid = ss_objective(y_ss)
    final_resid > T(1e-6) && @warn "Steady state may not have converged (residual = $final_resid)"

    _update_steady_state(spec, y_ss)
end

"""Return a new DSGESpec with updated steady_state field."""
function _update_steady_state(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    DSGESpec{T}(
        spec.endog, spec.exog, spec.params, spec.param_values,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, y_ss
    )
end
```

**Step 4: Wire up include and export**

In `src/MacroEconometricModels.jl`, after `include("dsge/parser.jl")`:
```julia
include("dsge/steady_state.jl")
```

Export: `compute_steady_state` already listed in design exports.

**Step 5: Run tests, commit**

```bash
git add src/dsge/steady_state.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add steady-state solver via Optim.jl (NelderMead + LBFGS)"
```

---

### Task 4: Linearization

**Files:**
- Create: `src/dsge/linearize.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Linearization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Linearize: AR(1)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    @test ld isa LinearDSGE{Float64}
    # For y_t = ρ*y_{t-1} + σ*ε_t → Γ₀ = [1], Γ₁ = [ρ], Ψ = [σ], Π = empty
    @test ld.Gamma0[1,1] ≈ 1.0 atol=1e-4
    @test ld.Gamma1[1,1] ≈ 0.9 atol=1e-4
    @test ld.Psi[1,1] ≈ 1.0 atol=1e-4
    @test size(ld.Pi, 2) == 0  # no forward-looking variables
end

@testset "Linearize: forward-looking" begin
    # x_t = 0.5 * E_t[x_{t+1}] + ε_t
    # → Γ₀·x_t = Γ₁·x_{t-1} + Ψ·ε_t + Π·η_t
    # rewritten: x_t - 0.5*x_{t+1} = ε_t
    # Γ₀ = [1], Γ₁ = [0] (no lag), Ψ = [1], Π = [0.5] (coefficient on x_{t+1})
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    @test size(ld.Pi, 2) == 1  # 1 expectation error
    @test abs(ld.Pi[1,1]) > 0.1  # non-zero Π entry
end

@testset "LinearDSGE show" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    io = IOBuffer()
    show(io, ld)
    s = String(take!(io))
    @test occursin("Linearized DSGE", s)
end
```

**Step 2: Create src/dsge/linearize.jl**

```julia
# (GPL license header)

"""
Auto-linearization of DSGE models via numerical Jacobians.

Produces the Sims canonical form: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
"""

"""
    linearize(spec::DSGESpec{T}) → LinearDSGE{T}

Linearize a DSGE model around its steady state using numerical Jacobians.

The model `f(y_t, y_{t-1}, y_{t+1}, ε, θ) = 0` is expanded to first order:

    f_0·ŷ_t + f_1·ŷ_{t-1} + f_lead·ŷ_{t+1} + f_ε·ε = 0

where ŷ denotes deviations from steady state. Rearranging into Sims form:

    Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t

Requires `compute_steady_state` to have been called first.
"""
function linearize(spec::DSGESpec{T}) where {T<:AbstractFloat}
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    n = spec.n_endog
    n_ε = spec.n_exog
    n_η = spec.n_expect
    y_ss = spec.steady_state
    θ = spec.param_values
    ε_zero = zeros(T, n_ε)

    # Compute numerical Jacobians via central differences
    # f_0 = ∂f/∂y_t, f_1 = ∂f/∂y_{t-1}, f_lead = ∂f/∂y_{t+1}, f_ε = ∂f/∂ε

    f_0 = _dsge_jacobian(spec, y_ss, :current)     # n × n
    f_1 = _dsge_jacobian(spec, y_ss, :lag)          # n × n
    f_lead = _dsge_jacobian(spec, y_ss, :lead)      # n × n
    f_ε = _dsge_jacobian_shocks(spec, y_ss)         # n × n_ε

    # Sims canonical form:
    # f_0·ŷ_t + f_1·ŷ_{t-1} + f_lead·ŷ_{t+1} + f_ε·ε = 0
    #
    # Rearrange to: Γ₀·y_t = Γ₁·y_{t-1} + Ψ·ε_t + Π·η_t
    # where Γ₀ = -f_0 - f_lead (absorb lead into current after introducing η)
    #       Γ₁ = f_1
    #       Ψ  = f_ε
    #
    # Actually, the standard approach:
    # f_0·y_t = -f_1·y_{t-1} - f_ε·ε - f_lead·y_{t+1}
    #
    # Introduce η_{t+1} = y_{t+1} - E_t[y_{t+1}]:
    # (f_0 + f_lead)·y_t = -f_1·y_{t-1} - f_ε·ε + f_lead·(y_t - y_{t+1})
    # Wait, the correct formulation:
    #
    # The system is: f_0·y_t + f_1·y_{t-1} + f_lead·E_t[y_{t+1}] + f_ε·ε = 0
    # Let y_{t+1} = E_t[y_{t+1}] + η_{t+1}
    # → f_0·y_t + f_1·y_{t-1} + f_lead·(y_{t+1} - η_{t+1}) + f_ε·ε = 0
    # → f_0·y_t = -f_1·y_{t-1} - f_ε·ε - f_lead·y_{t+1} + f_lead·η_{t+1}
    #
    # In Sims form with augmented state [y_t; y_{t+1}]... but the standard
    # gensys form uses a single y_t vector. The trick:
    #
    # Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
    # where Γ₀ = f_0, Γ₁ = -f_1, Ψ = -f_ε, Π = -f_lead (columns for forward vars)

    Gamma0 = -f_0                           # n × n
    Gamma1 = f_1                            # n × n
    C = zeros(T, n)                         # constants (zero at SS)
    Psi = f_ε                               # n × n_ε

    # Π: select columns of -f_lead corresponding to forward-looking variables
    if n_η > 0
        # Build Π from the columns of f_lead that correspond to forward-looking vars
        # The expectation errors η apply to the forward-looking variables
        # Identify which variables appear with [t+1]
        fwd_var_indices = _forward_variable_indices(spec)
        Pi = -f_lead[:, fwd_var_indices]    # n × n_η
    else
        Pi = zeros(T, n, 0)
    end

    LinearDSGE{T}(Gamma0, Gamma1, C, Psi, Pi, spec)
end

"""Compute Jacobian of residual vector w.r.t. y_t, y_{t-1}, or y_{t+1}."""
function _dsge_jacobian(spec::DSGESpec{T}, y_ss::Vector{T}, which::Symbol) where {T}
    n = spec.n_endog
    θ = spec.param_values
    ε_zero = zeros(T, spec.n_exog)

    J = zeros(T, n, n)
    for j in 1:n
        h = max(T(1e-7), T(1e-7) * abs(y_ss[j]))
        y_plus = copy(y_ss)
        y_minus = copy(y_ss)
        y_plus[j] += h
        y_minus[j] -= h

        for i in 1:n
            fn = spec.residual_fns[i]
            if which == :current
                f_plus = fn(y_plus, y_ss, y_ss, ε_zero, θ)
                f_minus = fn(y_minus, y_ss, y_ss, ε_zero, θ)
            elseif which == :lag
                f_plus = fn(y_ss, y_plus, y_ss, ε_zero, θ)
                f_minus = fn(y_ss, y_minus, y_ss, ε_zero, θ)
            else  # :lead
                f_plus = fn(y_ss, y_ss, y_plus, ε_zero, θ)
                f_minus = fn(y_ss, y_ss, y_minus, ε_zero, θ)
            end
            J[i, j] = (f_plus - f_minus) / (2h)
        end
    end
    J
end

"""Compute Jacobian of residual vector w.r.t. shocks ε."""
function _dsge_jacobian_shocks(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    ε_zero = zeros(T, n_ε)

    J = zeros(T, n, n_ε)
    h = T(1e-7)
    for j in 1:n_ε
        ε_plus = copy(ε_zero)
        ε_minus = copy(ε_zero)
        ε_plus[j] += h
        ε_minus[j] -= h

        for i in 1:n
            fn = spec.residual_fns[i]
            f_plus = fn(y_ss, y_ss, y_ss, ε_plus, θ)
            f_minus = fn(y_ss, y_ss, y_ss, ε_minus, θ)
            J[i, j] = (f_plus - f_minus) / (2h)
        end
    end
    J
end

"""Find indices of endogenous variables that appear with [t+1] in any equation."""
function _forward_variable_indices(spec::DSGESpec{T}) where {T}
    fwd_vars = Set{Int}()
    for eq in spec.equations
        refs = _extract_time_refs(eq, spec.endog, spec.exog)
        for (var, offsets) in refs
            if var ∈ spec.endog && any(o > 0 for o in offsets)
                idx = findfirst(==(var), spec.endog)
                push!(fwd_vars, idx)
            end
        end
    end
    sort(collect(fwd_vars))
end
```

**Step 3: Wire up and commit**

```bash
git add src/dsge/linearize.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add auto-linearization via numerical Jacobians"
```

---

### Task 5: Gensys Solver

**Files:**
- Create: `src/dsge/gensys.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Gensys Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "Gensys: AR(1) model" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test sol isa DSGESolution{Float64}
    @test is_determined(sol)
    @test sol.G1[1,1] ≈ 0.9 atol=1e-4
    @test sol.impact[1,1] ≈ 1.0 atol=1e-4
end

@testset "Gensys: forward-looking model" begin
    # x_t = 0.5 * E_t[x_{t+1}] + ε_t → solution: x_t = 2*ε_t (geometric sum)
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test is_determined(sol)
    # G1 should be 0 (no state persistence), impact should be 2.0
    @test abs(sol.G1[1,1]) < 0.1
    @test sol.impact[1,1] ≈ 2.0 atol=0.2
end

@testset "Gensys: existence/uniqueness flags" begin
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test sol.eu[1] == 1  # exists
    @test sol.eu[2] == 1  # unique
end

@testset "Gensys: default method" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec)  # should default to :gensys
    @test sol.method == :gensys
end
```

**Step 2: Create src/dsge/gensys.jl**

Implement Sims (2002) algorithm using Julia's `schur` from LinearAlgebra:

```julia
# (GPL license header)

"""
Sims (2002) gensys solver for linear rational expectations models.

Solves: Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
Returns: y_t = G1·y_{t-1} + impact·ε_t + C_sol
"""

"""
    gensys(Γ0, Γ1, C, Ψ, Π; div=1.0+1e-8) → (G1, impact, C_sol, eu, eigenvalues)

Solve the linear RE system via QZ decomposition.

`eu = [exist, unique]` where 1 = yes, 0 = no.
`div` is the dividing line between stable (|λ| < div) and unstable eigenvalues.
"""
function gensys(Gamma0::Matrix{T}, Gamma1::Matrix{T}, C::Vector{T},
                Psi::Matrix{T}, Pi::Matrix{T};
                div::T=T(1.0 + 1e-8)) where {T<:AbstractFloat}
    n = size(Gamma0, 1)
    eu = [0, 0]  # [existence, uniqueness]

    # QZ (generalized Schur) decomposition: Gamma0 = Q'*S*Z', Gamma1 = Q'*T*Z'
    F = schur(complex(Gamma0), complex(Gamma1))
    S, TT = F.S, F.T
    Q = F.Q'  # conjugate transpose (Schur convention)
    Z = F.Z

    # Compute generalized eigenvalues
    gev = Vector{ComplexF64}(undef, n)
    for i in 1:n
        if abs(S[i,i]) > eps(T)
            gev[i] = TT[i,i] / S[i,i]
        else
            gev[i] = complex(T(Inf))
        end
    end

    # Reorder: stable eigenvalues (|λ| < div) first
    nunstab = count(i -> abs(gev[i]) >= div, 1:n)
    nstab = n - nunstab

    # Sort by magnitude of eigenvalues (stable first)
    perm = sortperm(abs.(gev))
    S, TT, Q, Z, gev = _reorder_schur(S, TT, Q, Z, gev, perm)

    # Partition: stable block (1:nstab), unstable block (nstab+1:n)
    Q1 = Q[1:nstab, :]
    Q2 = Q[nstab+1:n, :]
    Z11 = Z[:, 1:nstab]
    Z12 = Z[:, nstab+1:n]
    S11 = S[1:nstab, 1:nstab]
    T11 = TT[1:nstab, 1:nstab]

    # Existence check: can expectation errors be eliminated?
    # Check rank condition on Q2 * Π
    if size(Pi, 2) > 0
        Q2Pi = Q2 * complex(Pi)
        etawt = Q2 * complex(Psi)

        # Check if system of equations for η has a solution
        ueta, deta, veta = svd(Q2Pi)
        deta_vals = real.(diag(deta isa AbstractMatrix ? deta : Diagonal(deta)))
        # Actually svd returns (U, s, Vt) where s is a vector
        if deta isa AbstractVector
            deta_vals = real.(deta)
        end
        bigev = count(d -> abs(d) > eps(T) * 1e6, deta_vals)

        if bigev >= nunstab
            eu[1] = 1  # existence
        end

        # Uniqueness check: number of unstable eigenvalues == number of forward-looking vars
        n_fwd = size(Pi, 2)
        if nunstab == n_fwd
            eu[2] = 1  # uniqueness
        elseif nunstab < n_fwd
            eu[2] = 1  # also unique (fewer unstable than forward)
        end
    else
        # No forward-looking variables
        if nunstab == 0
            eu = [1, 1]
        end
    end

    # Build solution matrices
    # G1 = real(Z11 * (S11 \ T11) * Z11')  ... but need to be careful
    if nstab > 0 && nstab <= n
        Z1 = Z[:, 1:nstab]
        S11_inv = inv(S11)
        G1_complex = Z1 * S11_inv * T11 * Z1'

        # Impact matrix
        Q1_Psi = Q1 * complex(Psi)
        impact_complex = Z1 * (S11 \ Q1_Psi)

        G1 = real(Matrix{T}(G1_complex))
        impact = real(Matrix{T}(impact_complex))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, size(Psi, 2))
    end

    # Constants
    C_sol = if norm(C) > eps(T)
        real(Vector{T}((I - G1) \ C))
    else
        zeros(T, n)
    end

    (G1=G1, impact=impact, C_sol=C_sol, eu=eu, eigenvalues=gev)
end

"""Reorder Schur decomposition by permutation."""
function _reorder_schur(S, TT, Q, Z, gev, perm)
    n = size(S, 1)
    S_new = S[perm, perm]
    TT_new = TT[perm, perm]
    Q_new = Q[perm, :]
    Z_new = Z[:, perm]
    gev_new = gev[perm]
    (S_new, TT_new, Q_new, Z_new, gev_new)
end
```

**Step 3: Create the unified `solve` function (add to gensys.jl or a separate solve.jl)**

Add at the end of gensys.jl:

```julia
"""
    solve(spec::DSGESpec; method=:gensys, kwargs...) → DSGESolution or PerfectForesightPath

Solve a DSGE model.

# Methods
- `:gensys` — Sims (2002) QZ decomposition (default)
- `:blanchard_kahn` — Blanchard-Kahn (1980) eigenvalue counting
- `:perfect_foresight` — deterministic Newton solver (requires `T` and `shock_path` kwargs)
"""
function solve(spec::DSGESpec{T}; method::Symbol=:gensys, kwargs...) where {T<:AbstractFloat}
    # Ensure steady state is computed
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end

    if method == :gensys
        ld = linearize(spec)
        result = gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi)
        return DSGESolution{T}(
            result.G1, result.impact, result.C_sol, result.eu,
            :gensys, result.eigenvalues, spec, ld
        )
    elseif method == :blanchard_kahn
        ld = linearize(spec)
        result = blanchard_kahn(ld, spec)
        return result
    elseif method == :perfect_foresight
        return perfect_foresight(spec; kwargs...)
    else
        throw(ArgumentError("method must be :gensys, :blanchard_kahn, or :perfect_foresight"))
    end
end
```

**Step 4: Wire up, run tests, commit**

```bash
git add src/dsge/gensys.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add gensys solver (Sims 2002) and unified solve() API"
```

**Implementation note:** The gensys implementation above is a simplified version. The full Sims (2002) algorithm handles the QZ reordering more carefully using Givens rotations rather than simple permutation. If the permutation-based approach fails on complex models, implement the proper `ordschur!` approach using Julia's `LinearAlgebra.ordschur`. Test with the simple AR(1) and forward-looking cases first, then iterate.

---

### Task 6: Blanchard-Kahn Solver

**Files:**
- Create: `src/dsge/blanchard_kahn.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Blanchard-Kahn
# ─────────────────────────────────────────────────────────────────────────────

@testset "BK: AR(1) model" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:blanchard_kahn)
    @test sol isa DSGESolution{Float64}
    @test is_determined(sol)
    @test sol.G1[1,1] ≈ 0.9 atol=1e-4
    @test sol.method == :blanchard_kahn
end

@testset "BK: agrees with gensys" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol_g = solve(spec; method=:gensys)
    sol_bk = solve(spec; method=:blanchard_kahn)
    @test sol_g.G1 ≈ sol_bk.G1 atol=1e-4
    @test sol_g.impact ≈ sol_bk.impact atol=1e-4
end
```

**Step 2: Create src/dsge/blanchard_kahn.jl**

The BK solver works on the system `A·E_t[z_{t+1}] = B·z_t + C·ε_t` where `z = [predetermined; jump]`. It requires the user (or the linearize step) to classify variables.

The implementation converts from gensys canonical form to BK form, solves, and converts back to `DSGESolution`.

**Step 3: Wire up, test, commit**

```bash
git add src/dsge/blanchard_kahn.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add Blanchard-Kahn (1980) solver"
```

---

### Task 7: Perfect Foresight Solver

**Files:**
- Create: `src/dsge/perfect_foresight.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Perfect Foresight
# ─────────────────────────────────────────────────────────────────────────────

@testset "Perfect foresight: AR(1) impulse" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 50
    shocks = zeros(T_periods, 1)
    shocks[1, 1] = 1.0  # unit shock at t=1
    pf = solve(spec; method=:perfect_foresight, T=T_periods, shock_path=shocks)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
    @test size(pf.path) == (T_periods, 1)
    # y_1 = 0.9*0 + 1.0*1.0 = 1.0
    @test pf.deviations[1, 1] ≈ 1.0 atol=0.1
    # y_2 = 0.9*1.0 = 0.9
    @test pf.deviations[2, 1] ≈ 0.9 atol=0.1
    # Converges to SS
    @test abs(pf.deviations[end, 1]) < 0.01
end
```

**Step 2: Create src/dsge/perfect_foresight.jl**

Newton solver for stacked system. Use `SparseArrays` for the block-tridiagonal Jacobian.

**Step 3: Wire up, test, commit**

```bash
git add src/dsge/perfect_foresight.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add perfect foresight solver (Newton on stacked system)"
```

---

### Task 8: Simulation and IRF/FEVD Bridge

**Files:**
- Create: `src/dsge/simulation.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Simulation & IRF/FEVD Bridge
# ─────────────────────────────────────────────────────────────────────────────

@testset "Simulate: stochastic" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    Random.seed!(42)
    sim = simulate(sol, 200)
    @test size(sim) == (200, 1)
    @test std(sim[:, 1]) > 0  # not all zeros
    @test std(sim[:, 1]) < 1  # bounded
end

@testset "Simulate: deterministic (given shocks)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    shocks = zeros(10, 1)
    shocks[1, 1] = 1.0
    sim = simulate(sol, 10; shock_draws=shocks)
    @test sim[1, 1] ≈ 1.0 atol=1e-6
    @test sim[2, 1] ≈ 0.9 atol=1e-6
end

@testset "IRF bridge to ImpulseResponse" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    irf_result = irf(sol, 20)
    @test irf_result isa ImpulseResponse{Float64}
    @test irf_result.horizon == 20
    @test irf_result.values[1, 1, 1] ≈ 1.0 atol=1e-4  # impact
    @test irf_result.values[2, 1, 1] ≈ 0.9 atol=1e-4  # h=2
    @test length(irf_result.variables) == 1
    @test length(irf_result.shocks) == 1
end

@testset "FEVD bridge to FEVD type" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    fevd_result = fevd(sol, 20)
    @test fevd_result isa FEVD{Float64}
    # Single shock → 100% variance explained
    @test all(fevd_result.proportions[1, 1, :] .≈ 1.0)
end

@testset "plot_result works with DSGE IRF" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    sol = solve(spec)
    irf_result = irf(sol, 20)
    p = plot_result(irf_result)
    @test p isa PlotOutput
end
```

**Step 2: Create src/dsge/simulation.jl**

```julia
# (GPL license header)

"""
Stochastic simulation and IRF/FEVD bridge for DSGE solutions.
"""

"""
    simulate(sol::DSGESolution{T}, T_periods::Int;
             shock_draws=nothing, rng=Random.default_rng()) → Matrix{T}

Simulate the solved DSGE model: y_t = G1·y_{t-1} + impact·ε_t + C_sol.

Returns T_periods × n_endog matrix of levels (SS + deviations).
"""
function simulate(sol::DSGESolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(sol)
    n_ε = nshocks(sol)
    y_ss = sol.spec.steady_state

    # Draw or use provided shocks
    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_ε)
        ε = T.(shock_draws)
    else
        ε = randn(rng, T, T_periods, n_ε)
    end

    # Simulate deviations from steady state
    dev = zeros(T, T_periods, n)
    for t in 1:T_periods
        y_prev = t == 1 ? zeros(T, n) : dev[t-1, :]
        dev[t, :] = sol.G1 * y_prev + sol.impact * ε[t, :] + sol.C_sol
    end

    # Return levels (SS + deviations)
    levels = dev .+ y_ss'
    levels
end

# =============================================================================
# IRF dispatch for DSGESolution
# =============================================================================

"""
    irf(sol::DSGESolution{T}, horizon::Int; kwargs...) → ImpulseResponse{T}

Compute analytical impulse responses from a solved DSGE model.

IRF at horizon h for shock j: Φ_h[:,j] = G1^h * impact[:,j]
"""
function irf(sol::DSGESolution{T}, horizon::Int;
             ci_type::Symbol=:none, reps::Int=200, conf_level::Real=0.95
             ) where {T<:AbstractFloat}
    n = nvars(sol)
    n_ε = nshocks(sol)

    # Analytical IRFs
    point_irf = zeros(T, horizon, n, n_ε)
    G1_power = Matrix{T}(I, n, n)

    for h in 1:horizon
        if h == 1
            G1_power = Matrix{T}(I, n, n)
        else
            G1_power = G1_power * sol.G1
        end
        for j in 1:n_ε
            point_irf[h, :, j] = G1_power * sol.impact[:, j]
        end
    end

    # Variable and shock names
    var_names = sol.spec.varnames
    shock_names = [string(s) for s in sol.spec.exog]

    ci_lower = zeros(T, horizon, n, n_ε)
    ci_upper = zeros(T, horizon, n, n_ε)
    cl = ci_type == :none ? zero(T) : T(conf_level)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type, nothing, cl)
end

# =============================================================================
# FEVD dispatch for DSGESolution
# =============================================================================

"""
    fevd(sol::DSGESolution{T}, horizon::Int) → FEVD{T}

Compute FEVD from a solved DSGE model.
"""
function fevd(sol::DSGESolution{T}, horizon::Int) where {T<:AbstractFloat}
    irf_result = irf(sol, horizon)
    n = nvars(sol)
    decomp, props = _compute_fevd(irf_result.values, n, horizon)
    var_names = sol.spec.varnames
    shock_names = [string(s) for s in sol.spec.exog]
    FEVD{T}(decomp, props, var_names, shock_names)
end
```

**Step 3: Wire up includes**

In `src/MacroEconometricModels.jl`, add after `include("dsge/gensys.jl")` (or wherever the last dsge file is):

```julia
include("dsge/simulation.jl")
```

**Important**: The `irf` and `fevd` dispatches on `DSGESolution` must be included **after** the existing `irf`/`fevd` functions in `var/irf.jl` and `var/fevd.jl`. Place the DSGE simulation.jl include after the innovation accounting block (after line 234 `include("lp/fevd.jl")`).

Add exports:
```julia
export solve, simulate
```

(`irf` and `fevd` are already exported — the new dispatches on `DSGESolution` just add methods.)

**Step 4: Run tests, commit**

```bash
git add src/dsge/simulation.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add simulation, IRF/FEVD bridge to existing types"
```

---

### Task 9: GMM Estimation

**Files:**
- Create: `src/dsge/estimation.jl`
- Modify: `src/MacroEconometricModels.jl`
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 9: GMM Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "IRF matching: recover AR(1) parameter" begin
    # True model: y_t = 0.8 * y_{t-1} + ε_t
    Random.seed!(42)
    rng = Random.MersenneTwister(42)
    T_obs = 500
    y_true = zeros(T_obs)
    for t in 2:T_obs
        y_true[t] = 0.8 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5  # starting guess (not true value)
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:irf_matching, irf_horizon=10)
    @test est isa DSGEEstimation{Float64}
    @test est.converged
    @test est.theta[1] ≈ 0.8 atol=0.15  # recovered ρ near 0.8
end

@testset "Euler GMM: basic" begin
    Random.seed!(123)
    rng = Random.MersenneTwister(123)
    T_obs = 300
    y_true = zeros(T_obs)
    for t in 2:T_obs
        y_true[t] = 0.7 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:euler_gmm, n_lags_instruments=4)
    @test est isa DSGEEstimation{Float64}
    @test est.theta[1] ≈ 0.7 atol=0.2
end

@testset "DSGEEstimation show and report" begin
    Random.seed!(42)
    rng = Random.MersenneTwister(42)
    y_true = zeros(200)
    for t in 2:200
        y_true[t] = 0.8 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:irf_matching, irf_horizon=10)
    io = IOBuffer()
    show(io, est)
    s = String(take!(io))
    @test occursin("DSGE Estimation", s)
    @test occursin("Estimated Parameters", s)
end
```

**Step 2: Create src/dsge/estimation.jl**

```julia
# (GPL license header)

"""
GMM estimation of DSGE model parameters.

Supports two methods:
- IRF matching (Christiano, Eichenbaum & Evans 2005)
- Euler equation GMM (Hansen & Singleton 1982)
"""

"""
    estimate_dsge(spec, data, param_names; method=:irf_matching, ...) → DSGEEstimation

Estimate DSGE deep parameters via GMM.

# Arguments
- `spec::DSGESpec{T}` — model specification (initial parameter values used as starting point)
- `data::AbstractMatrix{T}` — T_obs × n_vars data matrix
- `param_names::Vector{Symbol}` — which parameters to estimate

# Keywords
- `method::Symbol` — `:irf_matching` or `:euler_gmm`
- `target_irfs::ImpulseResponse` — pre-computed target IRFs (optional; if missing, VAR IRFs computed)
- `var_lags::Int=4` — lag order for VAR when computing target IRFs
- `irf_horizon::Int=20` — horizon for IRF matching
- `weighting::Symbol=:two_step` — GMM weighting
- `n_lags_instruments::Int=4` — instrument lags for Euler equation GMM
"""
function estimate_dsge(spec::DSGESpec{T}, data::AbstractMatrix,
                        param_names::Vector{Symbol};
                        method::Symbol=:irf_matching,
                        target_irfs::Union{Nothing,ImpulseResponse}=nothing,
                        var_lags::Int=4, irf_horizon::Int=20,
                        weighting::Symbol=:two_step,
                        n_lags_instruments::Int=4) where {T<:AbstractFloat}
    data_T = T.(data)

    if method == :irf_matching
        return _estimate_irf_matching(spec, data_T, param_names;
                                       target_irfs=target_irfs,
                                       var_lags=var_lags,
                                       irf_horizon=irf_horizon,
                                       weighting=weighting)
    elseif method == :euler_gmm
        return _estimate_euler_gmm(spec, data_T, param_names;
                                    n_lags=n_lags_instruments,
                                    weighting=weighting)
    else
        throw(ArgumentError("method must be :irf_matching or :euler_gmm"))
    end
end

function _estimate_irf_matching(spec::DSGESpec{T}, data::Matrix{T},
                                 param_names::Vector{Symbol};
                                 target_irfs=nothing, var_lags=4,
                                 irf_horizon=20, weighting=:two_step) where {T}
    n_est = length(param_names)

    # Step 1: Compute target IRFs from VAR if not provided
    if target_irfs === nothing
        var_model = estimate_var(data, var_lags)
        target_irfs = irf(var_model, irf_horizon; method=:cholesky)
    end
    target_vec = vec(target_irfs.values)
    n_moments = length(target_vec)

    # Step 2: Build moment function
    # θ is the vector of parameters being estimated
    theta0 = T[spec.param_values[p] for p in param_names]

    function moment_fn(theta, _data)
        # Update spec with candidate parameters
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        # Build updated spec
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[]
        )
        # Solve and compute IRFs
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol)
                return fill(T(1e6), 1, n_moments)  # penalty for indeterminacy
            end
            model_irfs = irf(sol, irf_horizon)
            model_vec = vec(model_irfs.values)
            # Distance: each "observation" is one moment
            reshape(model_vec .- target_vec, 1, :)
        catch
            fill(T(1e6), 1, n_moments)
        end
    end

    # Step 3: Estimate via existing GMM
    gmm_result = estimate_gmm(moment_fn, theta0, data; weighting=weighting)

    # Step 4: Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = gmm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[]
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :irf_matching, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end

function _estimate_euler_gmm(spec::DSGESpec{T}, data::Matrix{T},
                              param_names::Vector{Symbol};
                              n_lags=4, weighting=:two_step) where {T}
    T_obs, n_vars = size(data)
    n_est = length(param_names)
    theta0 = T[spec.param_values[p] for p in param_names]

    # Build instruments: lagged variables
    t_start = n_lags + 2  # need lags + 1 lead
    T_eff = T_obs - t_start

    function moment_fn(theta, _data)
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end

        n_eq = spec.n_endog
        n_inst = n_vars * n_lags  # lagged variables as instruments
        moments = zeros(T, T_eff, n_eq * n_inst)

        for (idx, t) in enumerate(t_start:(T_obs-1))
            y_t = data[t, :]
            y_lag = data[t-1, :]
            y_lead = data[t+1, :]
            ε_zero = zeros(T, spec.n_exog)

            # Compute Euler equation residuals
            resids = zeros(T, n_eq)
            for i in 1:n_eq
                resids[i] = spec.residual_fns[i](y_t, y_lag, y_lead, ε_zero, new_pv)
            end

            # Instruments: n_lags lags of all variables
            Z = zeros(T, n_inst)
            col = 1
            for lag in 1:n_lags
                for v in 1:n_vars
                    Z[col] = data[t-lag, v]
                    col += 1
                end
            end

            # Moments: Z ⊗ residuals (Kronecker product)
            moments[idx, :] = kron(resids, Z)
        end
        moments
    end

    gmm_result = estimate_gmm(moment_fn, theta0, data; weighting=weighting)

    # Build solution at estimated params
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = gmm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[]
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :euler_gmm, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end

# Float fallback
function estimate_dsge(spec::DSGESpec{T}, data::AbstractMatrix, param_names::Vector{Symbol};
                        kwargs...) where {T}
    estimate_dsge(spec, Float64.(data), param_names; kwargs...)
end
```

**Step 3: Wire up, test, commit**

```bash
git add src/dsge/estimation.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add GMM estimation (IRF matching + Euler equation)"
```

---

### Task 10: Integration — runtests.jl and Full Workflow

**Files:**
- Modify: `test/runtests.jl` (add DSGE test group)
- Test: `test/dsge/test_dsge.jl` (add integration test)

**Step 1: Add integration test at end of test_dsge.jl**

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Full Workflow Integration
# ─────────────────────────────────────────────────────────────────────────────

@testset "Full workflow: specify → solve → analyze" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end

    # Solve
    sol = solve(spec)
    @test is_determined(sol)

    # IRF
    irf_result = irf(sol, 40)
    @test irf_result isa ImpulseResponse
    @test irf_result.values[1, 1, 1] ≈ 0.01 atol=1e-3

    # FEVD
    fevd_result = fevd(sol, 40)
    @test fevd_result isa FEVD

    # Simulate
    Random.seed!(42)
    sim = simulate(sol, 200)
    @test size(sim) == (200, 1)

    # Plot (smoke test)
    p = plot_result(irf_result)
    @test p isa PlotOutput

    # Display
    io = IOBuffer()
    show(io, sol)
    @test occursin("DSGE Solution", String(take!(io)))
end
```

**Step 2: Add DSGE to runtests.jl**

Add a new test group after Group 6 (line 99):

```julia
    # Group 7: DSGE Models
    ("DSGE Models" => [
        "dsge/test_dsge.jl",
    ]),
```

Add to the sequential section (after line 280):

```julia
        # Group 7: DSGE Models
        @testset "DSGE Models" begin include("dsge/test_dsge.jl") end
```

**Step 3: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: ALL PASS (existing tests unaffected, DSGE tests pass)

**Step 4: Commit**

```bash
git add test/runtests.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): integrate test suite, add full workflow integration test"
```

---

### Task 11: Final Polish — refs(), report(), CLAUDE.md

**Files:**
- Modify: `src/summary.jl` (add `report` dispatch for DSGESolution)
- Modify: `src/summary_refs.jl` (add DSGE references)
- Modify: `src/MacroEconometricModels.jl` (verify all exports)

**Step 1: Add DSGE references**

In the `_REFERENCES` database (in `src/summary_refs.jl`), add:

```julia
:sims2002 => Reference("Sims, C. A.", 2002, ...),
:blanchard_kahn1980 => Reference("Blanchard, O. J. & Kahn, C. M.", 1980, ...),
:christiano2005 => Reference("Christiano, L. J., Eichenbaum, M. & Evans, C. L.", 2005, ...),
:hansen_singleton1982 => Reference("Hansen, L. P. & Singleton, K. J.", 1982, ...),
```

**Step 2: Add `refs()` dispatch for DSGESolution**

```julia
function refs(::DSGESolution; format::Symbol=:text)
    _format_refs([:sims2002, :blanchard_kahn1980]; format=format)
end

function refs(::DSGEEstimation; format::Symbol=:text)
    # Method-dependent
    _format_refs([:sims2002, :christiano2005, :hansen_singleton1982]; format=format)
end
```

**Step 3: Verify all exports are complete**

Ensure `MacroEconometricModels.jl` exports:
```julia
export AbstractDSGEModel, DSGESpec, LinearDSGE, DSGESolution
export PerfectForesightPath, DSGEEstimation
export @dsge
export solve, compute_steady_state, linearize
export gensys, blanchard_kahn
export estimate_dsge
export simulate
export nshocks, is_determined, is_stable
```

**Step 4: Test refs**

```julia
@testset "refs() for DSGE" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    sol = solve(spec)
    r = refs(sol)
    @test occursin("Sims", r)
end
```

**Step 5: Commit**

```bash
git add src/summary.jl src/summary_refs.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add refs(), report() dispatches, verify exports"
```

---

## Include Order Summary

The final include order in `src/MacroEconometricModels.jl` for DSGE files:

```julia
# After nowcast/ block (line 157) and before GMM (line 160):
include("dsge/types.jl")
include("dsge/parser.jl")
include("dsge/steady_state.jl")
include("dsge/linearize.jl")
include("dsge/gensys.jl")
include("dsge/blanchard_kahn.jl")
include("dsge/perfect_foresight.jl")
include("dsge/estimation.jl")

# After lp/fevd.jl (line 234), before summary.jl:
include("dsge/simulation.jl")  # needs irf/fevd infrastructure
```

---

## Critical Implementation Notes

1. **Macro parsing is the hardest part.** Julia parses `parameters: α = 0.33` in surprising ways. Use `dump(:(parameters: α = 0.33))` interactively to understand the AST. Be prepared to iterate on `_try_parse_declaration`. A fallback syntax like `parameters(α = 0.33)` (function call style) may be needed.

2. **Gensys QZ reordering.** The simple permutation-based reorder in Task 5 is a simplification. Julia's `LinearAlgebra.ordschur` can reorder the Schur form properly. If simple models work but complex ones fail, switch to `ordschur`.

3. **The `E[t](expr)` parsing.** Julia parses `E[t](1/C[t+1])` as `call(ref(E, t), call(/, 1, ref(C, call(+, t, 1))))`. Verify with `Meta.show_sexpr`.

4. **eval() in parser.** The `_build_residual_fn` uses `eval` to create closures. This is acceptable in macro context since it happens at compile time. However, the functions are not world-age compatible — they work inside the module but test carefully for edge cases.

5. **Perfect foresight Jacobian.** Use `SparseArrays` for the T×n by T×n block-tridiagonal system. Each block row has at most 3 non-zero n×n blocks (lag, current, lead). `sparse(I, J, V)` construction is efficient.

6. **Name conflicts.** `solve` is a new export — check it doesn't conflict with anything in `LinearAlgebra` (it doesn't; `LinearAlgebra` doesn't export `solve`). `simulate` is also new. Check no conflicts with existing package exports.
