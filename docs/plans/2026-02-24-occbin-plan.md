# OccBin Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Guerrieri & Iacoviello (2015) piecewise-linear OccBin solver for DSGE models with one or two occasionally binding constraints.

**Architecture:** New `src/dsge/occbin.jl` file containing constraint parsing, regime derivation, backward iteration for time-varying decision rules, guess-and-verify convergence loop, and IRF computation. Four new types added to `types.jl`: `OccBinConstraint`, `OccBinRegime`, `OccBinSolution`, `OccBinIRF`. Integration with plotting, references, and display.

**Tech Stack:** Julia, LinearAlgebra, existing DSGE pipeline (`@dsge`, `linearize`, `solve`)

---

### Task 1: Add OccBin Types to types.jl

**Files:**
- Modify: `src/dsge/types.jl:315` (after DSGEEstimation show method, before EOF)
- Test: `test/dsge/test_dsge.jl:1383` (before `end # top-level @testset`)

**Step 1: Write the failing test**

Add before line 1384 in `test/dsge/test_dsge.jl`:

```julia
# ─────────────────────────────────────────────────────────────────────────────
# Section 15: OccBin Constraint Solution
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: types" begin
    # OccBinConstraint
    c = OccBinConstraint{Float64}(:(i[t] >= 0), :i, 0.0, :geq, :(i[t] = 0))
    @test c.variable == :i
    @test c.bound == 0.0
    @test c.direction == :geq

    # OccBinRegime
    A = [0.5 0.0; 0.0 0.3]
    B = [1.0 0.1; 0.0 1.0]
    C = [0.2 0.0; 0.0 0.1]
    D = [1.0 0.0; 0.0 1.0]
    r = OccBinRegime{Float64}(A, B, C, D)
    @test size(r.A) == (2, 2)
    @test size(r.D) == (2, 2)

    # OccBinSolution
    lp = zeros(10, 2)
    pp = zeros(10, 2)
    ss = [1.0, 0.5]
    rh = zeros(Int, 10, 1)
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = y[t]
    end
    sol = OccBinSolution{Float64}(lp, pp, ss, rh, true, 3, spec, ["y", "i"])
    @test sol.converged
    @test sol.iterations == 3
    @test size(sol.piecewise_path) == (10, 2)

    # OccBinIRF
    oirf = OccBinIRF{Float64}(lp, pp, rh, ["y", "i"], "e")
    @test oirf.shock_name == "e"
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels; println(isdefined(MacroEconometricModels, :OccBinConstraint))'`
Expected: `false`

**Step 3: Write minimal implementation**

Add at the end of `src/dsge/types.jl` (after line 314):

```julia
# =============================================================================
# OccBin — Occasionally Binding Constraints (Guerrieri & Iacoviello 2015)
# =============================================================================

"""
    OccBinConstraint{T} — parsed inequality constraint for OccBin solver.

Fields:
- `expr` — original Julia expression (e.g., `:(i[t] >= 0)`)
- `variable` — constrained variable symbol
- `bound` — bound value
- `direction` — `:geq` or `:leq`
- `bind_expr` — substitution expression when constraint binds
"""
struct OccBinConstraint{T<:AbstractFloat}
    expr::Expr
    variable::Symbol
    bound::T
    direction::Symbol
    bind_expr::Expr
end

"""
    OccBinRegime{T} — linearized coefficient matrices for one regime.

Stores the model in the form: A·y_{t-1} + B·y_t + C·y_{t+1} + D·ε_t = 0

Fields:
- `A` — lag coefficients (n × n)
- `B` — current coefficients (n × n)
- `C` — forward coefficients (n × n)
- `D` — shock coefficients (n × n_shocks)
"""
struct OccBinRegime{T<:AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
end

"""
    OccBinSolution{T} — result of `occbin_solve`.

Fields:
- `linear_path` — unconstrained solution (T_periods × n)
- `piecewise_path` — constrained solution (T_periods × n)
- `steady_state` — SS values
- `regime_history` — regime indicator per period per constraint (T_periods × n_constraints)
- `converged` — convergence flag
- `iterations` — iteration count
- `spec` — model specification
- `varnames` — display names
"""
struct OccBinSolution{T<:AbstractFloat}
    linear_path::Matrix{T}
    piecewise_path::Matrix{T}
    steady_state::Vector{T}
    regime_history::Matrix{Int}
    converged::Bool
    iterations::Int
    spec::DSGESpec{T}
    varnames::Vector{String}
end

"""
    OccBinIRF{T} — IRF comparison: linear vs piecewise under constraint.

Fields:
- `linear` — unconstrained IRF (horizon × n)
- `piecewise` — constrained IRF (horizon × n)
- `regime_history` — binding periods per constraint
- `varnames` — variable display names
- `shock_name` — name of the shocked variable
"""
struct OccBinIRF{T<:AbstractFloat}
    linear::Matrix{T}
    piecewise::Matrix{T}
    regime_history::Matrix{Int}
    varnames::Vector{String}
    shock_name::String
end

function Base.show(io::IO, sol::OccBinSolution{T}) where {T}
    n_binding = sum(sol.regime_history .> 0)
    n_periods = size(sol.piecewise_path, 1)
    n_constraints = size(sol.regime_history, 2)
    spec_data = Any[
        "Variables"     sol.spec.n_endog;
        "Periods"       n_periods;
        "Constraints"   n_constraints;
        "Binding periods" n_binding;
        "Converged"     sol.converged ? "Yes" : "No";
        "Iterations"    sol.iterations;
    ]
    _pretty_table(io, spec_data;
        title = "OccBin Piecewise-Linear Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, oirf::OccBinIRF{T}) where {T}
    max_dev = maximum(abs.(oirf.piecewise .- oirf.linear))
    n_binding = sum(oirf.regime_history .> 0)
    spec_data = Any[
        "Shock"           oirf.shock_name;
        "Variables"       length(oirf.varnames);
        "Horizon"         size(oirf.piecewise, 1);
        "Binding periods" n_binding;
        "Max deviation"   round(max_dev; digits=6);
    ]
    _pretty_table(io, spec_data;
        title = "OccBin IRF Comparison",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

report(sol::OccBinSolution) = show(stdout, sol)
report(oirf::OccBinIRF) = show(stdout, oirf)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: DSGE Models section passes with new OccBin type tests

**Step 5: Commit**

```bash
git add src/dsge/types.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add OccBin types — OccBinConstraint, OccBinRegime, OccBinSolution, OccBinIRF"
```

---

### Task 2: Constraint Parsing and Regime Derivation

**Files:**
- Create: `src/dsge/occbin.jl`
- Modify: `src/MacroEconometricModels.jl:167` (add include after perfect_foresight.jl)
- Modify: `src/MacroEconometricModels.jl:335` (add exports)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to the Section 15 testsets in `test/dsge/test_dsge.jl`:

```julia
@testset "OccBin: parse_constraint" begin
    spec = @dsge begin
        parameters: rho = 0.5, phi = 1.5, sigma = 0.01
        endogenous: y, pi_var, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        pi_var[t] = 0.5 * y[t]
        i[t] = phi * pi_var[t]
    end

    # Basic >= constraint
    c = parse_constraint(:(i[t] >= 0), spec)
    @test c.variable == :i
    @test c.bound == 0.0
    @test c.direction == :geq

    # <= constraint
    c2 = parse_constraint(:(y[t] <= 1.0), spec)
    @test c2.variable == :y
    @test c2.bound == 1.0
    @test c2.direction == :leq

    # Invalid variable
    @test_throws ArgumentError parse_constraint(:(z[t] >= 0), spec)
end

@testset "OccBin: _derive_alternative_regime" begin
    spec = @dsge begin
        parameters: rho = 0.5, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    c = parse_constraint(:(i[t] >= 0), spec)
    alt_spec = MacroEconometricModels._derive_alternative_regime(spec, c)

    # Alternative spec should have same variables but different equation for i
    @test alt_spec.n_endog == spec.n_endog
    @test alt_spec.endog == spec.endog

    # In the alternative regime, i[t] = 0, so residual for equation 2 at SS should be:
    # i - 0 = 0 (at SS where i_ss = 0)
    y_ss = alt_spec.steady_state
    θ = alt_spec.param_values
    ε_zero = zeros(spec.n_exog)
    resid = alt_spec.residual_fns[2](y_ss, y_ss, y_ss, ε_zero, θ)
    @test abs(resid) < 1e-10  # equation 2 is now i[t] = 0
end

@testset "OccBin: _extract_regime" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    regime = MacroEconometricModels._extract_regime(spec)
    @test isa(regime, OccBinRegime{Float64})
    @test size(regime.A) == (1, 1)
    @test size(regime.B) == (1, 1)
    @test size(regime.C) == (1, 1)
    @test size(regime.D) == (1, 1)
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels; println(isdefined(MacroEconometricModels, :parse_constraint))'`
Expected: `false`

**Step 3: Write minimal implementation**

Create `src/dsge/occbin.jl`:

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
OccBin piecewise-linear solver for DSGE models with occasionally binding constraints.

References:
- Guerrieri, L. and Iacoviello, M. (2015). OccBin: A Toolkit for Solving Dynamic Models
  with Occasionally Binding Constraints Easily. Journal of Monetary Economics, 70, 22-38.
"""

using LinearAlgebra

"""
    parse_constraint(expr::Expr, spec::DSGESpec{T}) → OccBinConstraint{T}

Parse a constraint expression like `:(i[t] >= 0)` or `:(debt[t] <= collateral)`.

The constraint must reference an endogenous variable with time index `[t]`.
"""
function parse_constraint(expr::Expr, spec::DSGESpec{T}) where {T<:AbstractFloat}
    # Expect: (call, >=, var[t], bound) or (comparison, var[t], >=, bound)
    variable, bound_val, direction = _parse_constraint_expr(expr, spec)

    variable ∈ spec.endog || throw(ArgumentError(
        "Constraint variable :$variable not found in endogenous variables: $(spec.endog)"))

    bind_expr = Expr(:(=), Expr(:ref, variable, :t), bound_val)

    OccBinConstraint{T}(expr, variable, T(bound_val), direction, bind_expr)
end

"""Parse the constraint expression into (variable, bound, direction)."""
function _parse_constraint_expr(expr::Expr, spec::DSGESpec)
    # Handle: :(var[t] >= bound) or :(var[t] <= bound)
    if expr.head == :call && length(expr.args) == 3
        op = expr.args[1]
        lhs = expr.args[2]
        rhs = expr.args[3]

        if op == :>= || op == :≥
            var = _extract_constrained_var(lhs)
            return (var, _eval_bound(rhs), :geq)
        elseif op == :<= || op == :≤
            var = _extract_constrained_var(lhs)
            return (var, _eval_bound(rhs), :leq)
        end
    end
    # Handle comparison syntax: :(var[t] >= bound)
    if expr.head == :comparison && length(expr.args) == 3
        lhs = expr.args[1]
        op = expr.args[2]
        rhs = expr.args[3]
        if op == :>= || op == :≥
            var = _extract_constrained_var(lhs)
            return (var, _eval_bound(rhs), :geq)
        elseif op == :<= || op == :≤
            var = _extract_constrained_var(lhs)
            return (var, _eval_bound(rhs), :leq)
        end
    end
    throw(ArgumentError("Cannot parse constraint expression: $expr. " *
        "Expected format: :(var[t] >= bound) or :(var[t] <= bound)"))
end

"""Extract variable name from var[t] expression."""
function _extract_constrained_var(ex)
    if ex isa Expr && ex.head == :ref && length(ex.args) == 2 && ex.args[2] === :t
        return ex.args[1]::Symbol
    end
    throw(ArgumentError("Constraint LHS must be var[t], got: $ex"))
end

"""Evaluate bound value (literal number or expression)."""
function _eval_bound(ex)
    if ex isa Number
        return Float64(ex)
    elseif ex isa Symbol
        return 0.0  # will be resolved later
    end
    try
        return Float64(eval(ex))
    catch
        throw(ArgumentError("Cannot evaluate constraint bound: $ex"))
    end
end

"""
    _derive_alternative_regime(spec::DSGESpec{T}, constraint::OccBinConstraint{T}) → DSGESpec{T}

Create an alternative model specification where the constrained variable's equation
is replaced by `var[t] = bound`.
"""
function _derive_alternative_regime(spec::DSGESpec{T}, constraint::OccBinConstraint{T}) where {T}
    var = constraint.variable
    bound = constraint.bound
    var_idx = findfirst(==(var), spec.endog)
    var_idx === nothing && throw(ArgumentError("Variable :$var not in endogenous variables"))

    # Build new residual function for the binding equation: var[t] - bound = 0
    new_resid_fn = let idx = var_idx, b = bound
        (y_t, y_lag, y_lead, ε, θ) -> y_t[idx] - b
    end

    # Build new equation expression
    new_eq = Expr(:(=), Expr(:ref, var, :t), bound)

    # Copy and replace
    new_residual_fns = copy(spec.residual_fns)
    new_residual_fns[var_idx] = new_resid_fn

    new_equations = copy(spec.equations)
    new_equations[var_idx] = new_eq

    # Recompute forward-looking indices (binding equation var[t]=bound has no leads)
    new_forward_indices = Int[]
    for (i, eq) in enumerate(new_equations)
        if i == var_idx
            continue  # binding constraint is never forward-looking
        elseif i in spec.forward_indices
            push!(new_forward_indices, i)
        end
    end
    n_expect_new = length(new_forward_indices)

    # Create alternative spec (reuse steady state from reference)
    alt_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, spec.param_values,
        new_equations, new_residual_fns,
        n_expect_new, new_forward_indices, spec.steady_state, spec.ss_fn
    )

    alt_spec
end

"""
    _extract_regime(spec::DSGESpec{T}) → OccBinRegime{T}

Extract linearized coefficient matrices (A, B, C, D) from a DSGESpec.

The model in implicit form: f(y_t, y_{t-1}, y_{t+1}, ε) = 0
Linearized: f_1·ŷ_{t-1} + f_0·ŷ_t + f_lead·ŷ_{t+1} + f_ε·ε = 0

Returns OccBinRegime with A=f_1, B=f_0, C=f_lead, D=f_ε.
"""
function _extract_regime(spec::DSGESpec{T}) where {T}
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first"))

    y_ss = spec.steady_state
    f_0 = _dsge_jacobian(spec, y_ss, :current)
    f_1 = _dsge_jacobian(spec, y_ss, :lag)
    f_lead = _dsge_jacobian(spec, y_ss, :lead)
    f_ε = _dsge_jacobian_shocks(spec, y_ss)

    OccBinRegime{T}(f_1, f_0, f_lead, f_ε)
end
```

Add include in `src/MacroEconometricModels.jl` at line 167 (after `include("dsge/perfect_foresight.jl")`):

```julia
include("dsge/occbin.jl")
```

Add exports at line 335 (after `export nshocks, is_determined, is_stable`):

```julia
export OccBinConstraint, OccBinRegime, OccBinSolution, OccBinIRF
export parse_constraint, occbin_solve, occbin_irf
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: DSGE Models section passes with constraint parsing tests

**Step 5: Commit**

```bash
git add src/dsge/occbin.jl src/MacroEconometricModels.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add OccBin constraint parsing and regime derivation"
```

---

### Task 3: One-Constraint Solver (Backward Iteration + Guess-and-Verify)

**Files:**
- Modify: `src/dsge/occbin.jl` (append solver functions)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to Section 15 in `test/dsge/test_dsge.jl`:

```julia
@testset "OccBin: _map_regime" begin
    violvec = BitVector([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    regimes, starts = MacroEconometricModels._map_regime(violvec)
    @test regimes == [0, 1, 0, 1, 0]
    @test starts == [1, 3, 6, 8, 9]
end

@testset "OccBin: one-constraint ZLB" begin
    # Simple model: y[t] = rho*y[t-1] + e[t], i[t] = phi*y[t]
    # ZLB: i[t] >= 0
    # Large negative shock should trigger ZLB
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)

    # Large negative shock to trigger ZLB
    shock_path = zeros(40, spec.n_exog)
    shock_path[1, 1] = -2.0

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=40)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
    @test size(sol.piecewise_path) == (40, 2)
    @test size(sol.linear_path) == (40, 2)

    # Linear path should go negative for i
    i_idx = 2
    @test minimum(sol.linear_path[:, i_idx]) < 0.0

    # Piecewise path should clamp i >= 0 (within tolerance)
    @test minimum(sol.piecewise_path[:, i_idx]) >= -1e-8

    # Some periods should be binding
    @test sum(sol.regime_history[:, 1]) > 0
end

@testset "OccBin: no-binding case" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= -100.0), spec)

    # Tiny shock — constraint never binds
    shock_path = zeros(20, 1)
    shock_path[1, 1] = 0.01

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)
    @test sol.converged
    @test sum(sol.regime_history) == 0  # never binding

    # Linear and piecewise should be identical
    @test sol.piecewise_path ≈ sol.linear_path atol=1e-10
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels; println(isdefined(MacroEconometricModels, :occbin_solve))'`
Expected: `false` (function not yet defined) or method error

**Step 3: Write minimal implementation**

Append to `src/dsge/occbin.jl`:

```julia
# =============================================================================
# Regime Mapping
# =============================================================================

"""
    _map_regime(violvec::BitVector) → (regimes, starts)

Identify contiguous blocks of binding/non-binding periods.
Returns vectors of regime indicators and start indices.
"""
function _map_regime(violvec::BitVector)
    n = length(violvec)
    n == 0 && return (Int[], Int[])

    regimes = Int[violvec[1] ? 1 : 0]
    starts = Int[1]

    for i in 2:n
        current = violvec[i] ? 1 : 0
        if current != regimes[end]
            push!(regimes, current)
            push!(starts, i)
        end
    end

    (regimes, starts)
end

# =============================================================================
# Time-Varying Decision Rules — Backward Iteration
# =============================================================================

"""
    _backward_iteration(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                        P::Matrix{T}, Q::Matrix{T},
                        violvec::BitVector, shock_path::Matrix{T}) → (P_tv, D_tv, E)

Compute time-varying decision rules by backward iteration from the last
binding period. Returns arrays of per-period state transition matrices (P_tv),
constant terms (D_tv), and initial shock impact (E).

Algorithm (Guerrieri & Iacoviello 2015):
At T_max (last binding period), assuming linear rule applies from T_max+1:
    P_T = -[B* + C*·P]⁻¹ · A*
For t = T_max-1 down to 1, using the appropriate regime's coefficients:
    P_t = -[B_t + C_t·P_{t+1}]⁻¹ · A_t
    D_t = -[B_t + C_t·P_{t+1}]⁻¹ · [C_t·D_{t+1} + D_shock_t]
"""
function _backward_iteration(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                              P::Matrix{T}, Q::Matrix{T},
                              violvec::BitVector, shock_path::Matrix{T}) where {T}
    n = size(P, 1)
    nperiods = length(violvec)
    n_shocks = size(Q, 2)

    # Find last binding period
    T_max = findlast(violvec)
    if T_max === nothing
        # No binding — return empty
        P_tv = Array{T,3}(undef, n, n, 0)
        D_tv = Matrix{T}(undef, n, 0)
        E = Q  # standard shock impact
        return (P_tv, D_tv, E)
    end

    P_tv = zeros(T, n, n, T_max)
    D_tv = zeros(T, n, T_max)

    # At T_max: assume linear rule from T_max+1 onward
    # Select regime coefficients
    rgm = violvec[T_max] ? alt : ref
    invmat = robust_inv(rgm.B + rgm.C * P)
    P_tv[:, :, T_max] = -invmat * rgm.A
    # Shock/constant term at T_max
    shock_contrib = size(shock_path, 1) >= T_max ? shock_path[T_max, :] : zeros(T, n_shocks)
    D_tv[:, T_max] = -invmat * (rgm.C * Q * shock_contrib + rgm.D * shock_contrib)

    # Backward iterate from T_max-1 to 1
    for t in (T_max - 1):-1:1
        rgm_t = violvec[t] ? alt : ref
        P_next = P_tv[:, :, t + 1]
        invmat_t = robust_inv(rgm_t.B + rgm_t.C * P_next)
        P_tv[:, :, t] = -invmat_t * rgm_t.A

        shock_t = size(shock_path, 1) >= t ? shock_path[t, :] : zeros(T, n_shocks)
        D_tv[:, t] = -invmat_t * (rgm_t.C * D_tv[:, t + 1] + rgm_t.D * shock_t)
    end

    # Shock impact for period 1
    rgm_1 = violvec[1] ? alt : ref
    P_1 = T_max >= 2 ? P_tv[:, :, 2] : P
    invmat_1 = robust_inv(rgm_1.B + rgm_1.C * P_1)
    E = -invmat_1 * rgm_1.D

    (P_tv, D_tv, E)
end

# =============================================================================
# Forward Simulation with Time-Varying Rules
# =============================================================================

"""
    _simulate_piecewise(P_tv, D_tv, E, P_lin, init, shock_path, T_max, nperiods) → path

Simulate forward using time-varying decision rules for periods 1..T_max,
then revert to linear rule P_lin for T_max+1..nperiods.
"""
function _simulate_piecewise(P_tv::Array{T,3}, D_tv::Matrix{T}, E::Matrix{T},
                              P_lin::Matrix{T}, init::Vector{T},
                              shock_path::Matrix{T}, T_max::Int,
                              nperiods::Int) where {T}
    n = length(init)
    path = zeros(T, nperiods, n)

    # Period 1: shock impact
    shock_1 = size(shock_path, 1) >= 1 ? shock_path[1, :] : zeros(T, size(E, 2))
    if T_max >= 1
        path[1, :] = P_tv[:, :, 1] * init + D_tv[:, 1] + E * shock_1
    else
        path[1, :] = P_lin * init + E * shock_1
    end

    # Periods 2..T_max: time-varying rules
    for t in 2:min(T_max, nperiods)
        path[t, :] = P_tv[:, :, t] * path[t - 1, :] + D_tv[:, t]
    end

    # Periods T_max+1..nperiods: linear rule
    for t in (T_max + 1):nperiods
        path[t, :] = P_lin * path[t - 1, :]
    end

    path
end

# =============================================================================
# Linear Simulation (unconstrained)
# =============================================================================

"""Simulate the unconstrained linear solution for comparison."""
function _simulate_linear(P::Matrix{T}, Q::Matrix{T}, init::Vector{T},
                           shock_path::Matrix{T}, nperiods::Int) where {T}
    n = size(P, 1)
    path = zeros(T, nperiods, n)

    for t in 1:nperiods
        y_prev = t == 1 ? init : path[t - 1, :]
        shock_t = size(shock_path, 1) >= t ? shock_path[t, :] : zeros(T, size(Q, 2))
        path[t, :] = P * y_prev + Q * shock_t
    end

    path
end

# =============================================================================
# Constraint Evaluation
# =============================================================================

"""Evaluate whether the constraint is violated at each period."""
function _evaluate_constraint(path::Matrix{T}, spec::DSGESpec{T},
                               constraint::OccBinConstraint{T}) where {T}
    var_idx = findfirst(==(constraint.variable), spec.endog)
    nperiods = size(path, 1)
    violations = BitVector(undef, nperiods)

    for t in 1:nperiods
        val = path[t, var_idx]
        if constraint.direction == :geq
            violations[t] = val < constraint.bound
        else  # :leq
            violations[t] = val > constraint.bound
        end
    end

    violations
end

# =============================================================================
# Guess-and-Verify Loop — One Constraint
# =============================================================================

"""
    _guess_verify_one(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                      P, Q, spec, constraint, shock_path, nperiods; maxiter) → (path, regime_history, converged, iterations)

Iterate guess-and-verify for one constraint until convergence.
"""
function _guess_verify_one(ref::OccBinRegime{T}, alt::OccBinRegime{T},
                            P::Matrix{T}, Q::Matrix{T},
                            spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                            shock_path::Matrix{T}, nperiods::Int;
                            maxiter::Int=100) where {T}
    n = spec.n_endog
    init = zeros(T, n)  # deviations from SS start at zero

    # Initial guess: no violations
    violvec = falses(nperiods)
    converged = false
    iter = 0

    local path::Matrix{T}

    while iter < maxiter
        iter += 1

        # Compute time-varying rules given current guess
        P_tv, D_tv, E = _backward_iteration(ref, alt, P, Q, violvec, shock_path)

        T_max = findlast(violvec)
        T_max_val = T_max === nothing ? 0 : T_max

        # Simulate piecewise path
        path = _simulate_piecewise(P_tv, D_tv, E, P, init, shock_path, T_max_val, nperiods)

        # Check constraint violations
        new_violvec = _evaluate_constraint(path, spec, constraint)

        # Check for relaxation in constrained periods
        # If a period was constrained but no longer violates, relax it
        relax = violvec .& .!new_violvec

        # Update: add new violations, remove relaxed
        updated = (violvec .| new_violvec) .& .!relax

        if updated == violvec
            converged = true
            break
        end

        violvec = updated
    end

    if !converged
        @warn "OccBin: did not converge after $maxiter iterations"
    end

    if any(violvec[end])
        @warn "OccBin: constraint still binding at terminal period. Consider increasing nperiods."
    end

    regime_history = reshape(Int.(violvec), nperiods, 1)
    (path, regime_history, converged, iter)
end

# =============================================================================
# Public API — One Constraint
# =============================================================================

"""
    occbin_solve(spec::DSGESpec, constraint::OccBinConstraint;
                 shock_path, nperiods=40, maxiter=100) → OccBinSolution
    occbin_solve(spec::DSGESpec, constraint::OccBinConstraint, alt_spec::DSGESpec;
                 shock_path, nperiods=40, maxiter=100) → OccBinSolution

Solve a DSGE model with one occasionally binding constraint using the
piecewise-linear OccBin method (Guerrieri & Iacoviello 2015).

# Arguments
- `spec` — reference (unconstrained) model specification
- `constraint` — parsed constraint from `parse_constraint`
- `alt_spec` — (optional) explicit alternative regime specification
- `shock_path` — T_periods × n_shocks matrix of shock values
- `nperiods` — simulation horizon (default: 40)
- `maxiter` — max guess-and-verify iterations (default: 100)
"""
function occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T};
                      shock_path::AbstractMatrix{<:Real}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    # Ensure steady state
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end

    # Derive alternative regime
    alt_spec = _derive_alternative_regime(spec, constraint)
    if isempty(alt_spec.steady_state)
        alt_spec = compute_steady_state(alt_spec)
    end

    _occbin_solve_impl(spec, alt_spec, constraint, T.(shock_path), nperiods, maxiter)
end

# With explicit alternative spec
function occbin_solve(spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                      alt_spec::DSGESpec{T};
                      shock_path::AbstractMatrix{<:Real}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100) where {T<:AbstractFloat}
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end
    if isempty(alt_spec.steady_state)
        alt_spec = compute_steady_state(alt_spec)
    end

    _occbin_solve_impl(spec, alt_spec, constraint, T.(shock_path), nperiods, maxiter)
end

function _occbin_solve_impl(spec::DSGESpec{T}, alt_spec::DSGESpec{T},
                             constraint::OccBinConstraint{T},
                             shock_path::Matrix{T}, nperiods::Int,
                             maxiter::Int) where {T}
    # Solve reference model
    sol = solve(spec; method=:gensys)
    is_stable(sol) || @warn "OccBin: reference model is not stable"

    P = sol.G1       # state transition
    Q = sol.impact   # shock impact

    # Extract linearized regimes
    ref_regime = _extract_regime(spec)
    alt_regime = _extract_regime(alt_spec)

    # Linear (unconstrained) path
    init = zeros(T, spec.n_endog)
    linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)

    # Piecewise path via guess-and-verify
    pw_path, regime_history, converged, iterations =
        _guess_verify_one(ref_regime, alt_regime, P, Q, spec, constraint,
                          shock_path, nperiods; maxiter=maxiter)

    OccBinSolution{T}(
        linear_path, pw_path, spec.steady_state, regime_history,
        converged, iterations, spec, [string(s) for s in spec.endog]
    )
end
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: All OccBin tests pass including one-constraint ZLB and no-binding cases

**Step 5: Commit**

```bash
git add src/dsge/occbin.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add OccBin one-constraint solver with guess-and-verify loop"
```

---

### Task 4: Two-Constraint Extension

**Files:**
- Modify: `src/dsge/occbin.jl` (append two-constraint functions)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to Section 15 in `test/dsge/test_dsge.jl`:

```julia
@testset "OccBin: two-constraint" begin
    spec = @dsge begin
        parameters: rho = 0.8, phi = 1.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
        c[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= 0), spec)
    c2 = parse_constraint(:(c[t] >= 0), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -3.0

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=30)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
    @test size(sol.regime_history, 2) == 2  # two constraints

    # Both constraints should bind for at least some periods
    @test sum(sol.regime_history[:, 1]) > 0  # i >= 0
    @test sum(sol.regime_history[:, 2]) > 0  # c >= 0

    # Piecewise should respect both constraints
    i_idx = findfirst(==(:i), spec.endog)
    c_idx = findfirst(==(:c), spec.endog)
    @test minimum(sol.piecewise_path[:, i_idx]) >= -1e-8
    @test minimum(sol.piecewise_path[:, c_idx]) >= -1e-8
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels; m = methods(occbin_solve); println(length(m))'`
Expected: Only 2 methods (one-constraint variants), no two-constraint method

**Step 3: Write minimal implementation**

Append to `src/dsge/occbin.jl`:

```julia
# =============================================================================
# Two-Constraint Extension — 4 Regimes
# =============================================================================

"""
    _guess_verify_two(ref, alt1, alt2, alt12, P, Q, spec, c1, c2,
                      shock_path, nperiods; maxiter, curb_retrench) → (path, regime_history, converged, iterations)

Iterate guess-and-verify for two constraints.
Four regimes: (0,0) ref, (1,0) alt1, (0,1) alt2, (1,1) alt12.
"""
function _guess_verify_two(ref::OccBinRegime{T}, alt1::OccBinRegime{T},
                            alt2::OccBinRegime{T}, alt12::OccBinRegime{T},
                            P::Matrix{T}, Q::Matrix{T},
                            spec::DSGESpec{T},
                            c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                            shock_path::Matrix{T}, nperiods::Int;
                            maxiter::Int=100, curb_retrench::Bool=false) where {T}
    n = spec.n_endog
    init = zeros(T, n)

    violvec = falses(nperiods, 2)
    converged = false
    iter = 0

    local path::Matrix{T}

    while iter < maxiter
        iter += 1

        # Select regime per period based on 2D violvec
        P_tv, D_tv, E = _backward_iteration_two(ref, alt1, alt2, alt12, P, Q,
                                                   violvec, shock_path)

        T_max = _find_last_binding_two(violvec)

        path = _simulate_piecewise(P_tv, D_tv, E, P, init, shock_path, T_max, nperiods)

        # Evaluate both constraints
        new_viol1 = _evaluate_constraint(path, spec, c1)
        new_viol2 = _evaluate_constraint(path, spec, c2)

        # Relaxation
        relax1 = violvec[:, 1] .& .!new_viol1
        relax2 = violvec[:, 2] .& .!new_viol2

        if curb_retrench
            # Only relax one period at a time per constraint
            first_relax1 = findfirst(relax1)
            first_relax2 = findfirst(relax2)
            relax1 = falses(nperiods)
            relax2 = falses(nperiods)
            if first_relax1 !== nothing
                relax1[first_relax1] = true
            end
            if first_relax2 !== nothing
                relax2[first_relax2] = true
            end
        end

        updated1 = (violvec[:, 1] .| new_viol1) .& .!relax1
        updated2 = (violvec[:, 2] .| new_viol2) .& .!relax2

        updated = hcat(updated1, updated2)

        if updated == violvec
            converged = true
            break
        end

        violvec = updated
    end

    if !converged
        @warn "OccBin: two-constraint solver did not converge after $maxiter iterations"
    end

    regime_history = hcat(Int.(violvec[:, 1]), Int.(violvec[:, 2]))
    (path, regime_history, converged, iter)
end

"""Find the last period where either constraint binds."""
function _find_last_binding_two(violvec::BitMatrix)
    T_max = 0
    for t in axes(violvec, 1)
        if violvec[t, 1] || violvec[t, 2]
            T_max = t
        end
    end
    T_max
end

"""
Backward iteration for the two-constraint case.
Selects among 4 regime coefficient sets per period.
"""
function _backward_iteration_two(ref::OccBinRegime{T}, alt1::OccBinRegime{T},
                                  alt2::OccBinRegime{T}, alt12::OccBinRegime{T},
                                  P::Matrix{T}, Q::Matrix{T},
                                  violvec::BitMatrix, shock_path::Matrix{T}) where {T}
    n = size(P, 1)
    nperiods = size(violvec, 1)
    n_shocks = size(Q, 2)

    T_max = _find_last_binding_two(violvec)
    if T_max == 0
        P_tv = Array{T,3}(undef, n, n, 0)
        D_tv = Matrix{T}(undef, n, 0)
        return (P_tv, D_tv, Q)
    end

    P_tv = zeros(T, n, n, T_max)
    D_tv = zeros(T, n, T_max)

    # Select regime at each period
    function _select_regime(t)
        v1 = violvec[t, 1]
        v2 = violvec[t, 2]
        if v1 && v2
            return alt12
        elseif v1
            return alt1
        elseif v2
            return alt2
        else
            return ref
        end
    end

    # At T_max
    rgm = _select_regime(T_max)
    invmat = robust_inv(rgm.B + rgm.C * P)
    P_tv[:, :, T_max] = -invmat * rgm.A
    shock_T = size(shock_path, 1) >= T_max ? shock_path[T_max, :] : zeros(T, n_shocks)
    D_tv[:, T_max] = -invmat * (rgm.C * Q * shock_T + rgm.D * shock_T)

    # Backward iterate
    for t in (T_max - 1):-1:1
        rgm_t = _select_regime(t)
        P_next = P_tv[:, :, t + 1]
        invmat_t = robust_inv(rgm_t.B + rgm_t.C * P_next)
        P_tv[:, :, t] = -invmat_t * rgm_t.A

        shock_t = size(shock_path, 1) >= t ? shock_path[t, :] : zeros(T, n_shocks)
        D_tv[:, t] = -invmat_t * (rgm_t.C * D_tv[:, t + 1] + rgm_t.D * shock_t)
    end

    # Shock impact
    rgm_1 = _select_regime(1)
    P_1 = T_max >= 2 ? P_tv[:, :, 2] : P
    invmat_1 = robust_inv(rgm_1.B + rgm_1.C * P_1)
    E = -invmat_1 * rgm_1.D

    (P_tv, D_tv, E)
end

# =============================================================================
# Public API — Two Constraints
# =============================================================================

"""
    occbin_solve(spec, c1, c2; shock_path, nperiods=40, maxiter=100, curb_retrench=false)
    occbin_solve(spec, c1, c2, alt_specs::Dict; ...)

Solve with two occasionally binding constraints. Creates 4 regimes:
- (0,0): neither binding — reference model
- (1,0): constraint 1 binding only
- (0,1): constraint 2 binding only
- (1,1): both binding
"""
function occbin_solve(spec::DSGESpec{T},
                      c1::OccBinConstraint{T}, c2::OccBinConstraint{T};
                      shock_path::AbstractMatrix{<:Real}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100,
                      curb_retrench::Bool=false) where {T<:AbstractFloat}
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end

    # Derive 3 alternative regimes
    alt1_spec = _derive_alternative_regime(spec, c1)
    alt2_spec = _derive_alternative_regime(spec, c2)
    # Both binding: derive from alt1 with c2 applied
    alt12_spec = _derive_alternative_regime(alt1_spec, c2)

    for s in (alt1_spec, alt2_spec, alt12_spec)
        if isempty(s.steady_state)
            s = compute_steady_state(s)
        end
    end

    _occbin_solve_two_impl(spec, alt1_spec, alt2_spec, alt12_spec,
                            c1, c2, T.(shock_path), nperiods, maxiter, curb_retrench)
end

# With explicit alternative specs
function occbin_solve(spec::DSGESpec{T},
                      c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                      alt_specs::Dict;
                      shock_path::AbstractMatrix{<:Real}=zeros(T, 40, spec.n_exog),
                      nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100,
                      curb_retrench::Bool=false) where {T<:AbstractFloat}
    if isempty(spec.steady_state)
        spec = compute_steady_state(spec)
    end
    alt1 = get(alt_specs, (1,0), _derive_alternative_regime(spec, c1))
    alt2 = get(alt_specs, (0,1), _derive_alternative_regime(spec, c2))
    alt12 = get(alt_specs, (1,1), _derive_alternative_regime(alt1, c2))

    for s in (alt1, alt2, alt12)
        if isempty(s.steady_state)
            s = compute_steady_state(s)
        end
    end

    _occbin_solve_two_impl(spec, alt1, alt2, alt12,
                            c1, c2, T.(shock_path), nperiods, maxiter, curb_retrench)
end

function _occbin_solve_two_impl(spec::DSGESpec{T},
                                 alt1_spec::DSGESpec{T}, alt2_spec::DSGESpec{T},
                                 alt12_spec::DSGESpec{T},
                                 c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                                 shock_path::Matrix{T}, nperiods::Int,
                                 maxiter::Int, curb_retrench::Bool) where {T}
    sol = solve(spec; method=:gensys)
    is_stable(sol) || @warn "OccBin: reference model is not stable"

    P = sol.G1
    Q = sol.impact

    ref = _extract_regime(spec)
    alt1 = _extract_regime(alt1_spec)
    alt2 = _extract_regime(alt2_spec)
    alt12 = _extract_regime(alt12_spec)

    init = zeros(T, spec.n_endog)
    linear_path = _simulate_linear(P, Q, init, shock_path, nperiods)

    pw_path, regime_history, converged, iterations =
        _guess_verify_two(ref, alt1, alt2, alt12, P, Q, spec, c1, c2,
                          shock_path, nperiods; maxiter=maxiter,
                          curb_retrench=curb_retrench)

    OccBinSolution{T}(
        linear_path, pw_path, spec.steady_state, regime_history,
        converged, iterations, spec, [string(s) for s in spec.endog]
    )
end
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: Two-constraint test passes

**Step 5: Commit**

```bash
git add src/dsge/occbin.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add OccBin two-constraint solver with 4-regime logic"
```

---

### Task 5: OccBin IRF

**Files:**
- Modify: `src/dsge/occbin.jl` (append occbin_irf)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to Section 15:

```julia
@testset "OccBin: occbin_irf" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)

    # Large negative shock triggers ZLB
    oirf = occbin_irf(spec, constraint, 1, 30; magnitude=-2.0)
    @test isa(oirf, OccBinIRF{Float64})
    @test size(oirf.linear, 1) == 30
    @test size(oirf.piecewise, 1) == 30
    @test oirf.shock_name == "e"

    # Linear should go negative, piecewise should be clamped
    i_idx = 2
    @test minimum(oirf.linear[:, i_idx]) < 0.0
    @test minimum(oirf.piecewise[:, i_idx]) >= -1e-8

    # Small positive shock — no binding, should be identical
    oirf2 = occbin_irf(spec, constraint, 1, 20; magnitude=0.5)
    @test oirf2.linear ≈ oirf2.piecewise atol=1e-10
end
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels; println(isdefined(MacroEconometricModels, :occbin_irf))'`
Expected: `false`

**Step 3: Write minimal implementation**

Append to `src/dsge/occbin.jl`:

```julia
# =============================================================================
# OccBin IRF
# =============================================================================

"""
    occbin_irf(spec, constraint, shock_idx, horizon; magnitude=1.0, maxiter=100)
    occbin_irf(spec, c1, c2, shock_idx, horizon; magnitude=1.0, maxiter=100)

Compute impulse response functions under occasionally binding constraints.

Returns `OccBinIRF{T}` with both linear (unconstrained) and piecewise (constrained) paths.

# Arguments
- `spec` — model specification
- `constraint` / `c1, c2` — parsed constraint(s)
- `shock_idx` — index of the shock to perturb
- `horizon` — number of periods
- `magnitude` — shock size (default: 1.0 standard deviation)
- `maxiter` — max iterations for guess-and-verify
"""
function occbin_irf(spec::DSGESpec{T}, constraint::OccBinConstraint{T},
                    shock_idx::Int, horizon::Int;
                    magnitude::Real=one(T), maxiter::Int=100) where {T<:AbstractFloat}
    1 <= shock_idx <= spec.n_exog || throw(ArgumentError(
        "shock_idx=$shock_idx out of range 1:$(spec.n_exog)"))

    shock_path = zeros(T, horizon, spec.n_exog)
    shock_path[1, shock_idx] = T(magnitude)

    sol = occbin_solve(spec, constraint; shock_path=shock_path,
                       nperiods=horizon, maxiter=maxiter)

    shock_name = string(spec.exog[shock_idx])
    OccBinIRF{T}(sol.linear_path, sol.piecewise_path, sol.regime_history,
                  sol.varnames, shock_name)
end

# Two-constraint IRF
function occbin_irf(spec::DSGESpec{T}, c1::OccBinConstraint{T}, c2::OccBinConstraint{T},
                    shock_idx::Int, horizon::Int;
                    magnitude::Real=one(T), maxiter::Int=100,
                    curb_retrench::Bool=false) where {T<:AbstractFloat}
    1 <= shock_idx <= spec.n_exog || throw(ArgumentError(
        "shock_idx=$shock_idx out of range 1:$(spec.n_exog)"))

    shock_path = zeros(T, horizon, spec.n_exog)
    shock_path[1, shock_idx] = T(magnitude)

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path,
                       nperiods=horizon, maxiter=maxiter,
                       curb_retrench=curb_retrench)

    shock_name = string(spec.exog[shock_idx])
    OccBinIRF{T}(sol.linear_path, sol.piecewise_path, sol.regime_history,
                  sol.varnames, shock_name)
end
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: OccBin IRF tests pass

**Step 5: Commit**

```bash
git add src/dsge/occbin.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add occbin_irf for IRF under occasionally binding constraints"
```

---

### Task 6: Plotting Integration

**Files:**
- Modify: `src/plotting/models.jl:377` (append plot_result dispatches)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to Section 15:

```julia
@testset "OccBin: plot_result" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)
    oirf = occbin_irf(spec, constraint, 1, 30; magnitude=-2.0)

    p = plot_result(oirf)
    @test isa(p, PlotOutput)
    @test occursin("OccBin", p.html)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=30)
    p2 = plot_result(sol)
    @test isa(p2, PlotOutput)
end
```

**Step 2: Run test to verify it fails**

Expected: MethodError for `plot_result(::OccBinIRF)`

**Step 3: Write minimal implementation**

Append to `src/plotting/models.jl`:

```julia
# =============================================================================
# OccBin — Piecewise-Linear Solution Plots
# =============================================================================

"""
    plot_result(oirf::OccBinIRF; title="", save_path=nothing, ncols=2)

Plot OccBin IRF comparison: linear (dashed) vs piecewise (solid) per variable,
with shaded regions for binding periods.
"""
function plot_result(oirf::OccBinIRF{T};
                     title::String="", save_path::Union{String,Nothing}=nothing,
                     ncols::Int=2) where {T}
    horizon = size(oirf.piecewise, 1)
    n = length(oirf.varnames)
    panels = _PanelSpec[]

    for j in 1:n
        id = _next_plot_id("occbin_irf_$(j)")
        ptitle = oirf.varnames[j]

        # Build data: horizons, linear, piecewise, regime indicator
        rows = String[]
        for h in 1:horizon
            binding = any(oirf.regime_history[h, :] .> 0) ? 1 : 0
            push!(rows, "{h:$(h),lin:$(oirf.linear[h,j]),pw:$(oirf.piecewise[h,j]),bind:$(binding)}")
        end
        data_json = "[" * join(rows, ",") * "]"

        s_json = _series_json(["Linear", "Constrained"], [_PLOT_COLORS[3], _PLOT_COLORS[1]];
                              keys=["lin", "pw"])

        # Custom JS: two lines + shaded binding regions
        js = """
        const svg_$id = d3.select('#$id').append('svg').attr('width',W).attr('height',H);
        const g_$id = svg_$id.append('g').attr('transform',`translate(\${M.l},\${M.t})`);
        const d_$id = $data_json;
        const xS_$id = d3.scaleLinear().domain([1,$horizon]).range([0,W-M.l-M.r]);
        const yE_$id = d3.extent(d_$id, d=>Math.min(d.lin,d.pw)).concat(d3.extent(d_$id, d=>Math.max(d.lin,d.pw)));
        const yS_$id = d3.scaleLinear().domain([d3.min(yE_$id),d3.max(yE_$id)]).nice().range([H-M.t-M.b,0]);
        g_$id.append('g').attr('transform',`translate(0,\${H-M.t-M.b})`).call(d3.axisBottom(xS_$id).ticks(8));
        g_$id.append('g').call(d3.axisLeft(yS_$id).ticks(6));

        // Shaded binding regions
        d_$id.filter(d=>d.bind===1).forEach(d=>{
            g_$id.append('rect').attr('x',xS_$id(d.h)-3).attr('y',0)
                .attr('width',7).attr('height',H-M.t-M.b)
                .attr('fill','#fee2e2').attr('opacity',0.6);
        });

        // Zero reference line
        g_$id.append('line').attr('x1',0).attr('x2',W-M.l-M.r)
            .attr('y1',yS_$id(0)).attr('y2',yS_$id(0))
            .attr('stroke','#999').attr('stroke-dasharray','4,4');

        // Linear (dashed)
        g_$id.append('path').datum(d_$id)
            .attr('d',d3.line().x(d=>xS_$id(d.h)).y(d=>yS_$id(d.lin)))
            .attr('fill','none').attr('stroke','$(_PLOT_COLORS[3])').attr('stroke-width',1.5)
            .attr('stroke-dasharray','6,3');
        // Piecewise (solid)
        g_$id.append('path').datum(d_$id)
            .attr('d',d3.line().x(d=>xS_$id(d.h)).y(d=>yS_$id(d.pw)))
            .attr('fill','none').attr('stroke','$(_PLOT_COLORS[1])').attr('stroke-width',2);

        // Legend
        const lg_$id = g_$id.append('g').attr('transform','translate(10,10)');
        lg_$id.append('line').attr('x1',0).attr('x2',20).attr('y1',0).attr('y2',0)
            .attr('stroke','$(_PLOT_COLORS[3])').attr('stroke-width',1.5).attr('stroke-dasharray','6,3');
        lg_$id.append('text').attr('x',25).attr('y',4).text('Linear').style('font-size','11px');
        lg_$id.append('line').attr('x1',0).attr('x2',20).attr('y1',18).attr('y2',18)
            .attr('stroke','$(_PLOT_COLORS[1])').attr('stroke-width',2);
        lg_$id.append('text').attr('x',25).attr('y',22).text('Constrained').style('font-size','11px');
        """

        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin IRF — Shock: $(oirf.shock_name)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(sol::OccBinSolution; title="", save_path=nothing, ncols=2)

Plot OccBin piecewise solution paths with regime shading.
"""
function plot_result(sol::OccBinSolution{T};
                     title::String="", save_path::Union{String,Nothing}=nothing,
                     ncols::Int=2) where {T}
    nperiods = size(sol.piecewise_path, 1)
    n = length(sol.varnames)
    panels = _PanelSpec[]

    for j in 1:n
        id = _next_plot_id("occbin_sol_$(j)")
        ptitle = sol.varnames[j]

        rows = String[]
        for t in 1:nperiods
            binding = any(sol.regime_history[t, :] .> 0) ? 1 : 0
            push!(rows, "{t:$(t),lin:$(sol.linear_path[t,j]),pw:$(sol.piecewise_path[t,j]),bind:$(binding)}")
        end
        data_json = "[" * join(rows, ",") * "]"

        s_json = _series_json(["Linear", "Constrained"], [_PLOT_COLORS[3], _PLOT_COLORS[1]];
                              keys=["lin", "pw"])

        js = """
        const svg_$id = d3.select('#$id').append('svg').attr('width',W).attr('height',H);
        const g_$id = svg_$id.append('g').attr('transform',`translate(\${M.l},\${M.t})`);
        const d_$id = $data_json;
        const xS_$id = d3.scaleLinear().domain([1,$nperiods]).range([0,W-M.l-M.r]);
        const yE_$id = d3.extent(d_$id, d=>Math.min(d.lin,d.pw)).concat(d3.extent(d_$id, d=>Math.max(d.lin,d.pw)));
        const yS_$id = d3.scaleLinear().domain([d3.min(yE_$id),d3.max(yE_$id)]).nice().range([H-M.t-M.b,0]);
        g_$id.append('g').attr('transform',`translate(0,\${H-M.t-M.b})`).call(d3.axisBottom(xS_$id).ticks(8));
        g_$id.append('g').call(d3.axisLeft(yS_$id).ticks(6));

        d_$id.filter(d=>d.bind===1).forEach(d=>{
            g_$id.append('rect').attr('x',xS_$id(d.t)-3).attr('y',0)
                .attr('width',7).attr('height',H-M.t-M.b)
                .attr('fill','#fee2e2').attr('opacity',0.6);
        });

        g_$id.append('line').attr('x1',0).attr('x2',W-M.l-M.r)
            .attr('y1',yS_$id(0)).attr('y2',yS_$id(0))
            .attr('stroke','#999').attr('stroke-dasharray','4,4');

        g_$id.append('path').datum(d_$id)
            .attr('d',d3.line().x(d=>xS_$id(d.t)).y(d=>yS_$id(d.lin)))
            .attr('fill','none').attr('stroke','$(_PLOT_COLORS[3])').attr('stroke-width',1.5)
            .attr('stroke-dasharray','6,3');
        g_$id.append('path').datum(d_$id)
            .attr('d',d3.line().x(d=>xS_$id(d.t)).y(d=>yS_$id(d.pw)))
            .attr('fill','none').attr('stroke','$(_PLOT_COLORS[1])').attr('stroke-width',2);
        """

        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin Solution ($(sol.converged ? "converged" : "NOT converged"))"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: Plot tests pass

**Step 5: Commit**

```bash
git add src/plotting/models.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add plot_result dispatches for OccBinIRF and OccBinSolution"
```

---

### Task 7: References and Display Integration

**Files:**
- Modify: `src/summary_refs.jl:522` (add reference), `src/summary_refs.jl:698` (add type ref), `src/summary_refs.jl:1018` (add refs dispatch)
- Test: `test/dsge/test_dsge.jl` (add to Section 15)

**Step 1: Write the failing test**

Add to Section 15:

```julia
@testset "OccBin: refs()" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(20, 1)
    shock_path[1, 1] = -1.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)

    r = refs(sol)
    @test occursin("Guerrieri", r)
    @test occursin("Iacoviello", r)
    @test occursin("2015", r)

    # Symbol dispatch
    r2 = refs(:occbin)
    @test occursin("Guerrieri", r2)
end

@testset "OccBin: show/report" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(20, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)

    io = IOBuffer()
    show(io, sol)
    output = String(take!(io))
    @test occursin("OccBin", output)
    @test occursin("Converged", output)

    oirf = occbin_irf(spec, constraint, 1, 20; magnitude=-2.0)
    io2 = IOBuffer()
    show(io2, oirf)
    output2 = String(take!(io2))
    @test occursin("OccBin", output2)
    @test occursin("Shock", output2)
end
```

**Step 2: Run test to verify it fails**

Expected: refs() error for OccBinSolution

**Step 3: Write minimal implementation**

Add to `src/summary_refs.jl` — in `_REFERENCES` dict (before the closing `)` at line 523):

```julia
    :guerrieri_iacoviello2015 => (key=:guerrieri_iacoviello2015,
        authors="Guerrieri, Luca and Iacoviello, Matteo", year=2015,
        title="OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily",
        journal="Journal of Monetary Economics", volume="70", issue="", pages="22--38",
        doi="10.1016/j.jmoneco.2014.08.005", isbn="", publisher="", entry_type=:article),
```

Add to `_TYPE_REFS` dict (after `:euler_gmm` at line 698, before `:fred_md`):

```julia
    :smm => [:ruge_murcia2012, :hansen1982],
    :analytical_gmm => [:hamilton1994, :hansen1982],
    :OccBinSolution => [:guerrieri_iacoviello2015],
    :OccBinIRF => [:guerrieri_iacoviello2015],
    :occbin => [:guerrieri_iacoviello2015],
    :occbin_solve => [:guerrieri_iacoviello2015],
    :occbin_irf => [:guerrieri_iacoviello2015],
```

Add refs dispatches (after line 1018, after `refs(io::IO, ::DSGESpec; kw...)`):

```julia
refs(io::IO, ::OccBinSolution; kw...) = refs(io, _TYPE_REFS[:OccBinSolution]; kw...)
refs(io::IO, ::OccBinIRF; kw...) = refs(io, _TYPE_REFS[:OccBinIRF]; kw...)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: refs() and show/report tests pass

**Step 5: Commit**

```bash
git add src/summary_refs.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add Guerrieri & Iacoviello (2015) references and OccBin display"
```

---

### Task 8: Edge Cases, Convergence, and Full Test Suite

**Files:**
- Test: `test/dsge/test_dsge.jl` (add edge case tests to Section 15)

**Step 1: Write edge case tests**

Add to Section 15:

```julia
@testset "OccBin: explicit alternative spec" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    # Explicit alternative: i[t] = 0 (ZLB)
    alt_spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.0
    end
    alt_spec = compute_steady_state(alt_spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint, alt_spec; shock_path=shock_path, nperiods=30)
    @test sol.converged
    @test minimum(sol.piecewise_path[:, 2]) >= -1e-8
end

@testset "OccBin: <= constraint direction" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, debt
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        debt[t] = 0.8 * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(debt[t] <= 1.0), spec)
    @test constraint.direction == :leq
    @test constraint.bound == 1.0

    shock_path = zeros(20, 1)
    shock_path[1, 1] = 3.0  # positive shock pushes debt above bound

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)
    @test sol.converged
    debt_idx = 2
    @test maximum(sol.piecewise_path[:, debt_idx]) <= 1.0 + 1e-8
end

@testset "OccBin: maxiter warning" begin
    spec = @dsge begin
        parameters: rho = 0.99
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 1.5 * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(5, 1)
    shock_path[1, 1] = -5.0

    # With maxiter=1, should not converge
    sol = @test_warn r"did not converge" occbin_solve(spec, constraint;
        shock_path=shock_path, nperiods=5, maxiter=1)
    @test !sol.converged
end
```

**Step 2: Run the full test suite**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: All ~7700+ tests pass (including all new OccBin tests), 2 broken (pre-existing)

**Step 3: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): add OccBin edge cases — explicit alt spec, <= direction, maxiter warning"
```

---

## Summary

| Task | Component | New Tests | Key Files |
|------|-----------|-----------|-----------|
| 1 | OccBin types | 4 constructor tests | `types.jl` |
| 2 | Constraint parsing + regime derivation | 3 testsets | `occbin.jl`, `MacroEconometricModels.jl` |
| 3 | One-constraint solver | 3 testsets (map_regime, ZLB, no-binding) | `occbin.jl` |
| 4 | Two-constraint extension | 1 testset (4-regime) | `occbin.jl` |
| 5 | OccBin IRF | 1 testset | `occbin.jl` |
| 6 | Plotting | 1 testset | `plotting/models.jl` |
| 7 | References + display | 2 testsets (refs, show) | `summary_refs.jl` |
| 8 | Edge cases + full suite | 3 testsets | `test_dsge.jl` |
