# Klein (2000) DSGE Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the Klein (2000) generalized Schur decomposition solver as a third first-order DSGE solution method, accessible via `solve(spec; method=:klein)`.

**Architecture:** The Klein solver reuses the existing `linearize()` output (Sims canonical form matrices) and applies a QZ decomposition with eigenvalue reordering. It infers the number of predetermined variables from the Gamma1 matrix, checks the Blanchard-Kahn condition, and constructs the same `DSGESolution{T}` type as gensys and BK. All downstream functions (simulate, irf, fevd, estimate_dsge) work unchanged.

**Tech Stack:** Julia `LinearAlgebra` (schur, ordschur), existing DSGE pipeline.

---

### Task 1: Create `klein.jl` with `_count_predetermined` and `klein()` function

**Files:**
- Create: `src/dsge/klein.jl`
- Test: `test/dsge/test_dsge.jl`

**Context:** The Klein solver is structurally similar to the existing `gensys` (src/dsge/gensys.jl:35-137) and `blanchard_kahn` (src/dsge/blanchard_kahn.jl:34-120) implementations. All three use QZ decomposition with eigenvalue reordering. The key difference is that Klein uses the Blanchard-Kahn counting condition (n_stable == n_predetermined) instead of gensys's Π-matrix rank check, and Klein operates on the pencil `(Γ₁, Γ₀)` rather than `(Γ₀, Γ₁)`.

The `LinearDSGE{T}` type (src/dsge/types.jl:122-167) has fields: `Gamma0`, `Gamma1`, `C`, `Psi`, `Pi`, `spec`. The `DSGESolution{T}` type (src/dsge/types.jl:173-228) has fields: `G1`, `impact`, `C_sol`, `eu`, `method`, `eigenvalues`, `spec`, `linear`.

**Step 1: Create `src/dsge/klein.jl`**

Write the following file:

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
Klein (2000) generalized Schur decomposition solver for linear RE models.

Solves: Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t
Returns: y_t = G1 * y_{t-1} + impact * eps_t + C_sol

Uses the QZ decomposition of the pencil (Gamma1, Gamma0) with eigenvalue
reordering and the Blanchard-Kahn counting condition on predetermined variables.

Reference:
Klein, Paul. 2000. "Using the Generalized Schur Form to Solve a Multivariate
Linear Rational Expectations Model." Journal of Economic Dynamics and Control
24 (10): 1405-1423.
"""

"""
    _count_predetermined(ld::LinearDSGE{T}) → Int

Count the number of predetermined (state) variables by detecting non-zero columns
in Gamma1. A variable is predetermined if it appears with a lag (y[t-1]).
"""
function _count_predetermined(ld::LinearDSGE{T}) where {T}
    n = size(ld.Gamma1, 2)
    tol = eps(T) * T(100)
    count(j -> any(x -> abs(x) > tol, @view(ld.Gamma1[:, j])), 1:n)
end

"""
    klein(Gamma0, Gamma1, C, Psi, n_predetermined; div=1.0) → NamedTuple

Solve the linear RE system via the Klein (2000) QZ decomposition method.

The system is in Sims canonical form:
`Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t`

Klein computes the generalized Schur decomposition of the pencil `(Gamma1, Gamma0)`,
reorders eigenvalues so stable roots (|λ| < div) come first, and checks the
Blanchard-Kahn condition: n_stable == n_predetermined.

# Arguments
- `Gamma0` — n × n coefficient on y_t
- `Gamma1` — n × n coefficient on y_{t-1}
- `C` — n × 1 constant vector
- `Psi` — n × n_shocks shock loading matrix
- `n_predetermined` — number of predetermined (state) variables

# Keywords
- `div::Real=1.0` — dividing line for stable vs unstable eigenvalues

# Returns
Named tuple `(G1, impact, C_sol, eu, eigenvalues)` where:
- `G1` — n × n state transition matrix
- `impact` — n × n_shocks impact matrix
- `C_sol` — n × 1 constants
- `eu` — `[existence, uniqueness]`: 1=yes, 0=no
- `eigenvalues` — generalized eigenvalues from QZ decomposition
"""
function klein(Gamma0::AbstractMatrix{T}, Gamma1::AbstractMatrix{T},
               C::AbstractVector{T}, Psi::AbstractMatrix{T},
               n_predetermined::Int;
               div::Real=1.0) where {T<:AbstractFloat}
    n = size(Gamma0, 1)
    n_jump = n - n_predetermined
    eu = [0, 0]

    # QZ decomposition of pencil (Gamma1, Gamma0)
    # Generalized eigenvalues: λ_i = T_ii / S_ii
    # where Q' * Gamma1 * Z = T_mat, Q' * Gamma0 * Z = S_mat
    F = schur(complex(Gamma1), complex(Gamma0))

    # Compute generalized eigenvalue magnitudes
    gev_mag = zeros(n)
    for i in 1:n
        if abs(F.S[i,i]) > eps(T)
            gev_mag[i] = abs(F.T[i,i] / F.S[i,i])
        else
            gev_mag[i] = Inf
        end
    end

    # Reorder: stable eigenvalues (|λ| < div) first
    stable_select = BitVector(gev_mag .< T(div))
    F_ordered = ordschur(F, stable_select)

    n_stable = count(stable_select)
    n_unstable = n - n_stable

    # Compute eigenvalues after reordering
    eigenvalues = Vector{ComplexF64}(undef, n)
    for i in 1:n
        if abs(F_ordered.S[i,i]) > eps(T)
            eigenvalues[i] = F_ordered.T[i,i] / F_ordered.S[i,i]
        else
            eigenvalues[i] = complex(T(Inf))
        end
    end

    # Blanchard-Kahn condition: n_stable must equal n_predetermined
    if n_stable == n_predetermined
        eu = [1, 1]  # existence and uniqueness
    elseif n_stable > n_predetermined
        eu = [1, 0]  # indeterminate (multiple solutions)
    else
        eu = [0, 0]  # no stable solution (explosive)
    end

    # Extract ordered Schur matrices
    S = F_ordered.S
    TT = F_ordered.T
    Z = F_ordered.Z
    Q = F_ordered.Q
    Qp = Q'  # conjugate transpose

    # Build solution matrices
    if n_stable > 0
        Z1 = Z[:, 1:n_stable]
        S11 = S[1:n_stable, 1:n_stable]
        T11 = TT[1:n_stable, 1:n_stable]
        Q1 = Qp[1:n_stable, :]

        # State transition: G1 = Z1 * S11^{-1} * T11 * Z1'
        G1_c = Z1 * (S11 \ T11) * Z1'

        # Impact: impact = Z1 * S11^{-1} * Q1 * Psi
        impact_c = Z1 * (S11 \ (Q1 * complex(Psi)))

        G1 = real(Matrix{T}(G1_c))
        impact = real(Matrix{T}(impact_c))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, size(Psi, 2))
    end

    # Constants: C_sol = (I - G1)^{-1} * C
    C_sol = if norm(C) > eps(T)
        real(Vector{T}((I - complex(G1)) \ complex(C)))
    else
        zeros(T, n)
    end

    (G1=G1, impact=impact, C_sol=C_sol, eu=eu, eigenvalues=eigenvalues)
end
```

**Step 2: Run tests to verify backward compatibility**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All existing tests pass (Klein not yet wired in).

**Step 3: Commit**

```bash
git add src/dsge/klein.jl
git commit -m "feat(dsge): add Klein (2000) solver implementation (#49)"
```

---

### Task 2: Wire Klein into `solve()` dispatcher and include order

**Files:**
- Modify: `src/dsge/gensys.jl:149-170` (solve function)
- Modify: `src/MacroEconometricModels.jl:166-167` (include order)
- Test: `test/dsge/test_dsge.jl`

**Context:** The `solve()` function in `src/dsge/gensys.jl` dispatches on `method::Symbol`. Currently supports `:gensys`, `:blanchard_kahn`, `:perfect_foresight`. We add `:klein`. The include of `klein.jl` goes between `blanchard_kahn.jl` (line 166) and `perfect_foresight.jl` (line 167) in `src/MacroEconometricModels.jl`.

**Step 1: Add `include("dsge/klein.jl")` to MacroEconometricModels.jl**

In `src/MacroEconometricModels.jl`, after line 166 (`include("dsge/blanchard_kahn.jl")`), add:

```julia
include("dsge/klein.jl")
```

**Step 2: Add `:klein` case to `solve()`**

In `src/dsge/gensys.jl`, modify the `solve()` function. Replace the error message in the `else` clause (line 167-169):

```julia
    elseif method == :klein
        ld = linearize(spec)
        n_pre = _count_predetermined(ld)
        result = klein(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, n_pre)
        return DSGESolution{T}(
            result.G1, result.impact, result.C_sol, result.eu,
            :klein, result.eigenvalues, spec, ld
        )
    else
        throw(ArgumentError("method must be :gensys, :blanchard_kahn, :klein, or :perfect_foresight"))
    end
```

Also update the docstring for `solve()` (line 140-148) to mention `:klein`:

```julia
"""
    solve(spec::DSGESpec{T}; method=:gensys, kwargs...) -> DSGESolution or PerfectForesightPath

Solve a DSGE model.

# Methods
- `:gensys` -- Sims (2002) QZ decomposition (default)
- `:blanchard_kahn` -- Blanchard-Kahn (1980) eigenvalue counting
- `:klein` -- Klein (2000) generalized Schur decomposition
- `:perfect_foresight` -- deterministic Newton solver
"""
```

**Step 3: Run tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All existing tests pass.

**Step 4: Commit**

```bash
git add src/dsge/gensys.jl src/MacroEconometricModels.jl
git commit -m "feat(dsge): wire Klein solver into solve() dispatcher (#49)"
```

---

### Task 3: Tests — equivalence, predetermined detection, BK condition, downstream

**Files:**
- Modify: `test/dsge/test_dsge.jl`

**Context:** Tests verify: (1) Klein produces identical G1/impact to gensys/BK for known models, (2) `_count_predetermined` correctly detects state variables, (3) `eu` flags are correct, (4) downstream functions (simulate, irf, fevd) work with Klein solutions, (5) Klein works with augmented models from issue #54.

The test file has a main `@testset "DSGE Models" begin ... end` (or `"DSGE Module"`) block. Add tests before the closing `end`.

**Step 1: Add Klein tests**

Add to `test/dsge/test_dsge.jl` before the closing `end` of the main testset:

```julia
@testset "Klein (2000) Solver (#49)" begin
    @testset "Predetermined variable detection" begin
        # AR(1): y[t] = ρ*y[t-1] + σ*ε[t] — 1 predetermined
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        ld = linearize(spec)
        @test MacroEconometricModels._count_predetermined(ld) == 1

        # Purely forward-looking: x[t] = β*E[t](x[t+1]) + ε[t] — 0 predetermined
        spec2 = @dsge begin
            parameters: β = 0.5, σ = 1.0
            endogenous: x
            exogenous: ε
            x[t] = β * x[t+1] + σ * ε[t]
        end
        spec2 = compute_steady_state(spec2)
        ld2 = linearize(spec2)
        @test MacroEconometricModels._count_predetermined(ld2) == 0

        # NK model: 2 equations, 1 predetermined (y[t-1])
        spec3 = @dsge begin
            parameters: β = 0.99, κ = 0.5, φ_π = 1.5, ρ = 0.8, σ = 0.01
            endogenous: π, y
            exogenous: ε
            π[t] = β * π[t+1] + κ * y[t] + σ * ε[t]
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec3 = compute_steady_state(spec3)
        ld3 = linearize(spec3)
        @test MacroEconometricModels._count_predetermined(ld3) == 1
    end

    @testset "Equivalence with gensys — AR(1)" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test sol_k.method == :klein
        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
        @test sol_k.C_sol ≈ sol_g.C_sol atol=1e-8
    end

    @testset "Equivalence with gensys — forward-looking" begin
        spec = @dsge begin
            parameters: β = 0.5, σ = 1.0
            endogenous: x
            exogenous: ε
            x[t] = β * x[t+1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
    end

    @testset "Equivalence with gensys — NK 3-equation" begin
        spec = @dsge begin
            parameters: β = 0.99, κ = 0.3, φ_π = 1.5, φ_y = 0.125, ρ_v = 0.5, σ_v = 0.25
            endogenous: π, y, i
            exogenous: ε_v
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - (i[t] - π[t+1]) + σ_v * ε_v[t]
            i[t] = φ_π * π[t] + φ_y * y[t] + ρ_v * ε_v[t]
            steady_state = [0.0, 0.0, 0.0]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-6
        @test sol_k.impact ≈ sol_g.impact atol=1e-6
    end

    @testset "BK condition — eu flags" begin
        # Determined model
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)
        @test sol.eu == [1, 1]

        # Explosive model: ρ > 1 with no forward-looking vars
        spec2 = @dsge begin
            parameters: ρ = 1.5, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec2 = compute_steady_state(spec2)
        sol2 = solve(spec2; method=:klein)
        @test sol2.eu[1] == 0  # no stable solution
    end

    @testset "Downstream: simulate, irf, fevd" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)

        # Simulate
        sim = simulate(sol, 100; shock_draws=zeros(100, 1))
        @test size(sim) == (100, 1)
        @test all(sim .≈ 0.0)  # zero shocks → stays at SS

        # IRF
        ir = irf(sol, 20)
        @test length(ir.variables) == 1
        @test ir.variables == ["y"]
        @test abs(ir.values[1, 1, 1] - 1.0) < 0.01  # σ=1 impact

        # FEVD
        fv = fevd(sol, 20)
        @test length(fv.variables) == 1
        @test all(fv.proportions[:, 1, :] .≈ 1.0)  # single shock = 100%
    end

    @testset "Augmented model compatibility (#54)" begin
        spec = @dsge begin
            parameters: a1 = 0.5, a2 = 0.3, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
            steady_state = [0.0, 0.0]
        end
        spec = compute_steady_state(spec)
        @test spec.augmented

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8

        # IRF should show only original variable
        ir = irf(sol_k, 20)
        @test length(ir.variables) == 1
        @test ir.variables == ["y"]
    end

    @testset "Display shows :klein method" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)

        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("klein", output)
    end
end
```

**Step 2: Run DSGE tests**

Run: `julia --project=. -e 'using Test, MacroEconometricModels; import MacroEconometricModels: set_display_backend, PlotOutput; include("test/dsge/test_dsge.jl")'`
Expected: All tests pass including new Klein tests.

**Step 3: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): add Klein solver tests — equivalence, BK condition, downstream (#49)"
```

---

## Execution Notes

**Dependency order:** Task 1 → Task 2 → Task 3 (strictly sequential — each depends on the previous)

**Key risk:** The QZ decomposition of `(Gamma1, Gamma0)` vs `(Gamma0, Gamma1)` — the pencil order matters for eigenvalue interpretation. Klein uses `schur(Gamma1, Gamma0)` where eigenvalues `λ = T_ii/S_ii` represent the system's dynamic roots. This is the reverse of gensys which uses `schur(Gamma0, Gamma1)`. Verify via the AR(1) test: `ρ=0.9` should produce a stable eigenvalue of 0.9.

**Predetermined detection edge case:** For augmented models (issue #54), auxiliary lag variables (`__lag_*`) appear in Gamma1 as predetermined, and news shock auxiliaries (`__news_*`) also appear in Gamma1 via their identity equations `__news_j[t] = __news_{j-1}[t-1]`. This is correct — the augmented system has more predetermined variables, and Klein's BK condition counts them all.

**No changes needed to:** DSGESolution type, simulate, irf, fevd, analytical_moments, estimate_dsge, perfect_foresight, occbin, display (method symbol auto-displays).
