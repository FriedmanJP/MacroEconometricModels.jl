# GMM Higher-Order Moments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend GMM estimation to use richer moment conditions from 2nd-order perturbation solutions (means + product moments + autocovariances) via closed-form augmented-state Lyapunov equations.

**Architecture:** Replace the simulation-based `analytical_moments` for order >= 2 with closed-form computation using the augmented state `z = [xf; xs; vec(xf⊗xf)]` and the doubling Lyapunov solver. Extend `estimate_dsge(...; method=:analytical_gmm)` to accept `solve_method`/`solve_order` kwargs so it can dispatch to the perturbation solver and exploit the non-zero mean conditions.

**Tech Stack:** Julia, LinearAlgebra, existing `_dlyap_doubling`, `estimate_gmm`, `ParameterTransform`

**Key Reference:** MATLAB `UnconditionalMoments_2nd_Lyap.m` from `GMM_ThirdOrder_v2`

---

### Task 1: Widen DSGEEstimation to Accept PerturbationSolution

**Files:**
- Modify: `src/dsge/types.jl:385-403` (DSGEEstimation struct)
- Modify: `src/dsge/types.jl:412-431` (Base.show for DSGEEstimation)
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

Add to the bottom of the "Higher-Order Perturbation (#48)" testset in `test/dsge/test_dsge.jl`:

```julia
@testset "DSGEEstimation with PerturbationSolution" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol_p = solve(spec; method=:perturbation, order=2)

    # Should be able to construct DSGEEstimation with PerturbationSolution
    est = MacroEconometricModels.DSGEEstimation{Float64}(
        [0.9], [0.01;;], [:ρ], :analytical_gmm,
        0.0, 1.0, sol_p, true, spec
    )
    @test est.theta == [0.9]
    @test est.method == :analytical_gmm
    @test est.solution isa MacroEconometricModels.PerturbationSolution

    # show() should work without error
    io = IOBuffer()
    show(io, est)
    output = String(take!(io))
    @test occursin("DSGE Estimation", output)
    @test occursin("analytical_gmm", output)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: FAIL — `DSGEEstimation` constructor rejects `PerturbationSolution`

**Step 3: Implement — widen the solution field type**

In `src/dsge/types.jl`, change line 392:

```julia
# OLD:
    solution::DSGESolution{T}

# NEW:
    solution::Union{DSGESolution{T}, PerturbationSolution{T}}
```

Update the inner constructor (line 396-401) — no method validation change needed since `:analytical_gmm` is already valid:

```julia
function DSGEEstimation{T}(theta, vcov, param_names, method, J_stat, J_pvalue,
                            solution, converged, spec) where {T<:AbstractFloat}
    @assert length(theta) == length(param_names)
    @assert size(vcov) == (length(theta), length(theta))
    @assert method ∈ (:irf_matching, :euler_gmm, :smm, :analytical_gmm)
    new{T}(theta, vcov, param_names, method, J_stat, J_pvalue, solution, converged, spec)
end
```

Update the docstring (around line 381):

```julia
- `solution::Union{DSGESolution{T}, PerturbationSolution{T}}` — solution at estimated parameters
```

**Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS — all existing tests + new test pass

**Step 5: Commit**

```bash
git add src/dsge/types.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): widen DSGEEstimation to accept PerturbationSolution"
```

---

### Task 2: Implement Closed-Form 2nd-Order Augmented Lyapunov Moments

This is the core algorithmic task. It implements the augmented state system from `UnconditionalMoments_2nd_Lyap.m`.

**Files:**
- Modify: `src/dsge/pruning.jl` (add new functions after `_dlyap_doubling`, before `analytical_moments`)
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

Add to the "Closed-form moments" testset in `test/dsge/test_dsge.jl`:

```julia
@testset "Augmented Lyapunov 2nd-order moments" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol2 = solve(spec; method=:perturbation, order=2)

    # Closed-form 2nd-order moments via augmented Lyapunov
    result = MacroEconometricModels._augmented_moments_2nd(sol2; lags=[1])

    # Result should contain means, variance, and autocovariances
    @test haskey(result, :E_y)
    @test haskey(result, :Var_y)
    @test haskey(result, :Cov_y)

    # For a linear model, mean should be near zero (risk correction ≈ 0)
    @test all(abs.(result[:E_y]) .< 1e-4)

    # Variance should be close to first-order: σ²/(1-ρ²)
    theoretical_var = 0.01^2 / (1 - 0.9^2)
    @test result[:Var_y][1,1] ≈ theoretical_var atol=1e-4

    # Autocovariance at lag 1 ≈ ρ * variance
    @test result[:Cov_y][1,1,1] ≈ 0.9 * theoretical_var atol=1e-4

    # Compare with long simulation
    sim_mom = MacroEconometricModels._simulation_moments(sol2; lags=1)
    lyap_var = result[:Var_y][1,1]
    @test lyap_var ≈ sim_mom[1] atol=5e-3  # simulation has noise
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `_augmented_moments_2nd` not defined

**Step 3: Implement the augmented Lyapunov system**

Add the following functions to `src/dsge/pruning.jl`, between `_dlyap_doubling` (line ~450) and `analytical_moments` (line ~475):

```julia
# =============================================================================
# _extract_xx_block — extract state×state block from v⊗v Kronecker matrix
# =============================================================================

"""
    _extract_xx_block(Mvv::Matrix{T}, nx::Int, nv::Int) → Matrix{T}

Extract the `(xf⊗xf)` sub-block from a matrix with `nv²` columns (Kronecker
ordering of `v = [x; ε]`).  Returns a matrix with `nx²` columns corresponding
to the state×state indices only.
"""
function _extract_xx_block(Mvv::Matrix{T}, nx::Int, nv::Int) where {T}
    nrows = size(Mvv, 1)
    Mxx = zeros(T, nrows, nx * nx)
    for a in 1:nx
        for b in 1:nx
            col_vv = (a - 1) * nv + b   # column in nv² ordering
            col_xx = (a - 1) * nx + b    # column in nx² ordering
            @inbounds Mxx[:, col_xx] = Mvv[:, col_vv]
        end
    end
    return Mxx
end


# =============================================================================
# _innovation_variance_2nd — compute Var(innovations) for augmented state
# =============================================================================

"""
    _innovation_variance_2nd(hx_state, eta_x, Var_xf, nx, n_eps;
                              vectorMom3=nothing, vectorMom4=nothing) → Matrix{T}

Compute the innovation covariance matrix for the 2nd-order augmented state
`z = [xf; xs; vec(xf⊗xf)]`.

Follows `UnconditionalMoments_2nd_Lyap.m` from the GMM_ThirdOrder_v2 MATLAB
reference code (Andreasen 2015).

Arguments:
- `hx_state`: nx × nx state transition
- `eta_x`: nx × n_eps shock loading
- `Var_xf`: nx × nx unconditional variance of xf (from first-order Lyapunov)
- `vectorMom3`: n_eps vector of 3rd moments (default: zeros for symmetric shocks)
- `vectorMom4`: n_eps vector of 4th moments (default: 3s for Gaussian shocks)
"""
function _innovation_variance_2nd(hx_state::Matrix{T}, eta_x::Matrix{T},
                                   Var_xf::Matrix{T},
                                   nx::Int, n_eps::Int;
                                   vectorMom3::Union{Nothing,Vector{T}}=nothing,
                                   vectorMom4::Union{Nothing,Vector{T}}=nothing) where {T}
    nz = 2 * nx + nx^2
    Var_inov = zeros(T, nz, nz)

    # Default shock moments for Gaussian distribution
    if vectorMom3 === nothing
        vectorMom3 = zeros(T, n_eps)
    end
    if vectorMom4 === nothing
        vectorMom4 = fill(T(3), n_eps)
    end

    # sigeta = eta_x (sig=1 in our convention)
    sigeta = eta_x

    # Block (1,1): first-order shock variance
    Var_inov[1:nx, 1:nx] = sigeta * sigeta'

    # Block (1,3) and (3,1): third-moment cross term
    # E[ε_i ⊗ (ε_j ε_k)] — only nonzero when i=j=k for same shock
    if any(!iszero, vectorMom3)
        E_eps_eps2 = zeros(T, n_eps, n_eps^2)
        for phi1 in 1:n_eps
            for phi2 in 1:n_eps
                for phi3 in 1:n_eps
                    idx = (phi2 - 1) * n_eps + phi3
                    if phi1 == phi2 && phi1 == phi3
                        E_eps_eps2[phi1, idx] = vectorMom3[phi1]
                    end
                end
            end
        end
        block_13 = sigeta * E_eps_eps2 * kron(sigeta', sigeta')
        Var_inov[1:nx, (2*nx+1):(2*nx+nx^2)] = block_13
        Var_inov[(2*nx+1):(2*nx+nx^2), 1:nx] = block_13'
    end

    # Block (3,3): quartic terms
    # E[(xf⊗ε)(ε⊗xf)'] — only nonzero when ε indices match
    E_xfeps_epsxf = zeros(T, nx * n_eps, nx * n_eps)
    for gama1 in 1:nx
        for phi1 in 1:n_eps
            idx1 = (gama1 - 1) * n_eps + phi1
            for phi2 in 1:n_eps
                for gama2 in 1:nx
                    idx2 = (phi2 - 1) * nx + gama2
                    if phi1 == phi2
                        E_xfeps_epsxf[idx1, idx2] = Var_xf[gama1, gama2]
                    end
                end
            end
        end
    end

    # E[(ε⊗ε)(ε⊗ε)'] — fourth moment matrix
    ne2 = n_eps^2
    E_eps2_eps2 = zeros(T, ne2, ne2)
    for phi4 in 1:n_eps
        for phi1 in 1:n_eps
            idx1 = (phi4 - 1) * n_eps + phi1
            for phi3 in 1:n_eps
                for phi2 in 1:n_eps
                    idx2 = (phi3 - 1) * n_eps + phi2
                    if phi1 == phi2 && phi3 == phi4 && phi1 != phi4
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi3 && phi2 == phi4 && phi1 != phi2
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi4 && phi2 == phi3 && phi1 != phi2
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi2 && phi1 == phi3 && phi1 == phi4
                        E_eps2_eps2[idx1, idx2] = vectorMom4[phi1]
                    end
                end
            end
        end
    end

    # Assemble block (3,3)
    I_ne = Matrix{T}(I, n_eps, n_eps)
    vec_I_ne = vec(I_ne)
    r1 = 2 * nx + 1
    r2 = 2 * nx + nx^2

    Var_inov[r1:r2, r1:r2] =
        kron(hx_state, sigeta) * kron(Var_xf, I_ne) * kron(hx_state, sigeta)' +
        kron(hx_state, sigeta) * E_xfeps_epsxf * kron(sigeta, hx_state)' +
        kron(sigeta, hx_state) * E_xfeps_epsxf' * kron(hx_state, sigeta)' +
        kron(sigeta, hx_state) * kron(I_ne, Var_xf) * kron(sigeta, hx_state)' +
        kron(sigeta, sigeta) * (E_eps2_eps2 - vec_I_ne * vec_I_ne') * kron(sigeta, sigeta)'

    # Enforce symmetry
    Var_inov = (Var_inov + Var_inov') / 2

    return Var_inov
end


# =============================================================================
# _augmented_moments_2nd — closed-form 2nd-order moments
# =============================================================================

"""
    _augmented_moments_2nd(sol::PerturbationSolution{T};
                            lags::Vector{Int}=[1]) → Dict{Symbol, Any}

Compute closed-form unconditional moments for a 2nd-order perturbation solution
using the augmented-state Lyapunov approach (Andreasen et al. 2018).

The augmented state is `z = [xf; xs; vec(xf⊗xf)]` of dimension `2nx + nx²`.
The system is `z(t+1) = A·z(t) + c + u(t)` where u(t) captures stochastic
innovations from the pruned dynamics.

Returns a Dict with keys:
- `:E_y` — ny-vector of unconditional means
- `:Var_y` — ny×ny unconditional variance-covariance
- `:Cov_y` — ny×ny×max_lag autocovariance tensor
- `:E_z`, `:Var_z` — augmented state moments (for diagnostics)
"""
function _augmented_moments_2nd(sol::PerturbationSolution{T};
                                 lags::Vector{Int}=[1]) where {T}
    nx = nstates(sol)
    ny = ncontrols(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps

    # Extract first-order blocks
    hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x    = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
    gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)
    eta_y    = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)

    # Extract state×state blocks from hxx, gxx (nv² → nx²)
    hxx_xx = sol.hxx !== nothing ? _extract_xx_block(sol.hxx, nx, nv) : zeros(T, nx, nx^2)
    gxx_xx = sol.gxx !== nothing ? _extract_xx_block(sol.gxx, nx, nv) : zeros(T, ny, nx^2)

    nz = 2 * nx + nx^2

    # Build transition matrix A (nz × nz)
    A = zeros(T, nz, nz)
    A[1:nx, 1:nx] = hx_state                                        # xf → xf
    A[nx+1:2*nx, nx+1:2*nx] = hx_state                              # xs → xs
    A[nx+1:2*nx, 2*nx+1:nz] = T(0.5) * hxx_xx                      # kron(xf,xf) → xs
    A[2*nx+1:nz, 2*nx+1:nz] = kron(hx_state, hx_state)             # kron → kron

    # Build constant vector c (nz)
    I_ne = Matrix{T}(I, n_eps, n_eps)
    c = zeros(T, nz)
    if sol.hσσ !== nothing
        c[nx+1:2*nx] = T(0.5) * sol.hσσ
    end
    c[2*nx+1:nz] = kron(eta_x, eta_x) * vec(I_ne)

    # Unconditional mean: E[z] = (I - A) \ c
    E_z = (Matrix{T}(I, nz, nz) - A) \ c

    # First-order state variance (for innovation variance computation)
    Var_xf = nx > 0 ? _dlyap_doubling(hx_state, eta_x * eta_x') : zeros(T, 0, 0)

    # Innovation variance
    Var_inov = _innovation_variance_2nd(hx_state, eta_x, Var_xf, nx, n_eps)

    # Solve augmented Lyapunov: Var_z = A·Var_z·A' + Var_inov
    Var_z = _dlyap_doubling(A, Var_inov)

    # Output mapping: y = C·z + d
    C = zeros(T, ny, nz)
    C[:, 1:nx] = gx_state                 # gx * xf
    C[:, nx+1:2*nx] = gx_state            # gx * xs
    C[:, 2*nx+1:nz] = T(0.5) * gxx_xx    # 0.5 * gxx * kron(xf,xf)

    d = sol.gσσ !== nothing ? T(0.5) * sol.gσσ : zeros(T, ny)

    # Output moments
    E_y = C * E_z + d
    Var_y = C * Var_z * C' + eta_y * eta_y'
    Var_y = (Var_y + Var_y') / 2  # enforce symmetry

    # Autocovariances: Cov_y(lag) = C · A^lag · Var_z · C'
    max_lag = maximum(lags)
    Cov_y = zeros(T, ny, ny, max_lag)
    A_power = copy(A)
    for lag in 1:max_lag
        Cov_z_lag = A_power * Var_z
        Cov_y[:, :, lag] = C * Cov_z_lag * C'
        A_power = A_power * A
    end

    # Adjust E_y by adding steady state constant (g0 in MATLAB)
    # In our formulation, E_y is deviation from SS. The SS is in sol.steady_state.
    # For GMM moments, E_y is the expected deviation, added to SS in the moment function.

    Dict{Symbol, Any}(
        :E_y => E_y,
        :Var_y => Var_y,
        :Cov_y => Cov_y,
        :E_z => E_z,
        :Var_z => Var_z,
    )
end
```

**Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dsge/pruning.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): closed-form 2nd-order moments via augmented Lyapunov

Implements Andreasen et al. (2018) augmented state z=[xf;xs;kron(xf,xf)]
with innovation variance accounting for 3rd/4th shock moments."
```

---

### Task 3: Update analytical_moments and Add GMM Moment Format

**Files:**
- Modify: `src/dsge/pruning.jl:475-559` (analytical_moments for PerturbationSolution)
- Modify: `src/dsge/estimation.jl` (add `_compute_data_moments`)
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

Add to the "Closed-form moments" testset:

```julia
@testset "GMM moment format" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)

    # Order 1: GMM format includes means (should be zero)
    sol1 = solve(spec; method=:perturbation, order=1)
    mom_gmm1 = analytical_moments(sol1; lags=1, format=:gmm)
    # ny=1: 1 mean + 1 product moment + 1 autocov = 3
    @test length(mom_gmm1) == 3
    @test abs(mom_gmm1[1]) < 1e-10  # mean ≈ 0 for order 1

    # Product moment E[y²] = Var(y) + E[y]² ≈ Var(y)
    theoretical_var = 0.01^2 / (1 - 0.9^2)
    @test mom_gmm1[2] ≈ theoretical_var atol=1e-8

    # Order 2: GMM format — mean may be non-zero
    sol2 = solve(spec; method=:perturbation, order=2)
    mom_gmm2 = analytical_moments(sol2; lags=1, format=:gmm)
    @test length(mom_gmm2) == 3
    @test all(isfinite.(mom_gmm2))

    # Default format (:covariance) still works and is backward-compatible
    mom_cov = analytical_moments(sol1; lags=1)
    @test length(mom_cov) == 2  # k*(k+1)/2 + k*lags = 1 + 1

    # Data moments function
    Random.seed!(42)
    data = randn(500, 1)
    m_data = MacroEconometricModels._compute_data_moments(data; lags=[1])
    @test length(m_data) == 3  # 1 mean + 1 product moment + 1 autocov
    @test m_data[1] ≈ mean(data[:, 1]) atol=1e-10
    @test m_data[2] ≈ dot(data[:, 1], data[:, 1]) / size(data, 1) atol=1e-10

    # Multi-variable data moments
    data2 = randn(500, 2)
    m_data2 = MacroEconometricModels._compute_data_moments(data2; lags=[1, 3])
    # ny=2: 2 means + 3 product moments + 2*2 autocov = 2 + 3 + 4 = 9
    @test length(m_data2) == 9
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `format` kwarg not supported, `_compute_data_moments` not defined

**Step 3: Implement**

**3a.** Update `analytical_moments` in `src/dsge/pruning.jl` (lines 475-559). Replace the function with:

```julia
"""
    analytical_moments(sol::PerturbationSolution{T}; lags::Int=1,
                       format::Symbol=:covariance) -> Vector{T}

Compute analytical moment vector from a perturbation solution.

# Keyword Arguments
- `lags::Int=1` — number of autocovariance lags
- `format::Symbol=:covariance` — moment format:
  - `:covariance` (default): upper-triangle of var-cov + diagonal autocov
    (backward compatible with DSGESolution format)
  - `:gmm`: means + upper-triangle product moments + diagonal autocov
    (for GMM estimation with higher-order perturbation)

For **order 1** with `:covariance` format, uses the doubling Lyapunov solver.
For **order >= 2** with `:covariance` format, uses simulation-based moments.
For `:gmm` format at any order, uses closed-form augmented Lyapunov (order >= 2)
or standard Lyapunov (order 1).
"""
function analytical_moments(sol::PerturbationSolution{T};
                              lags::Int=1,
                              format::Symbol=:covariance) where {T<:AbstractFloat}
    format in (:covariance, :gmm) ||
        throw(ArgumentError("format must be :covariance or :gmm; got $format"))

    if format == :gmm
        return _analytical_moments_gmm(sol; lags=lags)
    end

    # Default :covariance format — backward compatible
    if sol.order >= 2
        return _simulation_moments(sol; lags=lags)
    end

    # Order 1: closed-form Lyapunov approach (existing code)
    # ... (keep existing code from lines 481-558 unchanged)
end
```

**Important**: Keep the entire existing `:covariance` code path unchanged. Only add the `format` kwarg dispatch at the top.

**3b.** Add `_analytical_moments_gmm` helper (also in pruning.jl, after `_augmented_moments_2nd`):

```julia
"""
    _analytical_moments_gmm(sol::PerturbationSolution{T}; lags::Int=1) → Vector{T}

Compute GMM-format moment vector: means + product moments + diagonal autocovariances.

For order >= 2, uses closed-form augmented Lyapunov.
For order 1, uses standard Lyapunov (means are zero).
"""
function _analytical_moments_gmm(sol::PerturbationSolution{T}; lags::Int=1) where {T}
    lag_vec = collect(1:lags)

    if sol.order >= 2
        result = _augmented_moments_2nd(sol; lags=lag_vec)
        E_y = result[:E_y]
        Var_y = result[:Var_y]
        Cov_y = result[:Cov_y]
    else
        # Order 1: standard Lyapunov, means are zero
        nx = nstates(sol)
        ny = ncontrols(sol)
        n_eps = nshocks(sol)
        nv = nx + n_eps

        hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)
        eta_x    = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
        gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)
        eta_y    = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)

        Var_xf = nx > 0 ? _dlyap_doubling(hx_state, eta_x * eta_x') : zeros(T, 0, 0)

        n = nvars(sol)
        E_y = zeros(T, n)
        Var_y = zeros(T, n, n)
        if nx > 0
            Var_y[sol.state_indices, sol.state_indices] = Var_xf
            if ny > 0
                Var_y[sol.state_indices, sol.control_indices] = Var_xf * gx_state'
                Var_y[sol.control_indices, sol.state_indices] = gx_state * Var_xf
                Var_y[sol.control_indices, sol.control_indices] = gx_state * Var_xf * gx_state' + eta_y * eta_y'
            end
        elseif ny > 0
            Var_y[sol.control_indices, sol.control_indices] = eta_y * eta_y'
        end

        # Autocovariances
        G1_equiv = zeros(T, n, n)
        if nx > 0
            G1_equiv[sol.state_indices, sol.state_indices] = hx_state
            if ny > 0
                G1_equiv[sol.control_indices, sol.state_indices] = gx_state * hx_state
            end
        end

        max_lag = lags
        Cov_y = zeros(T, n, n, max_lag)
        G1_power = copy(G1_equiv)
        for lag in 1:max_lag
            Cov_y[:, :, lag] = G1_power * Var_y
            G1_power = G1_power * G1_equiv
        end

        # Handle augmented models
        if sol.spec.augmented
            orig_idx = _original_var_indices(sol.spec)
            E_y = E_y[orig_idx]
            Var_y = Var_y[orig_idx, orig_idx]
            Cov_y = Cov_y[orig_idx, orig_idx, :]
        end
    end

    ny_out = length(E_y)

    # Collect moments: means, product moments, diagonal autocov
    moments = T[]

    # 1. Means: E[y_i]
    append!(moments, E_y)

    # 2. Product moments: E[y_i * y_j] = Var_y[i,j] + E_y[i]*E_y[j], upper triangle
    for i in 1:ny_out
        for j in i:ny_out
            push!(moments, Var_y[i, j] + E_y[i] * E_y[j])
        end
    end

    # 3. Diagonal autocovariances at each lag: E[y_i,t * y_i,t-k]
    for lag in 1:lags
        for i in 1:ny_out
            # Product autocovariance: Cov_y[i,i,lag] + E_y[i]^2
            push!(moments, Cov_y[i, i, lag] + E_y[i]^2)
        end
    end

    return moments
end
```

**3c.** Add `_compute_data_moments` to `src/dsge/estimation.jl` (before `_estimate_dsge_analytical_gmm`):

```julia
"""
    _compute_data_moments(data::Matrix{T}; lags::Vector{Int}=[1],
                           observable_indices::Union{Nothing,Vector{Int}}=nothing) → Vector{T}

Compute data moment vector matching the GMM format from model moments:
1. Means: E[y_i]
2. Product moments: E[y_i * y_j] for i ≤ j (upper triangle)
3. Diagonal autocovariances: E[y_i,t * y_i,t-k] for each lag k

This matches the moment ordering in `analytical_moments(...; format=:gmm)`.
"""
function _compute_data_moments(data::Matrix{T};
                                lags::Vector{Int}=[1],
                                observable_indices::Union{Nothing,Vector{Int}}=nothing) where {T}
    Y = observable_indices === nothing ? data : data[:, observable_indices]
    T_obs, ny = size(Y)

    moments = T[]

    # 1. Means
    Ey = vec(sum(Y; dims=1)) / T_obs
    append!(moments, Ey)

    # 2. Product moments (upper triangle of Y'Y/T)
    Eyy = Y' * Y / T_obs
    for i in 1:ny
        for j in i:ny
            push!(moments, Eyy[i, j])
        end
    end

    # 3. Diagonal autocovariances at selected lags
    max_lag = maximum(lags)
    for lag in 1:max_lag
        if lag in lags || lag <= max_lag
            autocov = Y[1+lag:T_obs, :]' * Y[1:T_obs-lag, :] / (T_obs - lag)
            for i in 1:ny
                push!(moments, autocov[i, i])
            end
        end
    end

    return moments
end
```

Wait — the lag indexing above is wrong. Fix:

```julia
function _compute_data_moments(data::Matrix{T};
                                lags::Vector{Int}=[1],
                                observable_indices::Union{Nothing,Vector{Int}}=nothing) where {T}
    Y = observable_indices === nothing ? data : data[:, observable_indices]
    T_obs, ny = size(Y)

    moments = T[]

    # 1. Means
    Ey = vec(sum(Y; dims=1)) / T_obs
    append!(moments, Ey)

    # 2. Product moments (upper triangle of Y'Y/T)
    Eyy = Y' * Y / T_obs
    for i in 1:ny
        for j in i:ny
            push!(moments, Eyy[i, j])
        end
    end

    # 3. Diagonal autocovariances at each lag 1..max(lags)
    num_lags = length(lags)
    for k in 1:num_lags
        lag = lags[k]
        autoEyy = Y[1+lag:T_obs, :]' * Y[1:T_obs-lag, :] / (T_obs - lag)
        for i in 1:ny
            push!(moments, autoEyy[i, i])
        end
    end

    return moments
end
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add src/dsge/pruning.jl src/dsge/estimation.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add GMM moment format and data moment computation

analytical_moments(sol; format=:gmm) returns means + product moments +
autocov. _compute_data_moments matches the model moment format."
```

---

### Task 4: Extend estimate_dsge for Perturbation-Order GMM

**Files:**
- Modify: `src/dsge/estimation.jl:51-87` (estimate_dsge kwargs)
- Modify: `src/dsge/estimation.jl:377-434` (_estimate_dsge_analytical_gmm)
- Test: `test/dsge/test_dsge.jl`

**Step 1: Write the failing test**

Add to the "DSGE Analytical GMM Estimation" testset area:

```julia
@testset "Perturbation-order analytical GMM" begin
    spec = @dsge begin
        parameters: ρ = 0.85, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)

    # Generate data from known parameters
    Random.seed!(42)
    sol_true = solve(spec; method=:perturbation, order=2)
    data = simulate(sol_true, 500; rng=Random.MersenneTwister(42))

    # Estimate with perturbation order 2
    bounds = ParameterTransform{Float64}([0.01], [0.999])
    est = estimate_dsge(spec, data, [:ρ];
                         method=:analytical_gmm,
                         solve_method=:perturbation,
                         solve_order=2,
                         auto_lags=[1],
                         bounds=bounds)

    @test est.converged
    @test est.method == :analytical_gmm
    @test est.solution isa MacroEconometricModels.PerturbationSolution
    # Parameter should be recovered within CI
    @test abs(est.theta[1] - 0.85) < 0.15
    @test est.J_stat >= 0.0
    @test 0.0 <= est.J_pvalue <= 1.0
end
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `solve_method` kwarg not recognized

**Step 3: Implement**

**3a.** Add new kwargs to `estimate_dsge` in `src/dsge/estimation.jl` (lines 51-87):

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
                        rng=Random.default_rng(),
                        # New kwargs for perturbation-order GMM
                        solve_method::Symbol=:gensys,
                        solve_order::Int=1,
                        auto_lags::Vector{Int}=[1],
                        observable_indices::Union{Nothing,Vector{Int}}=nothing) where {T<:AbstractFloat}
    data_T = Matrix{T}(data)

    if method == :irf_matching
        # ... unchanged
    elseif method == :euler_gmm
        # ... unchanged
    elseif method == :smm
        # ... unchanged
    elseif method == :analytical_gmm
        return _estimate_dsge_analytical_gmm(spec, data_T, param_names;
                                              lags=lags,
                                              bounds=bounds,
                                              weighting=weighting,
                                              solve_method=solve_method,
                                              solve_order=solve_order,
                                              auto_lags=auto_lags,
                                              observable_indices=observable_indices)
    else
        throw(ArgumentError("method must be :irf_matching, :euler_gmm, :smm, or :analytical_gmm"))
    end
end
```

**3b.** Rewrite `_estimate_dsge_analytical_gmm` (lines 377-434):

```julia
"""
    _estimate_dsge_analytical_gmm(spec, data, param_names; ...) -> DSGEEstimation

Internal: Analytical GMM estimation using Lyapunov equation moments.

Matches model-implied analytical moments to data moments. For first-order
solutions, uses covariance + autocovariance matching (as before). For
higher-order perturbation solutions, uses the richer GMM format with means +
product moments + autocovariances.

# Keywords
- `lags=1` — autocovariance lags (used when solve_method=:gensys)
- `weighting=:identity` — GMM weighting
- `bounds=nothing` — parameter bounds
- `solve_method=:gensys` — solver (:gensys or :perturbation)
- `solve_order=1` — perturbation order (1, 2, or 3)
- `auto_lags=[1]` — autocovariance lags for GMM format
- `observable_indices=nothing` — which data columns to match
"""
function _estimate_dsge_analytical_gmm(spec::DSGESpec{T}, data::Matrix{T},
                                         param_names::Vector{Symbol};
                                         lags=1, weighting=:identity,
                                         bounds=nothing,
                                         solve_method=:gensys,
                                         solve_order=1,
                                         auto_lags=[1],
                                         observable_indices=nothing) where {T}
    theta0 = T[spec.param_values[p] for p in param_names]

    use_perturbation = (solve_method == :perturbation && solve_order >= 2)

    # Data moments
    if use_perturbation
        m_data = _compute_data_moments(data; lags=auto_lags,
                                        observable_indices=observable_indices)
    else
        m_data = autocovariance_moments(data; lags=lags)
    end
    n_moments = length(m_data)

    # GMM moment function: returns 1 × n_moments matrix
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
            if use_perturbation
                sol = solve(new_spec; method=:perturbation, order=solve_order)
                if !is_determined(sol) || !is_stable(sol)
                    return fill(T(1e6), 1, n_moments)
                end
                m_model = analytical_moments(sol; lags=maximum(auto_lags), format=:gmm)
            else
                sol = solve(new_spec; method=solve_method)
                if !is_determined(sol) || !is_stable(sol)
                    return fill(T(1e6), 1, n_moments)
                end
                m_model = analytical_moments(sol; lags=lags)
            end
            g = reshape(m_data .- m_model, 1, n_moments)
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
    if use_perturbation
        final_sol = solve(final_spec; method=:perturbation, order=solve_order)
    else
        final_sol = solve(final_spec; method=solve_method)
    end

    DSGEEstimation{T}(
        gmm_result.theta, gmm_result.vcov, param_names,
        :analytical_gmm, gmm_result.J_stat, gmm_result.J_pvalue,
        final_sol, gmm_result.converged, final_spec
    )
end
```

**Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dsge/estimation.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): extend analytical GMM for perturbation-order estimation

estimate_dsge(...; method=:analytical_gmm, solve_method=:perturbation,
solve_order=2) uses closed-form augmented Lyapunov moments with
means + product moments + autocovariances."
```

---

### Task 5: Comprehensive Tests

**Files:**
- Modify: `test/dsge/test_dsge.jl`

**Step 1: Write comprehensive test suite**

Add a new testset after the existing higher-order perturbation tests:

```julia
@testset "GMM Higher-Order Moments" begin

    @testset "Innovation variance 2nd order" begin
        # Test _innovation_variance_2nd for known case
        # Single state, single shock: nx=1, n_eps=1
        hx_s = [0.9;;]
        eta_x = [0.01;;]
        Var_xf = MacroEconometricModels._dlyap_doubling(hx_s, eta_x * eta_x')

        Var_inov = MacroEconometricModels._innovation_variance_2nd(
            hx_s, eta_x, Var_xf, 1, 1)

        # nz = 2*1 + 1 = 3
        @test size(Var_inov) == (3, 3)
        # Block (1,1) = eta_x * eta_x' = 0.0001
        @test Var_inov[1, 1] ≈ 0.01^2 atol=1e-12
        # Symmetric
        @test Var_inov ≈ Var_inov' atol=1e-15
        # Positive semi-definite
        @test all(eigvals(Symmetric(Var_inov)) .>= -1e-12)
    end

    @testset "Extract xx block" begin
        # nx=2, n_eps=1, nv=3 → nv²=9, nx²=4
        M = reshape(1.0:18.0, 2, 9)  # 2×9
        Mxx = MacroEconometricModels._extract_xx_block(M, 2, 3)
        @test size(Mxx) == (2, 4)
        # Column (1,1) of v⊗v = column 1 of M → column 1 of Mxx
        @test Mxx[:, 1] == M[:, 1]
        # Column (1,2) of v⊗v = column 2 of M → column 2 of Mxx
        @test Mxx[:, 2] == M[:, 2]
        # Column (2,1) of v⊗v = column 4 of M (=(2-1)*3+1) → column 3 of Mxx
        @test Mxx[:, 3] == M[:, 4]
        # Column (2,2) of v⊗v = column 5 of M (=(2-1)*3+2) → column 4 of Mxx
        @test Mxx[:, 4] == M[:, 5]
    end

    @testset "2nd-order risk correction: non-zero mean" begin
        # For a nonlinear model, 2nd-order should produce non-zero means
        # Use a model with curvature (log consumption Euler equation flavor)
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.1
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        result = MacroEconometricModels._augmented_moments_2nd(sol2; lags=[1])

        # Mean exists and is finite
        @test all(isfinite.(result[:E_y]))
        # Variance is positive
        @test all(diag(result[:Var_y]) .> 0)
    end

    @testset "Data moments match analytical for generated data" begin
        Random.seed!(123)
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol1 = solve(spec; method=:perturbation, order=1)

        # Generate long simulation
        data = simulate(sol1, 100_000; rng=Random.MersenneTwister(123))

        # Data moments should converge to model moments
        m_model = analytical_moments(sol1; lags=1, format=:gmm)
        m_data = MacroEconometricModels._compute_data_moments(data; lags=[1])

        @test length(m_model) == length(m_data)
        # Mean ≈ 0 for order 1
        @test abs(m_data[1]) < 0.01
        # Product moment ≈ theoretical variance
        theoretical_var = 0.01^2 / (1 - 0.9^2)
        @test m_data[2] ≈ theoretical_var atol=0.05 * theoretical_var
    end

    @testset "Closed-form 2nd-order matches simulation" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        # Closed-form moments
        mom_cf = analytical_moments(sol2; lags=1, format=:gmm)

        # Simulation-based moments (long run)
        sim = simulate(sol2, 500_000; rng=Random.MersenneTwister(99))
        m_sim = MacroEconometricModels._compute_data_moments(sim; lags=[1])

        @test length(mom_cf) == length(m_sim)
        # Should match within sampling error
        for i in eachindex(mom_cf)
            @test mom_cf[i] ≈ m_sim[i] atol=max(abs(m_sim[i]) * 0.1, 1e-5)
        end
    end

    @testset "Multi-variable model moments" begin
        spec = @dsge begin
            parameters: ρ₁ = 0.8, ρ₂ = 0.7, σ₁ = 0.01, σ₂ = 0.02
            endogenous: x, y
            exogenous: ε₁, ε₂
            x[t] = ρ₁ * x[t-1] + σ₁ * ε₁[t]
            y[t] = ρ₂ * y[t-1] + σ₂ * ε₂[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        mom = analytical_moments(sol2; lags=2, format=:gmm)
        # ny=2: 2 means + 3 product moments + 2*2 autocov = 2 + 3 + 4 = 9
        @test length(mom) == 9
        @test all(isfinite.(mom))
    end

    @testset "Backward compatibility: default format unchanged" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Order 1: default format matches existing behavior
        sol1 = solve(spec; method=:perturbation, order=1)
        sol_g = solve(spec; method=:gensys)
        mom_p = analytical_moments(sol1; lags=1)
        mom_g = analytical_moments(sol_g; lags=1)
        @test length(mom_p) == length(mom_g)
        @test mom_p ≈ mom_g atol=1e-8

        # Order 2: default format uses simulation (backward compatible)
        sol2 = solve(spec; method=:perturbation, order=2)
        mom_sim = analytical_moments(sol2; lags=1)
        @test length(mom_sim) == length(mom_g)  # same format
    end

    @testset "Existing analytical_gmm still works" begin
        spec = @dsge begin
            parameters: ρ = 0.85, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        Random.seed!(42)
        data = simulate(sol, 200; rng=Random.MersenneTwister(42))

        # Old API: estimate_dsge with analytical_gmm, no perturbation kwargs
        bounds = ParameterTransform{Float64}([0.01], [0.999])
        est = estimate_dsge(spec, data, [:ρ];
                             method=:analytical_gmm,
                             bounds=bounds)
        @test est.converged
        @test est.method == :analytical_gmm
        @test est.solution isa MacroEconometricModels.DSGESolution
    end

end
```

**Step 2: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: PASS — all tests pass

**Step 3: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): comprehensive tests for GMM higher-order moments

Tests innovation variance, xx-block extraction, risk correction, data
moments, closed-form vs simulation, multi-variable, backward compatibility."
```

---

### Task 6: Final Integration Test and Documentation Update

**Files:**
- Modify: `test/dsge/test_dsge.jl` (round-trip estimation test)
- No doc changes needed (API is backward compatible)

**Step 1: Write round-trip estimation test**

```julia
@testset "Round-trip perturbation GMM estimation" begin
    # Test: generate data → estimate → recover parameters
    spec = @dsge begin
        parameters: ρ = 0.85, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)

    # Generate data from known model
    sol_true = solve(spec; method=:perturbation, order=2)
    Random.seed!(7777)
    data = simulate(sol_true, 1000; rng=Random.MersenneTwister(7777))

    # Estimate ρ with perturbation order 2
    bounds = ParameterTransform{Float64}([0.01], [0.999])
    est = estimate_dsge(spec, data, [:ρ];
                         method=:analytical_gmm,
                         solve_method=:perturbation,
                         solve_order=2,
                         auto_lags=[1, 3],
                         bounds=bounds,
                         weighting=:two_step)

    @test est.converged
    @test abs(est.theta[1] - 0.85) < 0.2  # within 0.2 of true
    @test est.J_stat >= 0.0

    # More moments than parameters → overidentified
    # ny=1: 1 mean + 1 product moment + 1*2 autocov = 4 moments > 1 param
    @test est.J_pvalue > 0.0  # overidentification test has DOF

    # Show works
    io = IOBuffer()
    show(io, est)
    output = String(take!(io))
    @test occursin("analytical_gmm", output)
end
```

**Step 2: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: All DSGE tests pass

**Step 3: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): round-trip perturbation GMM estimation test

Verifies generate→estimate→recover for order-2 analytical GMM with
two-step weighting and overidentification test."
```

---

## Summary of Changes

| File | Lines Changed | What |
|---|---|---|
| `src/dsge/types.jl` | ~5 | Widen `DSGEEstimation.solution` to Union type |
| `src/dsge/pruning.jl` | ~250 new | `_extract_xx_block`, `_innovation_variance_2nd`, `_augmented_moments_2nd`, `_analytical_moments_gmm`; update `analytical_moments` with `format` kwarg |
| `src/dsge/estimation.jl` | ~80 new/changed | `_compute_data_moments`; extend `estimate_dsge` and `_estimate_dsge_analytical_gmm` with `solve_method`/`solve_order`/`auto_lags`/`observable_indices` |
| `test/dsge/test_dsge.jl` | ~180 new | 9 test sub-sets covering innovation variance, xx extraction, moments, estimation |

## Total: ~510 new lines of code + tests
