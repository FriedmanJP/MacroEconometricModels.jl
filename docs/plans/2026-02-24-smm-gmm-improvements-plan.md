# SMM + GMM Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Simulated Method of Moments (SMM) estimation and improve GMM with Optim.jl optimization and parameter transforms.

**Architecture:** Three new files (`src/gmm/transforms.jl`, `src/gmm/smm.jl`, `test/gmm/test_smm.jl`) plus modifications to `src/gmm/gmm.jl` (optimizer upgrade), `src/dsge/estimation.jl` (`:smm` method), `src/dsge/types.jl` (allow `:smm`), `src/MacroEconometricModels.jl` (includes/exports), `src/summary_refs.jl` (references), `src/summary.jl` (report dispatch).

**Tech Stack:** Julia, Optim.jl (already imported), StatsAPI, Distributions, LinearAlgebra

**Design doc:** `docs/plans/2026-02-24-smm-gmm-improvements-design.md`

---

### Task 1: Parameter Transforms

Create `src/gmm/transforms.jl` with `ParameterTransform`, `to_unconstrained`, `to_constrained`, `transform_jacobian`.

**Files:**
- Create: `src/gmm/transforms.jl`
- Modify: `src/MacroEconometricModels.jl:169` (add include after gmm.jl)
- Modify: `src/MacroEconometricModels.jl:525-529` (add exports)
- Test: `test/gmm/test_smm.jl` (new file, first testset)

**Step 1: Write the failing test**

Create `test/gmm/test_smm.jl`:

```julia
# (GPL-3.0 header)

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

@testset "SMM Estimation" begin

@testset "Parameter Transforms" begin
    @testset "ParameterTransform construction" begin
        pt = ParameterTransform([0.0, -Inf, 0.0], [1.0, Inf, Inf])
        @test pt.lower == [0.0, -Inf, 0.0]
        @test pt.upper == [1.0, Inf, Inf]
    end

    @testset "Identity transform (unbounded)" begin
        pt = ParameterTransform([-Inf], [Inf])
        @test to_unconstrained(pt, [2.5]) ≈ [2.5]
        @test to_constrained(pt, [2.5]) ≈ [2.5]
    end

    @testset "Exp/log transform (lower bounded)" begin
        pt = ParameterTransform([0.0], [Inf])
        theta = [2.0]
        phi = to_unconstrained(pt, theta)
        @test phi ≈ [log(2.0)]
        @test to_constrained(pt, phi) ≈ theta
    end

    @testset "Negative exp transform (upper bounded)" begin
        pt = ParameterTransform([-Inf], [0.0])
        theta = [-3.0]
        phi = to_unconstrained(pt, theta)
        @test to_constrained(pt, phi) ≈ theta atol=1e-10
    end

    @testset "Logistic transform (bounded interval)" begin
        pt = ParameterTransform([0.0], [1.0])
        theta = [0.5]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
        # Boundary behavior
        @test to_constrained(pt, [-100.0])[1] > 0.0
        @test to_constrained(pt, [100.0])[1] < 1.0
    end

    @testset "Round-trip multiple parameters" begin
        pt = ParameterTransform([0.0, -1.0, 0.0, -Inf], [1.0, 1.0, Inf, Inf])
        theta = [0.3, 0.0, 2.5, -1.0]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
    end

    @testset "Jacobian diagonal" begin
        pt = ParameterTransform([0.0, -Inf], [1.0, Inf])
        phi = [0.0, 3.0]
        J = transform_jacobian(pt, phi)
        @test size(J) == (2, 2)
        @test J[1, 2] == 0.0  # diagonal
        @test J[2, 1] == 0.0
        @test J[1, 1] > 0.0   # positive for logistic
        @test J[2, 2] == 1.0  # identity for unbounded
    end
end

end  # outer testset
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: FAIL with "UndefVarError: ParameterTransform not defined"

**Step 3: Write minimal implementation**

Create `src/gmm/transforms.jl`:

```julia
# (GPL-3.0 header)

"""
Parameter transforms for constrained optimization via bijective maps
between constrained (model) and unconstrained (optimizer) spaces.
"""

"""
    ParameterTransform{T<:AbstractFloat}

Bijective parameter transform specification for constrained optimization.

Transform rules per element:
- `(-Inf, Inf)` → identity
- `(0, Inf)` → exp/log
- `(-Inf, 0)` → -exp/-log
- `(a, b)` → logistic: `a + (b-a) / (1 + exp(-x))`

# Fields
- `lower::Vector{T}` — lower bounds (-Inf = unbounded below)
- `upper::Vector{T}` — upper bounds (Inf = unbounded above)
"""
struct ParameterTransform{T<:AbstractFloat}
    lower::Vector{T}
    upper::Vector{T}

    function ParameterTransform{T}(lower::Vector{T}, upper::Vector{T}) where {T<:AbstractFloat}
        @assert length(lower) == length(upper) "lower and upper must have same length"
        for i in eachindex(lower)
            @assert lower[i] < upper[i] || (isinf(lower[i]) && isinf(upper[i])) "lower[$i] must be < upper[$i]"
        end
        new{T}(lower, upper)
    end
end

ParameterTransform(lower::Vector{T}, upper::Vector{T}) where {T<:AbstractFloat} =
    ParameterTransform{T}(lower, upper)
ParameterTransform(lower::Vector{<:Real}, upper::Vector{<:Real}) =
    ParameterTransform(Float64.(lower), Float64.(upper))

"""
    to_unconstrained(pt::ParameterTransform, theta::AbstractVector) -> Vector

Map parameters from constrained (model) space to unconstrained (optimizer) space.
"""
function to_unconstrained(pt::ParameterTransform{T}, theta::AbstractVector) where {T}
    phi = similar(theta, T)
    for i in eachindex(theta)
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            # Identity
            phi[i] = theta[i]
        elseif isinf(hi) && lo == zero(T)
            # (0, Inf) → log
            phi[i] = log(theta[i])
        elseif isinf(hi) && isfinite(lo)
            # (a, Inf) → log(theta - a)
            phi[i] = log(theta[i] - lo)
        elseif isinf(lo) && hi == zero(T)
            # (-Inf, 0) → log(-theta)
            phi[i] = log(-theta[i])
        elseif isinf(lo) && isfinite(hi)
            # (-Inf, b) → log(b - theta)
            phi[i] = log(hi - theta[i])
        else
            # (a, b) → logit: log((theta - a) / (b - theta))
            phi[i] = log((theta[i] - lo) / (hi - theta[i]))
        end
    end
    phi
end

"""
    to_constrained(pt::ParameterTransform, phi::AbstractVector) -> Vector

Map parameters from unconstrained (optimizer) space to constrained (model) space.
"""
function to_constrained(pt::ParameterTransform{T}, phi::AbstractVector) where {T}
    theta = similar(phi, T)
    for i in eachindex(phi)
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            theta[i] = phi[i]
        elseif isinf(hi) && lo == zero(T)
            theta[i] = exp(phi[i])
        elseif isinf(hi) && isfinite(lo)
            theta[i] = lo + exp(phi[i])
        elseif isinf(lo) && hi == zero(T)
            theta[i] = -exp(phi[i])
        elseif isinf(lo) && isfinite(hi)
            theta[i] = hi - exp(phi[i])
        else
            theta[i] = lo + (hi - lo) / (one(T) + exp(-phi[i]))
        end
    end
    theta
end

"""
    transform_jacobian(pt::ParameterTransform, phi::AbstractVector) -> Matrix

Diagonal Jacobian ∂θ/∂φ of the inverse transform (unconstrained → constrained).
Used for delta method SE correction.
"""
function transform_jacobian(pt::ParameterTransform{T}, phi::AbstractVector) where {T}
    n = length(phi)
    J = zeros(T, n, n)
    for i in 1:n
        lo, hi = pt.lower[i], pt.upper[i]
        if isinf(lo) && isinf(hi)
            J[i, i] = one(T)
        elseif isinf(hi) && lo == zero(T)
            J[i, i] = exp(phi[i])
        elseif isinf(hi) && isfinite(lo)
            J[i, i] = exp(phi[i])
        elseif isinf(lo) && hi == zero(T)
            J[i, i] = -exp(phi[i])
        elseif isinf(lo) && isfinite(hi)
            J[i, i] = -exp(phi[i])
        else
            e = exp(-phi[i])
            J[i, i] = (hi - lo) * e / (one(T) + e)^2
        end
    end
    J
end
```

**Step 4: Add include and exports to MacroEconometricModels.jl**

In `src/MacroEconometricModels.jl`, after line 169 (`include("gmm/gmm.jl")`), add:
```julia
include("gmm/transforms.jl")
```

In the GMM exports section (after line 529), add:
```julia
export ParameterTransform, to_unconstrained, to_constrained
```

**Step 5: Run test to verify it passes**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: All 7 testsets PASS

**Step 6: Commit**

```bash
git add src/gmm/transforms.jl test/gmm/test_smm.jl src/MacroEconometricModels.jl
git commit -m "feat(gmm): add ParameterTransform for constrained optimization"
```

---

### Task 2: GMM Optimizer Upgrade

Replace the hand-rolled BFGS in `minimize_gmm` with `Optim.optimize` (LBFGS + NelderMead fallback). Internal-only change — all existing tests must continue to pass.

**Files:**
- Modify: `src/gmm/gmm.jl:312-381` (replace `minimize_gmm` body)
- Modify: `src/gmm/gmm.jl:409-524` (add `bounds` kwarg to `estimate_gmm`)
- Test: `test/gmm/test_gmm.jl` (existing tests must pass)
- Test: `test/gmm/test_smm.jl` (add GMM+bounds test)

**Step 1: Write the failing test for bounded GMM**

Append to `test/gmm/test_smm.jl` (inside the outer `@testset "SMM Estimation"` block, after the Parameter Transforms block):

```julia
@testset "GMM with Parameter Transforms" begin
    # Simple mean estimation with bounds
    rng = Random.MersenneTwister(42)
    true_mu = 0.7  # true parameter in (0, 1)
    data = true_mu .+ 0.1 .* randn(rng, 200, 1)

    function mean_moments(theta, data)
        data .- theta[1]  # E[x - mu] = 0
    end

    bounds = ParameterTransform([0.0], [1.0])
    result = estimate_gmm(mean_moments, [0.5], data;
                          weighting=:identity, bounds=bounds)
    @test result.converged
    @test abs(result.theta[1] - true_mu) < 0.05
    @test result.theta[1] > 0.0
    @test result.theta[1] < 1.0
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: FAIL — `estimate_gmm` does not accept `bounds` kwarg

**Step 3: Replace minimize_gmm internals**

Replace `minimize_gmm` function body at `src/gmm/gmm.jl:312-381` with:

```julia
function minimize_gmm(moment_fn::Function, theta0::AbstractVector{T}, data,
                      W::AbstractMatrix{T}; max_iter::Int=100, tol::T=T(1e-8)) where {T<:AbstractFloat}
    obj(t) = gmm_objective(t, moment_fn, data, W)

    # Primary: LBFGS
    result = Optim.optimize(obj, theta0, Optim.LBFGS(),
                            Optim.Options(iterations=max_iter, f_reltol=tol))

    if !Optim.converged(result)
        # Fallback: NelderMead
        result_nm = Optim.optimize(obj, theta0, Optim.NelderMead(),
                                    Optim.Options(iterations=max_iter * 10, f_reltol=tol))
        if Optim.minimum(result_nm) < Optim.minimum(result)
            result = result_nm
        end
    end

    (theta=Optim.minimizer(result), objective=Optim.minimum(result),
     converged=Optim.converged(result), iterations=Optim.iterations(result))
end
```

**Step 4: Add bounds kwarg to estimate_gmm**

Modify the `estimate_gmm` signature at line 409 to add `bounds::Union{Nothing,ParameterTransform}=nothing`:

```julia
function estimate_gmm(moment_fn::Function, theta0::AbstractVector{T}, data;
                      weighting::Symbol=:two_step, max_iter::Int=100,
                      tol::T=T(1e-8), hac::Bool=true, bandwidth::Int=0,
                      bounds::Union{Nothing,ParameterTransform}=nothing) where {T<:AbstractFloat}
```

At the top of `estimate_gmm` body, after `n_params = length(theta0)`, add transform logic:

```julia
    # Set up parameter transform
    has_bounds = bounds !== nothing
    if has_bounds
        phi0 = to_unconstrained(bounds, theta0)
        # Wrap moment function to work in unconstrained space
        function transformed_moment_fn(phi, data)
            moment_fn(to_constrained(bounds, phi), data)
        end
        working_fn = transformed_moment_fn
        working_theta0 = phi0
    else
        working_fn = moment_fn
        working_theta0 = theta0
    end
```

Then replace every occurrence of `moment_fn` with `working_fn` and `theta0` with `working_theta0` in the estimation body (the 4 weighting branches). After computing `theta_hat`, map back:

```julia
    # Map back to constrained space
    if has_bounds
        phi_hat = theta_hat
        theta_hat = to_constrained(bounds, phi_hat)
    end
```

For the vcov computation with bounds, apply delta method after computing the unconstrained vcov:

```julia
    # Delta method SE correction for transforms
    if has_bounds
        J_transform = transform_jacobian(bounds, phi_hat)
        vcov = J_transform * vcov * J_transform'
    end
```

**Step 5: Run all existing GMM tests plus new bounds test**

Run: `julia --project=. -e 'include("test/gmm/test_gmm.jl"); include("test/gmm/test_smm.jl")'`
Expected: All existing GMM tests PASS + new bounds test PASSES

**Step 6: Commit**

```bash
git add src/gmm/gmm.jl test/gmm/test_smm.jl
git commit -m "feat(gmm): replace hand-rolled BFGS with Optim.jl + add bounds support"
```

---

### Task 3: SMMModel Type + autocovariance_moments

Add `SMMModel{T} <: AbstractGMMModel` struct and `autocovariance_moments` helper.

**Files:**
- Create: `src/gmm/smm.jl`
- Modify: `src/MacroEconometricModels.jl` (include after transforms.jl, add exports)
- Test: `test/gmm/test_smm.jl` (add testsets)

**Step 1: Write the failing tests**

Append inside the outer `@testset "SMM Estimation"` block:

```julia
@testset "autocovariance_moments" begin
    rng = Random.MersenneTwister(123)
    data = randn(rng, 500, 2)
    m = autocovariance_moments(data; lags=1)
    # k=2, lags=1: k*(k+1)/2 + k*lags = 3 + 2 = 5 moments
    @test length(m) == 5
    # First elements are upper-triangle of variance-covariance
    @test m[1] ≈ var(data[:, 1]) atol=0.01  # var(y1) — close but not exact due to N vs N-1
end

@testset "SMMModel construction and interface" begin
    # Build a minimal SMMModel manually
    theta = [0.5, 0.3]
    vcov = [0.01 0.0; 0.0 0.02]
    n_moments = 5
    W = Matrix{Float64}(I, n_moments, n_moments)
    g_bar = zeros(n_moments)
    weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)

    smm = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov, n_moments, 2, 200, weighting, W, g_bar,
        0.5, 0.48, true, 10, 5
    )

    @test coef(smm) == theta
    @test nobs(smm) == 200
    @test MacroEconometricModels.stderror(smm) ≈ sqrt.(diag(vcov))
    @test smm.sim_ratio == 5

    # show should not error
    io = IOBuffer()
    show(io, smm)
    str = String(take!(io))
    @test occursin("SMM", str)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: FAIL with "UndefVarError: autocovariance_moments not defined"

**Step 3: Write implementation**

Create `src/gmm/smm.jl`:

```julia
# (GPL-3.0 header)

"""
Simulated Method of Moments (SMM) estimation.

References:
- Ruge-Murcia, F. (2012). "Estimating Nonlinear DSGE Models by the Simulated Method
  of Moments." Journal of Economic Dynamics and Control, 36(6), 914-938.
- Lee, B.-S. & Ingram, B. F. (1991). "Simulation Estimation of Time-Series Models."
  Journal of Econometrics, 47(2-3), 197-205.
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# SMMModel Type
# =============================================================================

"""
    SMMModel{T} <: AbstractGMMModel

Simulated Method of Moments estimator.

Shares the `AbstractGMMModel` interface with `GMMModel` — `coef`, `vcov`, `nobs`,
`stderror`, `show`, `refs`, `report`, `j_test` all work.

# Fields
- `theta::Vector{T}` — estimated parameters
- `vcov::Matrix{T}` — asymptotic covariance matrix
- `n_moments::Int` — number of moment conditions
- `n_params::Int` — number of parameters
- `n_obs::Int` — number of data observations
- `weighting::GMMWeighting{T}` — weighting specification
- `W::Matrix{T}` — final weighting matrix
- `g_bar::Vector{T}` — moment discrepancy at solution
- `J_stat::T` — Hansen J-test statistic
- `J_pvalue::T` — J-test p-value
- `converged::Bool` — convergence flag
- `iterations::Int` — optimizer iterations
- `sim_ratio::Int` — τ = simulation periods / data periods
"""
struct SMMModel{T<:AbstractFloat} <: AbstractGMMModel
    theta::Vector{T}
    vcov::Matrix{T}
    n_moments::Int
    n_params::Int
    n_obs::Int
    weighting::GMMWeighting{T}
    W::Matrix{T}
    g_bar::Vector{T}
    J_stat::T
    J_pvalue::T
    converged::Bool
    iterations::Int
    sim_ratio::Int

    function SMMModel{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
                          g_bar, J_stat, J_pvalue, converged, iterations,
                          sim_ratio) where {T<:AbstractFloat}
        @assert length(theta) == n_params
        @assert size(vcov) == (n_params, n_params)
        @assert size(W) == (n_moments, n_moments)
        @assert length(g_bar) == n_moments
        @assert n_moments >= n_params "SMM requires at least as many moments as parameters"
        @assert sim_ratio >= 1
        new{T}(theta, vcov, n_moments, n_params, n_obs, weighting, W,
               g_bar, J_stat, J_pvalue, converged, iterations, sim_ratio)
    end
end

# StatsAPI interface
StatsAPI.coef(m::SMMModel) = m.theta
StatsAPI.vcov(m::SMMModel) = m.vcov
StatsAPI.nobs(m::SMMModel) = m.n_obs
StatsAPI.dof(m::SMMModel) = m.n_params
StatsAPI.islinear(::SMMModel) = false
StatsAPI.stderror(m::SMMModel) = sqrt.(max.(diag(m.vcov), zero(eltype(m.theta))))

is_overidentified(m::SMMModel) = m.n_moments > m.n_params
overid_df(m::SMMModel) = m.n_moments - m.n_params

function Base.show(io::IO, m::SMMModel{T}) where {T}
    spec = Any[
        "Parameters"   m.n_params;
        "Moments"      m.n_moments;
        "Observations"  m.n_obs;
        "Sim ratio (τ)" m.sim_ratio;
        "Weighting"    string(m.weighting.method);
        "Converged"    m.converged ? "Yes" : "No";
        "Iterations"   m.iterations
    ]
    _pretty_table(io, spec;
        title = "SMM Estimation Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    se = stderror(m)
    param_names = ["θ[$i]" for i in 1:m.n_params]
    _coef_table(io, "Coefficients", param_names, m.theta, se; dist=:z)
    if is_overidentified(m)
        j_data = Any[
            "J-statistic" _fmt(m.J_stat);
            "P-value"     _format_pvalue(m.J_pvalue);
            "DF"          overid_df(m)
        ]
        _pretty_table(io, j_data;
            title = "Hansen J-test",
            column_labels = ["", ""],
            alignment = [:l, :r],
        )
    end
end

function StatsAPI.confint(m::SMMModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    z = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.theta .- z .* se, m.theta .+ z .* se)
end

# j_test reuse: GMMModel's j_test works via duck-typing on fields
function j_test(m::SMMModel{T}) where {T}
    df = overid_df(m)
    if df <= 0
        return (J_stat=zero(T), p_value=one(T), df=0, reject_05=false,
                message="Model is just-identified, J-test not applicable")
    end
    (J_stat=m.J_stat, p_value=m.J_pvalue, df=df, reject_05=m.J_pvalue < T(0.05))
end

# =============================================================================
# Moment Functions
# =============================================================================

"""
    autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) -> Vector{T}

Compute standard DSGE moment vector from data matrix.

Returns: `[upper-triangle variance-covariance elements; diagonal autocovariances at each lag]`

For k variables, lags L: `k*(k+1)/2 + k*L` moments total.

# Arguments
- `data` — T_obs × k data matrix
- `lags` — number of autocovariance lags (default: 1)
"""
function autocovariance_moments(data::AbstractMatrix{T}; lags::Int=1) where {T<:AbstractFloat}
    n, k = size(data)
    means = vec(mean(data, dims=1))
    data_c = data .- means'

    moments = T[]

    # Upper triangle of variance-covariance matrix
    for i in 1:k
        for j in i:k
            push!(moments, dot(data_c[:, i], data_c[:, j]) / n)
        end
    end

    # Diagonal autocovariances at each lag
    for lag in 1:lags
        for i in 1:k
            acov = dot(data_c[(lag+1):n, i], data_c[1:(n-lag), i]) / n
            push!(moments, acov)
        end
    end

    moments
end

autocovariance_moments(data::AbstractMatrix{<:Real}; kwargs...) =
    autocovariance_moments(Float64.(data); kwargs...)

"""
    smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                          hac::Bool=true, bandwidth::Int=0) -> Matrix{T}

Compute optimal SMM weighting matrix from data moment contributions.
Centers the per-observation moment contributions and applies HAC with Bartlett kernel.

# Arguments
- `data` — T_obs × k data matrix
- `moments_fn` — function computing moment vector from data
- `hac` — use HAC correction (default: true)
- `bandwidth` — HAC bandwidth, 0 = automatic: `floor(4*(n/100)^(2/9))`
"""
function smm_weighting_matrix(data::AbstractMatrix{T}, moments_fn::Function;
                               hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    n = size(data, 1)
    m_full = moments_fn(data)
    q = length(m_full)

    # Compute per-observation moment contributions
    G = Matrix{T}(undef, n, q)
    for t in 1:n
        G[t, :] = moments_fn(data[t:t, :])
    end
    G_demean = G .- mean(G, dims=1)

    if hac
        Omega = long_run_covariance(G_demean; bandwidth=bandwidth, kernel=:bartlett)
    else
        Omega = (G_demean' * G_demean) / n
    end

    Omega_sym = Hermitian((Omega + Omega') / 2)
    eigvals_O = eigvals(Omega_sym)
    if minimum(eigvals_O) < eps(T)
        Omega_reg = Omega_sym + T(1e-8) * I
        return inv(Omega_reg)
    end

    robust_inv(Matrix(Omega_sym))
end
```

**Step 4: Add include and exports**

In `src/MacroEconometricModels.jl`, after `include("gmm/transforms.jl")`, add:
```julia
include("gmm/smm.jl")
```

In the GMM exports section, add:
```julia
export SMMModel, estimate_smm, autocovariance_moments
```

**Step 5: Run tests**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: All testsets PASS including autocovariance_moments and SMMModel construction

**Step 6: Commit**

```bash
git add src/gmm/smm.jl src/MacroEconometricModels.jl test/gmm/test_smm.jl
git commit -m "feat(gmm): add SMMModel type and autocovariance_moments"
```

---

### Task 4: estimate_smm Implementation

Implement the main `estimate_smm` function with two-step estimation, SE correction, and J-test.

**Files:**
- Modify: `src/gmm/smm.jl` (add `estimate_smm` function)
- Test: `test/gmm/test_smm.jl` (add estimation tests)

**Step 1: Write the failing tests**

Append inside the outer testset in `test/gmm/test_smm.jl`:

```julia
@testset "estimate_smm — AR(1) recovery" begin
    # True AR(1): y_t = rho * y_{t-1} + sigma * e_t
    rng = Random.MersenneTwister(42)
    true_rho = 0.8
    true_sigma = 0.5
    T_obs = 500

    # Generate data
    y = zeros(T_obs)
    for t in 2:T_obs
        y[t] = true_rho * y[t-1] + true_sigma * randn(rng)
    end
    data = reshape(y, :, 1)

    # Simulator
    function sim_ar1(theta, T_periods, burn; rng=Random.default_rng())
        rho, sigma = theta
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn)
            sim[t] = rho * sim[t-1] + sigma * randn(rng)
        end
        reshape(sim[(burn+1):end], :, 1)
    end

    # Moments: variance + first autocov
    function my_moments(d)
        autocovariance_moments(d; lags=1)
    end

    result = estimate_smm(sim_ar1, my_moments, [0.5, 0.3], data;
                          sim_ratio=5, burn=100, weighting=:two_step,
                          rng=Random.MersenneTwister(123))

    @test result isa SMMModel{Float64}
    @test result.converged
    @test abs(result.theta[1] - true_rho) < 0.15  # rho recovery
    @test abs(result.theta[2] - true_sigma) < 0.15  # sigma recovery
    @test result.sim_ratio == 5
    @test length(stderror(result)) == 2
    @test all(stderror(result) .> 0)
end

@testset "estimate_smm with bounds" begin
    rng = Random.MersenneTwister(42)
    true_rho = 0.8
    true_sigma = 0.5
    T_obs = 500

    y = zeros(T_obs)
    for t in 2:T_obs
        y[t] = true_rho * y[t-1] + true_sigma * randn(rng)
    end
    data = reshape(y, :, 1)

    function sim_ar1(theta, T_periods, burn; rng=Random.default_rng())
        rho, sigma = theta
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn)
            sim[t] = rho * sim[t-1] + sigma * randn(rng)
        end
        reshape(sim[(burn+1):end], :, 1)
    end

    bounds = ParameterTransform([-1.0, 0.0], [1.0, Inf])
    result = estimate_smm(sim_ar1, autocovariance_moments, [0.5, 0.3], data;
                          sim_ratio=5, burn=100, bounds=bounds,
                          rng=Random.MersenneTwister(123))

    @test result.converged
    @test -1.0 < result.theta[1] < 1.0  # bounded
    @test result.theta[2] > 0.0           # positive sigma
end

@testset "estimate_smm — identity weighting" begin
    rng = Random.MersenneTwister(99)
    y = 0.6 .* [zeros(1); randn(rng, 299)] .+ 0.4 .* randn(rng, 300)
    # simple cumsum-like
    for t in 2:300
        y[t] = 0.6 * y[t-1] + 0.4 * randn(rng)
    end
    data = reshape(y, :, 1)

    function sim_fn(theta, T_periods, burn; rng=Random.default_rng())
        rho = theta[1]
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn)
            sim[t] = rho * sim[t-1] + randn(rng)
        end
        reshape(sim[(burn+1):end], :, 1)
    end

    result = estimate_smm(sim_fn, d -> autocovariance_moments(d; lags=1),
                          [0.3], data;
                          sim_ratio=5, burn=50, weighting=:identity,
                          rng=Random.MersenneTwister(42))
    @test result isa SMMModel{Float64}
end

@testset "j_test on SMMModel" begin
    rng = Random.MersenneTwister(42)
    y = zeros(300)
    for t in 2:300
        y[t] = 0.7 * y[t-1] + 0.5 * randn(rng)
    end
    data = reshape(y, :, 1)

    function sim_fn(theta, T_periods, burn; rng=Random.default_rng())
        rho, sigma = theta
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn)
            sim[t] = rho * sim[t-1] + sigma * randn(rng)
        end
        reshape(sim[(burn+1):end], :, 1)
    end

    result = estimate_smm(sim_fn, d -> autocovariance_moments(d; lags=2),
                          [0.5, 0.3], data;
                          sim_ratio=5, burn=100, weighting=:two_step,
                          rng=Random.MersenneTwister(55))
    # 4 moments (var + 2 autocov), 2 params → overidentified
    jt = j_test(result)
    @test jt.df == result.n_moments - result.n_params
    @test jt.J_stat >= 0
    @test 0 <= jt.p_value <= 1
end

@testset "SMMModel report and show" begin
    rng = Random.MersenneTwister(42)
    y = zeros(200)
    for t in 2:200; y[t] = 0.7 * y[t-1] + randn(rng); end
    data = reshape(y, :, 1)

    function sim_fn(theta, T_periods, burn; rng=Random.default_rng())
        rho = theta[1]
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn); sim[t] = rho * sim[t-1] + randn(rng); end
        reshape(sim[(burn+1):end], :, 1)
    end

    result = estimate_smm(sim_fn, d -> autocovariance_moments(d; lags=1),
                          [0.5], data; sim_ratio=3, burn=50,
                          rng=Random.MersenneTwister(42))
    io = IOBuffer()
    show(io, result)
    str = String(take!(io))
    @test occursin("SMM", str)
    @test occursin("Sim ratio", str)

    # report should not error
    report(result)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: FAIL with "UndefVarError: estimate_smm not defined"

**Step 3: Implement estimate_smm**

Append to `src/gmm/smm.jl`:

```julia
# =============================================================================
# SMM Estimation
# =============================================================================

"""
    estimate_smm(simulator_fn, moments_fn, theta0, data;
                 sim_ratio=5, burn=100, weighting=:two_step,
                 bounds=nothing, hac=true, bandwidth=0,
                 max_iter=1000, tol=1e-8,
                 rng=Random.default_rng()) -> SMMModel{T}

Estimate parameters via Simulated Method of Moments.

Minimizes `Q(θ) = (m_data - m_sim(θ))' W (m_data - m_sim(θ))` where
`m_data` are data moments and `m_sim(θ)` are simulated moments.

# Arguments
- `simulator_fn(theta, T_periods, burn; rng)` — simulates T_periods obs after discarding burn
- `moments_fn(data) -> Vector{T}` — computes moment vector from any T×k data matrix
- `theta0` — initial parameter guess
- `data` — observed data matrix (T×k)

# Keywords
- `sim_ratio::Int=5` — τ = simulation periods / data periods
- `burn::Int=100` — burn-in periods for simulator
- `weighting::Symbol=:two_step` — `:identity` or `:two_step`
- `bounds::Union{Nothing,ParameterTransform}=nothing` — parameter bounds
- `hac::Bool=true` — HAC for weighting matrix
- `bandwidth::Int=0` — HAC bandwidth (0 = automatic)
- `max_iter::Int=1000` — max optimizer iterations
- `tol=1e-8` — convergence tolerance
- `rng` — random number generator

# References
- Ruge-Murcia (2012), Lee & Ingram (1991)
"""
function estimate_smm(simulator_fn::Function, moments_fn::Function,
                      theta0::AbstractVector, data::AbstractMatrix;
                      sim_ratio::Int=5, burn::Int=100,
                      weighting::Symbol=:two_step,
                      bounds::Union{Nothing,ParameterTransform}=nothing,
                      hac::Bool=true, bandwidth::Int=0,
                      max_iter::Int=1000, tol::Real=1e-8,
                      rng=Random.default_rng())
    T_type = eltype(data) <: AbstractFloat ? eltype(data) : Float64
    data_T = Matrix{T_type}(data)
    theta0_T = T_type.(theta0)
    tol_T = T_type(tol)

    n_obs = size(data_T, 1)
    n_params = length(theta0_T)
    T_sim = sim_ratio * n_obs

    # Compute data moments
    m_data = moments_fn(data_T)
    n_moments = length(m_data)

    @assert n_moments >= n_params "SMM requires at least as many moments as parameters"

    # Set up parameter transform
    has_bounds = bounds !== nothing
    if has_bounds
        phi0 = to_unconstrained(bounds, theta0_T)
    else
        phi0 = copy(theta0_T)
    end

    # SMM objective: Q(θ) = g(θ)' W g(θ) where g(θ) = m_data - m_sim(θ)
    function smm_moment_discrepancy(theta_or_phi)
        theta = has_bounds ? to_constrained(bounds, theta_or_phi) : theta_or_phi
        sim_data = simulator_fn(theta, T_sim, burn; rng=copy(rng))
        m_sim = moments_fn(Matrix{T_type}(sim_data))
        m_data .- m_sim
    end

    function smm_objective(phi, W)
        g = smm_moment_discrepancy(phi)
        dot(g, W * g)
    end

    # Step 1: Identity weighting
    W1 = Matrix{T_type}(I, n_moments, n_moments)
    obj1(phi) = smm_objective(phi, W1)

    result1 = Optim.optimize(obj1, phi0, Optim.NelderMead(),
                              Optim.Options(iterations=max_iter, f_reltol=tol_T))
    if !Optim.converged(result1)
        result1_lbfgs = Optim.optimize(obj1, phi0, Optim.LBFGS(),
                                        Optim.Options(iterations=max_iter, f_reltol=tol_T))
        if Optim.minimum(result1_lbfgs) < Optim.minimum(result1)
            result1 = result1_lbfgs
        end
    end

    phi_hat = Optim.minimizer(result1)
    converged = Optim.converged(result1)
    iterations = Optim.iterations(result1)

    if weighting == :two_step
        # Step 2: Optimal weighting from data
        W2 = smm_weighting_matrix(data_T, moments_fn; hac=hac, bandwidth=bandwidth)

        obj2(phi) = smm_objective(phi, W2)
        result2 = Optim.optimize(obj2, phi_hat, Optim.NelderMead(),
                                  Optim.Options(iterations=max_iter, f_reltol=tol_T))
        if !Optim.converged(result2)
            result2_lbfgs = Optim.optimize(obj2, phi_hat, Optim.LBFGS(),
                                            Optim.Options(iterations=max_iter, f_reltol=tol_T))
            if Optim.minimum(result2_lbfgs) < Optim.minimum(result2)
                result2 = result2_lbfgs
            end
        end

        phi_hat = Optim.minimizer(result2)
        W_final = W2
        converged = Optim.converged(result2)
        iterations += Optim.iterations(result2)
    else
        W_final = W1
    end

    # Map back to constrained space
    theta_hat = has_bounds ? to_constrained(bounds, phi_hat) : phi_hat

    # Final moment discrepancy
    g_bar = smm_moment_discrepancy(phi_hat)

    # Numerical Jacobian of simulated moments w.r.t. theta
    function sim_moments(theta)
        sim_data = simulator_fn(theta, T_sim, burn; rng=copy(rng))
        moments_fn(Matrix{T_type}(sim_data))
    end
    D = numerical_gradient(sim_moments, theta_hat)

    # Variance-covariance computation
    # For two-step (optimal W):  V = (1 + 1/τ) * (D'WD)^{-1} / n
    # For identity (sandwich):   V = (1 + 1/τ) * (D'WD)^{-1} D'W Ω W D (D'WD)^{-1} / n
    sim_correction = one(T_type) + one(T_type) / T_type(sim_ratio)

    bread = D' * W_final * D
    bread_inv = robust_inv(bread)

    if weighting == :two_step
        vcov = sim_correction * bread_inv / T_type(n_obs)
    else
        Omega = smm_data_covariance(data_T, moments_fn; hac=hac, bandwidth=bandwidth)
        meat = D' * W_final * Omega * W_final * D
        vcov = sim_correction * (bread_inv * meat * bread_inv) / T_type(n_obs)
    end

    # Delta method for transforms
    if has_bounds
        J_transform = transform_jacobian(bounds, phi_hat)
        vcov = J_transform * vcov * J_transform'
    end

    # Ensure symmetric
    vcov = (vcov + vcov') / 2

    # J-statistic
    J_stat, J_pvalue = if n_moments > n_params
        J = T_type(n_obs) * dot(g_bar, W_final * g_bar)
        J = max(J, zero(T_type))
        df = n_moments - n_params
        (J, one(T_type) - cdf(Chisq(df), J))
    else
        (zero(T_type), one(T_type))
    end

    weighting_spec = GMMWeighting{T_type}(weighting, max_iter, tol_T)

    SMMModel{T_type}(
        theta_hat, vcov, n_moments, n_params, n_obs, weighting_spec,
        W_final, g_bar, J_stat, J_pvalue, converged, iterations, sim_ratio
    )
end

"""
    smm_data_covariance(data, moments_fn; hac=true, bandwidth=0) -> Matrix

Compute long-run covariance Ω of data moment contributions for sandwich formula.
"""
function smm_data_covariance(data::AbstractMatrix{T}, moments_fn::Function;
                              hac::Bool=true, bandwidth::Int=0) where {T<:AbstractFloat}
    n = size(data, 1)
    m_full = moments_fn(data)
    q = length(m_full)

    G = Matrix{T}(undef, n, q)
    for t in 1:n
        G[t, :] = moments_fn(data[t:t, :])
    end
    G_demean = G .- mean(G, dims=1)

    if hac
        long_run_covariance(G_demean; bandwidth=bandwidth, kernel=:bartlett)
    else
        (G_demean' * G_demean) / n
    end
end
```

**Step 4: Run tests**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: All testsets PASS

**Step 5: Commit**

```bash
git add src/gmm/smm.jl test/gmm/test_smm.jl
git commit -m "feat(gmm): implement estimate_smm with two-step estimation"
```

---

### Task 5: DSGE Integration — :smm Method

Add `:smm` as a third method to `estimate_dsge`, extending the existing `:irf_matching` and `:euler_gmm`.

**Files:**
- Modify: `src/dsge/types.jl:279` (allow `:smm` in assertion)
- Modify: `src/dsge/estimation.jl` (add `:smm` branch + `_estimate_dsge_smm` function)
- Test: `test/dsge/test_dsge.jl` (add SMM estimation test)

**Step 1: Write the failing test**

Append to `test/dsge/test_dsge.jl` (inside its main testset):

```julia
@testset "DSGE SMM Estimation" begin
    # Simple RBC-like: y_t = rho * y_{t-1} + sigma * e_t
    rng = Random.MersenneTwister(42)
    spec = @dsge Float64 begin
        endog = [y]
        exog = [e]
        params = [rho, sigma]
        equations = begin
            y[t] = rho * y[t-1] + sigma * e[t]
        end
    end
    spec = MacroEconometricModels.set_param_values(spec, Dict(:rho => 0.7, :sigma => 1.0))
    spec = compute_steady_state(spec)

    # Simulate data from the model
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 500; rng=rng)

    # Estimate via SMM
    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:smm, sim_ratio=5, burn=100,
                        rng=Random.MersenneTwister(123))

    @test est isa DSGEEstimation{Float64}
    @test est.method == :smm
    @test est.converged
    @test abs(est.theta[1] - 0.7) < 0.2  # reasonable recovery
    @test is_determined(est.solution)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Test; using MacroEconometricModels; using Random; include("test/dsge/test_dsge.jl")'`
Expected: FAIL with "ArgumentError: method must be :irf_matching or :euler_gmm"

**Step 3: Modify DSGEEstimation assertion**

In `src/dsge/types.jl:279`, change:
```julia
@assert method ∈ (:irf_matching, :euler_gmm)
```
to:
```julia
@assert method ∈ (:irf_matching, :euler_gmm, :smm)
```

**Step 4: Add :smm branch to estimate_dsge**

In `src/dsge/estimation.jl`, modify the `estimate_dsge` function signature to accept SMM keywords:

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
                        rng=Random.default_rng()) where {T<:AbstractFloat}
```

Add `:smm` branch in the method dispatch (after `:euler_gmm`):

```julia
    elseif method == :smm
        return _estimate_dsge_smm(spec, data_T, param_names;
                                    sim_ratio=sim_ratio, burn=burn,
                                    weighting=weighting, moments_fn=moments_fn,
                                    bounds=bounds, rng=rng)
    else
        throw(ArgumentError("method must be :irf_matching, :euler_gmm, or :smm"))
```

Add the implementation function at the end of `src/dsge/estimation.jl`:

```julia
# =============================================================================
# SMM Estimation (Ruge-Murcia 2012)
# =============================================================================

"""
    _estimate_dsge_smm(spec, data, param_names; ...) -> DSGEEstimation

Internal: SMM estimation of DSGE parameters.

Builds a simulator from the DSGE spec: for each candidate θ, updates spec
parameters, solves the model, and simulates data.
"""
function _estimate_dsge_smm(spec::DSGESpec{T}, data::Matrix{T},
                              param_names::Vector{Symbol};
                              sim_ratio=5, burn=100, weighting=:two_step,
                              moments_fn=d -> autocovariance_moments(d; lags=1),
                              bounds=nothing, rng=Random.default_rng()) where {T}
    theta0 = T[spec.param_values[p] for p in param_names]

    # Build DSGE simulator
    function dsge_simulator(theta, T_periods, burn_in; rng=Random.default_rng())
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[]
        )
        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if !is_determined(sol)
                return fill(T(NaN), T_periods, spec.n_endog)
            end
            sim_full = simulate(sol, T_periods + burn_in; rng=rng)
            sim_full[(burn_in+1):end, :]
        catch
            fill(T(NaN), T_periods, spec.n_endog)
        end
    end

    smm_result = estimate_smm(dsge_simulator, moments_fn, theta0, data;
                               sim_ratio=sim_ratio, burn=0,  # burn handled inside simulator
                               weighting=weighting, bounds=bounds, rng=rng)

    # Build solution at estimated parameters
    final_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        final_pv[pn] = smm_result.theta[i]
    end
    final_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, final_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[]
    )
    final_spec = compute_steady_state(final_spec)
    final_sol = solve(final_spec; method=:gensys)

    DSGEEstimation{T}(
        smm_result.theta, smm_result.vcov, param_names,
        :smm, smm_result.J_stat, smm_result.J_pvalue,
        final_sol, smm_result.converged, final_spec
    )
end
```

Also need `using Random` at the top of estimation.jl if not already present.

**Step 5: Run tests**

Run DSGE tests: `julia --project=. -e 'using Test, MacroEconometricModels, Random; include("test/dsge/test_dsge.jl")'`
Run SMM tests: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/dsge/types.jl src/dsge/estimation.jl test/dsge/test_dsge.jl
git commit -m "feat(dsge): add :smm estimation method via estimate_dsge"
```

---

### Task 6: refs(), report(), Exports Polish + Test Registration

Add SMM references, report dispatches, register test file in runtests.jl, final export cleanup.

**Files:**
- Modify: `src/summary_refs.jl` (add SMM references + TYPE_REFS)
- Modify: `src/summary.jl` (report dispatch — already works via AbstractGMMModel)
- Modify: `test/runtests.jl` (add `test_smm.jl` to Group 5)
- Test: `test/gmm/test_smm.jl` (add refs test)

**Step 1: Write the failing test**

Append inside the outer testset in `test/gmm/test_smm.jl`:

```julia
@testset "SMM refs()" begin
    rng = Random.MersenneTwister(42)
    y = zeros(200)
    for t in 2:200; y[t] = 0.7 * y[t-1] + randn(rng); end
    data = reshape(y, :, 1)

    function sim_fn(theta, T_periods, burn; rng=Random.default_rng())
        rho = theta[1]
        sim = zeros(T_periods + burn)
        for t in 2:(T_periods + burn); sim[t] = rho * sim[t-1] + randn(rng); end
        reshape(sim[(burn+1):end], :, 1)
    end

    result = estimate_smm(sim_fn, d -> autocovariance_moments(d; lags=1),
                          [0.5], data; sim_ratio=3, burn=50,
                          rng=Random.MersenneTwister(42))

    io = IOBuffer()
    refs(io, result)
    str = String(take!(io))
    @test occursin("Ruge-Murcia", str) || occursin("Lee", str) || occursin("Hansen", str)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'include("test/gmm/test_smm.jl")'`
Expected: FAIL — `refs()` method error or missing reference

**Step 3: Add references**

In `src/summary_refs.jl`, add to `_REFERENCES` dict:

```julia
:ruge_murcia2012 => (key=:ruge_murcia2012, authors="Ruge-Murcia, Francisco J.", year=2012,
    title="Estimating Nonlinear DSGE Models by the Simulated Method of Moments",
    journal="Journal of Economic Dynamics and Control", volume="36", issue="6", pages="914--938",
    doi="10.1016/j.jedc.2012.01.008", isbn="", publisher="", entry_type=:article),
:lee_ingram1991 => (key=:lee_ingram1991, authors="Lee, Bong-Soo and Ingram, Beth Fisher", year=1991,
    title="Simulation Estimation of Time-Series Models",
    journal="Journal of Econometrics", volume="47", issue="2--3", pages="197--205",
    doi="10.1016/0304-4076(91)90098-X", isbn="", publisher="", entry_type=:article),
:duffie_singleton1993 => (key=:duffie_singleton1993, authors="Duffie, Darrell and Singleton, Kenneth J.", year=1993,
    title="Simulated Moments Estimation of Markov Models of Asset Prices",
    journal="Econometrica", volume="61", issue="4", pages="929--952",
    doi="10.2307/2951768", isbn="", publisher="", entry_type=:article),
```

Add to `_TYPE_REFS`:
```julia
:SMMModel => [:ruge_murcia2012, :lee_ingram1991, :hansen1982],
:ParameterTransform => [:hansen1982],
```

Add `refs()` dispatch:
```julia
refs(io::IO, ::SMMModel; kw...) = refs(io, _TYPE_REFS[:SMMModel]; kw...)
```

**Step 4: Register test file in runtests.jl**

In `test/runtests.jl`, add `"gmm/test_smm.jl"` to Group 5 (after `"gmm/test_gmm.jl"`):

In the `TEST_GROUPS` array, Group 5 line:
```julia
"gmm/test_gmm.jl",
"gmm/test_smm.jl",
```

And in the sequential fallback section, after the GMM testset:
```julia
@testset "SMM Estimation" begin include("gmm/test_smm.jl") end
```

**Step 5: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All existing tests PASS + new SMM tests PASS

**Step 6: Commit**

```bash
git add src/summary_refs.jl test/runtests.jl test/gmm/test_smm.jl
git commit -m "feat(gmm): add SMM references, register test_smm.jl in test runner"
```

---

## Summary

| Task | Description | New/Modified Files |
|------|-------------|-------------------|
| 1 | Parameter transforms | `src/gmm/transforms.jl` (new), `src/MacroEconometricModels.jl` |
| 2 | GMM optimizer upgrade | `src/gmm/gmm.jl` |
| 3 | SMMModel + autocovariance_moments | `src/gmm/smm.jl` (new), `src/MacroEconometricModels.jl` |
| 4 | estimate_smm implementation | `src/gmm/smm.jl` |
| 5 | DSGE :smm integration | `src/dsge/types.jl`, `src/dsge/estimation.jl` |
| 6 | refs/report/exports/tests | `src/summary_refs.jl`, `test/runtests.jl` |

All test files: `test/gmm/test_smm.jl` (new), `test/gmm/test_gmm.jl` (existing, must still pass), `test/dsge/test_dsge.jl` (extended).
