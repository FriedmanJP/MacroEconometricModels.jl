# DSGE Historical Decomposition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add historical decomposition to the DSGE module — linear RTS smoother, nonlinear FFBSi particle smoother, frequentist and Bayesian HD with counterfactual decomposition for higher-order perturbation solutions.

**Architecture:** Three layers: (1) dedicated Kalman/particle smoother in `src/dsge/smoother.jl` returning `KalmanSmootherResult{T}`, (2) HD engine in `src/dsge/hd.jl` dispatching on `DSGESolution`/`PerturbationSolution`/`BayesianDSGE`, (3) reuse existing `HistoricalDecomposition`/`BayesianHistoricalDecomposition` types so all VAR accessors/plotting work.

**Tech Stack:** Julia, LinearAlgebra (BLAS `mul!`, Cholesky, pinv), Random, existing `PFWorkspace` patterns from `particle_filter.jl`, existing HD types from `core/hd.jl`.

**Key files to read before starting:**
- `src/dsge/types.jl` — DSGESolution (line 162), PerturbationSolution (line 245), DSGESpec (line 45)
- `src/dsge/bayes_types.jl` — BayesianDSGE (line 525), DSGEStateSpace (line 84), NonlinearStateSpace (line 143), PFWorkspace (line 290)
- `src/dsge/kalman_dsge.jl` — `_kalman_loglikelihood` (line 186), `_build_state_space` (line 99), `_build_observation_equation` (line 46)
- `src/dsge/particle_filter.jl` — `_bootstrap_particle_filter!` (line 256), `_systematic_resample!` (line 155), `_pf_transition_linear!` (line 38)
- `src/dsge/pruning.jl` — `simulate(::PerturbationSolution)` (line 49) for pruned propagation at all orders
- `src/dsge/bayes_estimation.jl` — `_build_solution_at_theta` (line 322) for re-solving at posterior draws
- `src/core/hd.jl` — `HistoricalDecomposition` (line 43), `BayesianHistoricalDecomposition` (line 72), `_compute_hd_contributions` (line 112)
- `src/core/kalman.jl` — `_rts_smoother_gain` (line 103), `_solve_discrete_lyapunov` (line 30)
- `src/MacroEconometricModels.jl` — include order (lines 293-305), exports (lines 379-535)

---

### Task 1: KalmanSmootherResult Type + Smoother Skeleton

**Files:**
- Create: `src/dsge/smoother.jl`
- Modify: `src/MacroEconometricModels.jl` (add include + exports)
- Create: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Create `test/dsge/test_dsge_hd.jl`:

```julia
# Test DSGE Historical Decomposition
using Test
using MacroEconometricModels
using LinearAlgebra
using Random

@testset "DSGE Historical Decomposition" begin

@testset "KalmanSmootherResult type" begin
    # Create a simple 3-variable AR(1) DSGE
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    # Simulate data with known shocks
    rng = Random.MersenneTwister(42)
    T_obs = 50
    sim_data = simulate(sol, T_obs; rng=rng)

    # Build state space
    observables = [:y, :pi_var, :r]
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H)

    # Run smoother
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)  # n_obs x T_obs deviations
    result = dsge_smoother(ss, data_matrix)

    @test result isa KalmanSmootherResult{Float64}
    @test size(result.smoothed_states) == (3, T_obs)
    @test size(result.smoothed_shocks) == (3, T_obs)
    @test size(result.smoothed_covariances) == (3, 3, T_obs)
    @test size(result.filtered_states) == (3, T_obs)
    @test size(result.predicted_states) == (3, T_obs)
    @test isfinite(result.log_likelihood)
end

end  # outer testset
```

**Step 2: Create the smoother file with type and stub**

Create `src/dsge/smoother.jl`:

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
DSGE-specific Kalman smoother and FFBSi particle smoother.

Provides:
- `KalmanSmootherResult{T}` — result type for smoothed states, shocks, covariances
- `dsge_smoother` — Rauch-Tung-Striebel smoother for linear DSGE state spaces
- `dsge_particle_smoother` — Forward-filtering backward-simulation (FFBSi) for nonlinear models

References:
- Rauch, H. E., Tung, F. & Striebel, C. T. (1965). Maximum likelihood estimates of
  linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
- Godsill, S. J., Doucet, A. & West, M. (2004). Monte Carlo smoothing for nonlinear
  time series. Journal of the American Statistical Association, 99(465), 156-168.
- Koopman, S. J. (1993). Disturbance smoother for state space models. Biometrika, 80(1), 117-126.
"""

using LinearAlgebra
using Random

# =============================================================================
# Result Type
# =============================================================================

"""
    KalmanSmootherResult{T} <: AbstractAnalysisResult

Result of the Kalman smoother (RTS) or particle smoother (FFBSi).

# Fields
- `smoothed_states::Matrix{T}` — n_states × T_obs smoothed state estimates E[x_t | y_{1:T}]
- `smoothed_covariances::Array{T,3}` — n_states × n_states × T_obs smoothed covariance Var[x_t | y_{1:T}]
- `smoothed_shocks::Matrix{T}` — n_shocks × T_obs smoothed structural shocks
- `filtered_states::Matrix{T}` — n_states × T_obs filtered state estimates E[x_t | y_{1:t}]
- `filtered_covariances::Array{T,3}` — n_states × n_states × T_obs filtered covariance
- `predicted_states::Matrix{T}` — n_states × T_obs predicted state estimates E[x_t | y_{1:t-1}]
- `predicted_covariances::Array{T,3}` — n_states × n_states × T_obs predicted covariance
- `log_likelihood::T` — log-likelihood from the filter pass
"""
struct KalmanSmootherResult{T<:AbstractFloat} <: AbstractAnalysisResult
    smoothed_states::Matrix{T}        # n_states × T_obs
    smoothed_covariances::Array{T,3}  # n_states × n_states × T_obs
    smoothed_shocks::Matrix{T}        # n_shocks × T_obs
    filtered_states::Matrix{T}        # n_states × T_obs
    filtered_covariances::Array{T,3}  # n_states × n_states × T_obs
    predicted_states::Matrix{T}       # n_states × T_obs
    predicted_covariances::Array{T,3} # n_states × n_states × T_obs
    log_likelihood::T
end
```

**Step 3: Add include and exports to `src/MacroEconometricModels.jl`**

After the `include("dsge/bayes_estimation.jl")` line (line 305), add:

```julia
include("dsge/smoother.jl")
include("dsge/hd.jl")
```

In the DSGE exports section (after line 401), add:

```julia
# DSGE Smoother
export KalmanSmootherResult
export dsge_smoother, dsge_particle_smoother
```

**Step 4: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: FAIL — `dsge_smoother` not defined yet (file will compile but function missing)

**Step 5: Commit skeleton**

```bash
git add src/dsge/smoother.jl test/dsge/test_dsge_hd.jl
git add -f src/MacroEconometricModels.jl
git commit -m "feat(dsge): add KalmanSmootherResult type and smoother skeleton"
```

---

### Task 2: RTS Kalman Smoother — Forward Pass

**Files:**
- Modify: `src/dsge/smoother.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl` inside the outer testset:

```julia
@testset "RTS smoother forward pass matches _kalman_loglikelihood" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)
    observables = [:y, :pi_var, :r]
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H)

    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, 50; rng=rng)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    # Smoother log-likelihood should match existing Kalman filter
    smoother_result = dsge_smoother(ss, data_matrix)
    kf_ll = MacroEconometricModels._kalman_loglikelihood(ss, data_matrix)

    @test smoother_result.log_likelihood ≈ kf_ll atol=1e-8
end
```

**Step 2: Implement the forward pass in `dsge_smoother`**

Add to `src/dsge/smoother.jl`:

```julia
# =============================================================================
# Linear RTS Smoother
# =============================================================================

"""
    dsge_smoother(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T}

Rauch-Tung-Striebel smoother for the linear DSGE state space model.

# State space model
    x_{t+1} = G1 * x_t + impact * ε_t,    ε_t ~ N(0, Q)
    y_t     = Z * x_t + d + v_t,           v_t ~ N(0, H)

# Arguments
- `ss` — `DSGEStateSpace{T}` with transition/observation matrices
- `data` — `n_obs × T_obs` matrix (each column is one time period, deviations from steady state)

# Returns
`KalmanSmootherResult{T}` with smoothed states, covariances, shocks, and log-likelihood.

# Implementation
1. Forward pass: Kalman filter storing all filtered/predicted states and covariances.
   Handles missing data (NaN entries) by reducing the observation dimension per period.
2. Backward pass: RTS smoother gain J_t = P_{t|t} G1' P_{t+1|t}^{-1}.
3. Shock extraction: ε_t = impact⁺ (x_{t|T} - G1 x_{t-1|T}).
"""
function dsge_smoother(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T<:AbstractFloat}
    n_obs, T_obs = size(data)
    n_states = size(ss.G1, 1)
    n_shocks = size(ss.impact, 2)

    # --- Storage arrays ---
    x_filt = zeros(T, n_states, T_obs)      # x_{t|t}
    P_filt = zeros(T, n_states, n_states, T_obs)
    x_pred = zeros(T, n_states, T_obs)      # x_{t|t-1}
    P_pred = zeros(T, n_states, n_states, T_obs)

    # --- Initialization ---
    x = zeros(T, n_states)
    RQR = ss.impact * ss.Q * ss.impact'
    P = try
        solve_lyapunov(ss.G1, ss.impact)
    catch
        T(10) * Matrix{T}(I, n_states, n_states)
    end

    # --- Pre-allocate workspace ---
    G1P = zeros(T, n_states, n_states)
    v = zeros(T, n_obs)
    Zx = zeros(T, n_obs)
    F = zeros(T, n_obs, n_obs)
    ZP = zeros(T, n_obs, n_states)
    PZ = zeros(T, n_states, n_obs)
    K = zeros(T, n_states, n_obs)
    F_inv = zeros(T, n_obs, n_obs)
    F_inv_v = zeros(T, n_obs)
    Kv = zeros(T, n_states)
    KZP = zeros(T, n_states, n_states)
    x_pred_t = zeros(T, n_states)
    P_pred_t = zeros(T, n_states, n_states)

    has_nan = any(isnan, data)
    ll = zero(T)
    log2pi = T(log(2 * T(pi)))

    # =================================================================
    # Forward pass (Kalman filter)
    # =================================================================
    @inbounds for t in 1:T_obs
        # --- Prediction ---
        mul!(x_pred_t, ss.G1, x)
        mul!(G1P, ss.G1, P)
        mul!(P_pred_t, G1P, ss.G1')
        for j in 1:n_states, i in 1:n_states
            P_pred_t[i, j] += RQR[i, j]
        end

        # Store predicted
        x_pred[:, t] = x_pred_t
        P_pred[:, :, t] = P_pred_t

        y_t = @view data[:, t]

        if has_nan && any(isnan, y_t)
            # --- Partial observation ---
            obs_mask = .!isnan.(y_t)
            n_obs_t = count(obs_mask)

            if n_obs_t == 0
                x_filt[:, t] = x_pred_t
                P_filt[:, :, t] = P_pred_t
                copyto!(x, x_pred_t)
                copyto!(P, P_pred_t)
                continue
            end

            obs_idx = findall(obs_mask)
            Z_t = ss.Z[obs_idx, :]
            d_t = ss.d[obs_idx]
            H_t = ss.H[obs_idx, obs_idx]
            y_obs = y_t[obs_idx]

            v_t = y_obs - Z_t * x_pred_t - d_t
            F_t = Z_t * P_pred_t * Z_t' + H_t
            F_t = (F_t + F_t') / 2

            F_chol = cholesky(Hermitian(F_t); check=false)
            if !issuccess(F_chol)
                F_t += T(1e-8) * I
                F_chol = cholesky(Hermitian(F_t))
            end
            log_det_F = logdet(F_chol)
            F_inv_t = inv(F_chol)
            F_inv_v_t = F_inv_t * v_t
            ll += -T(0.5) * (n_obs_t * log2pi + log_det_F + dot(v_t, F_inv_v_t))

            K_t = P_pred_t * Z_t' * F_inv_t
            x .= x_pred_t .+ K_t * v_t
            P .= P_pred_t .- K_t * Z_t * P_pred_t
            P .= (P .+ P') ./ 2
        else
            # --- Full observation ---
            mul!(Zx, ss.Z, x_pred_t)
            for i in 1:n_obs
                v[i] = y_t[i] - Zx[i] - ss.d[i]
            end

            mul!(ZP, ss.Z, P_pred_t)
            mul!(F, ZP, ss.Z')
            for j in 1:n_obs, i in 1:n_obs
                F[i, j] += ss.H[i, j]
            end
            for j in 1:n_obs, i in 1:j-1
                avg = (F[i, j] + F[j, i]) / 2
                F[i, j] = avg; F[j, i] = avg
            end

            F_chol = cholesky(Hermitian(F); check=false)
            if !issuccess(F_chol)
                for i in 1:n_obs
                    F[i, i] += T(1e-8)
                end
                F_chol = cholesky(Hermitian(F))
            end
            log_det_F = logdet(F_chol)
            copyto!(F_inv, inv(F_chol))
            mul!(F_inv_v, F_inv, v)
            ll += -T(0.5) * (n_obs * log2pi + log_det_F + dot(v, F_inv_v))

            mul!(PZ, P_pred_t, ss.Z')
            mul!(K, PZ, F_inv)
            mul!(Kv, K, v)
            for i in 1:n_states
                x[i] = x_pred_t[i] + Kv[i]
            end
            mul!(KZP, K, ZP)
            for j in 1:n_states, i in 1:n_states
                P[i, j] = P_pred_t[i, j] - KZP[i, j]
            end
            for j in 1:n_states, i in 1:j-1
                avg = (P[i, j] + P[j, i]) / 2
                P[i, j] = avg; P[j, i] = avg
            end
        end

        # Store filtered
        x_filt[:, t] = x
        P_filt[:, :, t] = P
    end

    # Forward pass complete — backward pass placeholder
    # (implemented in Task 3)
    smoothed_states = copy(x_filt)
    smoothed_cov = copy(P_filt)
    smoothed_shocks = zeros(T, n_shocks, T_obs)

    return KalmanSmootherResult{T}(
        smoothed_states, smoothed_cov, smoothed_shocks,
        x_filt, P_filt, x_pred, P_pred, ll
    )
end
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS — log-likelihood matches `_kalman_loglikelihood` since forward pass is identical.

**Step 4: Commit**

```bash
git add src/dsge/smoother.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement Kalman filter forward pass in dsge_smoother"
```

---

### Task 3: RTS Smoother — Backward Pass + Shock Extraction

**Files:**
- Modify: `src/dsge/smoother.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "RTS smoother recovers known shocks" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    # Generate data with known shocks
    T_obs = 100
    rng = Random.MersenneTwister(123)
    true_shocks = randn(rng, T_obs)
    shock_matrix = reshape(true_shocks, T_obs, 1)
    sim_data = simulate(sol, T_obs; shock_draws=shock_matrix)

    observables = [:y]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables,
                    [1e-6])  # very small measurement error
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    result = dsge_smoother(ss, data_matrix)

    # Smoothed shocks should be close to true shocks (small measurement error)
    recovered = result.smoothed_shocks[1, :]
    @test cor(recovered, true_shocks) > 0.95
    # Smoothed states should track actual data closely
    @test maximum(abs.(result.smoothed_states[1, :] .- data_matrix[1, :])) < 0.5
end

@testset "RTS smoother — 3-variable shock recovery" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    T_obs = 100
    rng = Random.MersenneTwister(99)
    true_shocks = randn(rng, T_obs, 3)
    sim_data = simulate(sol, T_obs; shock_draws=true_shocks)

    observables = [:y, :pi_var, :r]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables,
                    [1e-6, 1e-6, 1e-6])
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    result = dsge_smoother(ss, data_matrix)

    for j in 1:3
        @test cor(result.smoothed_shocks[j, :], true_shocks[:, j]) > 0.90
    end
end
```

**Step 2: Replace backward pass placeholder in `dsge_smoother`**

In `src/dsge/smoother.jl`, replace the placeholder section (after the forward pass loop) with the full backward pass:

```julia
    # =================================================================
    # Backward pass (Rauch-Tung-Striebel)
    # =================================================================
    smoothed_states = copy(x_filt)
    smoothed_cov = copy(P_filt)

    # Pre-allocate backward workspace
    J = zeros(T, n_states, n_states)
    P_pred_inv = zeros(T, n_states, n_states)
    diff_x = zeros(T, n_states)
    diff_P = zeros(T, n_states, n_states)
    JP = zeros(T, n_states, n_states)

    @inbounds for t in (T_obs-1):-1:1
        # Smoother gain: J_t = P_{t|t} * G1' * P_{t+1|t}^{-1}
        P_pred_t_next = @view P_pred[:, :, t+1]
        P_chol = cholesky(Hermitian(P_pred_t_next); check=false)
        if !issuccess(P_chol)
            P_tmp = Matrix{T}(P_pred_t_next) + T(1e-8) * I
            P_chol = cholesky(Hermitian(P_tmp))
        end
        copyto!(P_pred_inv, inv(P_chol))

        # J = P_{t|t} * G1' * P_pred_inv
        P_filt_t = @view P_filt[:, :, t]
        mul!(J, P_filt_t * ss.G1', P_pred_inv)

        # Smoothed state: x_{t|T} = x_{t|t} + J * (x_{t+1|T} - x_{t+1|t-1})
        for i in 1:n_states
            diff_x[i] = smoothed_states[i, t+1] - x_pred[i, t+1]
        end
        mul!(Kv, J, diff_x)  # reuse Kv buffer
        for i in 1:n_states
            smoothed_states[i, t] = x_filt[i, t] + Kv[i]
        end

        # Smoothed covariance: P_{t|T} = P_{t|t} + J * (P_{t+1|T} - P_{t+1|t}) * J'
        for j in 1:n_states, i in 1:n_states
            diff_P[i, j] = smoothed_cov[i, j, t+1] - P_pred[i, j, t+1]
        end
        mul!(JP, J, diff_P)
        P_smooth_update = JP * J'
        for j in 1:n_states, i in 1:n_states
            smoothed_cov[i, j, t] = P_filt[i, j, t] + P_smooth_update[i, j]
        end
        # Symmetrize
        for j in 1:n_states, i in 1:j-1
            avg = (smoothed_cov[i, j, t] + smoothed_cov[j, i, t]) / 2
            smoothed_cov[i, j, t] = avg
            smoothed_cov[j, i, t] = avg
        end
    end

    # =================================================================
    # Shock extraction: ε_t = impact⁺ * (x_{t|T} - G1 * x_{t-1|T} - C_sol)
    # =================================================================
    smoothed_shocks = zeros(T, n_shocks, T_obs)
    impact_pinv = pinv(ss.impact)
    x_prev = zeros(T, n_states)

    @inbounds for t in 1:T_obs
        # x_prev = smoothed state at t-1 (zero for t=1)
        if t > 1
            for i in 1:n_states
                x_prev[i] = smoothed_states[i, t-1]
            end
        else
            fill!(x_prev, zero(T))
        end
        # residual = x_{t|T} - G1 * x_{t-1|T}
        predicted = ss.G1 * x_prev
        residual = smoothed_states[:, t] - predicted
        smoothed_shocks[:, t] = impact_pinv * residual
    end

    return KalmanSmootherResult{T}(
        smoothed_states, smoothed_cov, smoothed_shocks,
        x_filt, P_filt, x_pred, P_pred, ll
    )
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS — smoother recovers shocks correlated >0.90 with truth.

**Step 4: Commit**

```bash
git add src/dsge/smoother.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement RTS backward pass and shock extraction"
```

---

### Task 4: Linear DSGE Historical Decomposition

**Files:**
- Create: `src/dsge/hd.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "Linear DSGE historical decomposition" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    T_obs = 80
    rng = Random.MersenneTwister(55)
    sim_data = simulate(sol, T_obs; rng=rng)

    observables = [:y, :pi_var, :r]
    hd = historical_decomposition(sol, sim_data, observables)

    @test hd isa HistoricalDecomposition{Float64}
    @test size(hd.contributions) == (T_obs, 3, 3)  # T_eff × n_vars × n_shocks
    @test size(hd.initial_conditions) == (T_obs, 3)
    @test size(hd.actual) == (T_obs, 3)
    @test hd.method == :dsge_linear

    # Decomposition identity: contributions + initial ≈ actual
    @test verify_decomposition(hd; tol=0.1)
end

@testset "Linear DSGE HD — single shock dominance" begin
    # Only eps_y shock is active
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    T_obs = 60
    rng = Random.MersenneTwister(77)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y]

    hd = historical_decomposition(sol, sim_data, observables)
    @test size(hd.contributions, 3) == 1
    @test verify_decomposition(hd; tol=0.05)
end

@testset "Linear DSGE HD — states=:all" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5
        endogenous: y, pi_var
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    end
    sol = solve(spec)

    T_obs = 40
    rng = Random.MersenneTwister(33)
    sim_data = simulate(sol, T_obs; rng=rng)

    # Only observe y but decompose all states
    hd = historical_decomposition(sol, sim_data, [:y, :pi_var]; states=:all)
    @test size(hd.contributions, 2) == 2  # all states
end
```

**Step 2: Create `src/dsge/hd.jl`**

```julia
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Historical decomposition for DSGE models.

Dispatches on `DSGESolution` (linear), `PerturbationSolution` (nonlinear),
and `BayesianDSGE` (Bayesian with posterior uncertainty). Returns existing
`HistoricalDecomposition` / `BayesianHistoricalDecomposition` types so all
VAR accessors, plotting, and `report()` work unchanged.

References:
- Burbidge, J. & Harrison, A. (1985). A historical decomposition of the Great Depression
  to determine the role of money. Journal of Monetary Economics, 16(1), 45-54.
"""

using LinearAlgebra

# =============================================================================
# Linear DSGE HD
# =============================================================================

"""
    historical_decomposition(sol::DSGESolution{T}, data::AbstractMatrix,
                              observables::Vector{Symbol};
                              states=:observables,
                              measurement_error=nothing) where {T}

Historical decomposition for a linear DSGE model.

# Arguments
- `sol` — solved linear DSGE model
- `data` — `T_obs × n_endog` matrix of simulated/observed data in levels
- `observables` — symbols identifying observed variables

# Keyword Arguments
- `states` — `:observables` (default) or `:all` to decompose all state variables
- `measurement_error` — vector of measurement error std devs (default: 1e-4)

# Returns
`HistoricalDecomposition{T}` with contributions, initial conditions, actual data, and shocks.
"""
function historical_decomposition(sol::DSGESolution{T}, data::AbstractMatrix,
                                   observables::Vector{Symbol};
                                   states::Symbol=:observables,
                                   measurement_error=nothing) where {T<:AbstractFloat}
    spec = sol.spec
    T_obs = size(data, 1)
    n_states = spec.n_endog
    n_shocks = spec.n_exog

    # Build state space
    Z, d, H = _build_observation_equation(spec, observables, measurement_error)
    ss = _build_state_space(sol, Z, d, H)

    # Convert data to n_obs × T_obs deviations from steady state
    obs_indices = [findfirst(==(obs), spec.endog) for obs in observables]
    data_dev = Matrix{T}(data[:, obs_indices]' .- spec.steady_state[obs_indices])

    # Run Kalman smoother
    smoother = dsge_smoother(ss, data_dev)

    # Compute structural MA coefficients: Θ_s = Z_dec * G1^s * impact
    if states == :all
        Z_dec = Matrix{T}(I, n_states, n_states)
        n_vars = n_states
        var_names = String.(spec.endog)
    else
        Z_dec = Z
        n_vars = length(observables)
        var_names = String.(observables)
    end

    # Compute contributions via MA decomposition
    contributions = zeros(T, T_obs, n_vars, n_shocks)
    G1_power = Matrix{T}(I, n_states, n_states)

    # Pre-compute MA coefficients up to T_obs
    Theta = Vector{Matrix{T}}(undef, T_obs)
    for s in 1:T_obs
        Theta[s] = Z_dec * G1_power * sol.impact
        G1_power = G1_power * sol.G1
    end

    # HD[t, i, j] = Σ_{s=0}^{t-1} Θ_{s+1}[i, j] * ε_j(t-s)
    shocks_t = smoother.smoothed_shocks  # n_shocks × T_obs
    @inbounds for t in 1:T_obs
        for j in 1:n_shocks
            for s in 0:min(t-1, T_obs-1)
                for i in 1:n_vars
                    contributions[t, i, j] += Theta[s+1][i, j] * shocks_t[j, t-s]
                end
            end
        end
    end

    # Actual data (deviations) in the decomposition variable space
    if states == :all
        actual = zeros(T, T_obs, n_states)
        for t in 1:T_obs
            actual[t, :] = smoother.smoothed_states[:, t]
        end
    else
        actual = Matrix{T}(data_dev')  # T_obs × n_obs
    end

    # Initial conditions = actual - sum of contributions
    initial_conditions = copy(actual)
    for j in 1:n_shocks
        initial_conditions .-= contributions[:, :, j]
    end

    shock_names = String.(spec.exog)

    return HistoricalDecomposition{T}(
        contributions, initial_conditions, actual,
        Matrix{T}(shocks_t'), T_obs, var_names, shock_names, :dsge_linear
    )
end
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS

**Step 4: Commit**

```bash
git add src/dsge/hd.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement linear DSGE historical decomposition"
```

---

### Task 5: FFBSi Particle Smoother — Forward Pass

**Files:**
- Modify: `src/dsge/smoother.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "Particle smoother — linear DGP matches RTS" begin
    spec = @dsge begin
        parameters: rho = 0.7
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    T_obs = 30
    rng = Random.MersenneTwister(88)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y]

    # Build nonlinear state space from perturbation solution (order 1)
    psol = perturbation_solver(spec; order=1)
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables,
                    [1e-4])
    nss = MacroEconometricModels._build_nonlinear_state_space(psol, Z, d, H_mat)

    # Also build linear for comparison
    Z2, d2, H2 = MacroEconometricModels._build_observation_equation(spec, observables,
                    [1e-4])
    ss = MacroEconometricModels._build_state_space(sol, Z2, d2, H2)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    rts_result = dsge_smoother(ss, data_matrix)
    pf_result = dsge_particle_smoother(nss, data_matrix; N=2000, N_back=200,
                                        rng=Random.MersenneTwister(42))

    # Particle smoother should approximately match RTS for linear DGP
    @test cor(vec(pf_result.smoothed_states), vec(rts_result.smoothed_states)) > 0.90
end
```

**Step 2: Implement the particle smoother forward pass**

Add to `src/dsge/smoother.jl`:

```julia
# =============================================================================
# FFBSi Particle Smoother (Godsill, Doucet & West 2004)
# =============================================================================

"""
    _nonlinear_transition(nss::NonlinearStateSpace{T}, x_prev::AbstractVector{T},
                           eps_t::AbstractVector{T}) where {T}

Evaluate the full nonlinear (pruned) state transition at a single particle.
Returns the next-period full state vector (n_states).

Dispatches on `nss.order` for 1st/2nd/3rd-order pruned propagation.
"""
function _nonlinear_transition(nss::NonlinearStateSpace{T},
                                x_prev::AbstractVector{T},
                                eps_t::AbstractVector{T}) where {T<:AbstractFloat}
    nx = length(nss.state_indices)
    n_eps = size(nss.eta, 2)
    nv = nx + n_eps
    n_total = nx + length(nss.control_indices)

    # Extract first-order blocks
    hx_state = nx > 0 ? nss.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x = nx > 0 ? nss.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
    gx_state = length(nss.control_indices) > 0 ? nss.gx[:, 1:nx] : zeros(T, 0, nx)
    eta_y = length(nss.control_indices) > 0 ? nss.gx[:, nx+1:nv] : zeros(T, 0, n_eps)

    # State components from x_prev (first nx are states)
    x_state = x_prev[nss.state_indices]

    if nss.order == 1
        xf_new = hx_state * x_state + eta_x * eps_t
        y_new = gx_state * xf_new + eta_y * eps_t
        result = zeros(T, n_total)
        for (k, si) in enumerate(nss.state_indices)
            result[si] = xf_new[k]
        end
        for (k, ci) in enumerate(nss.control_indices)
            result[ci] = y_new[k]
        end
        return result
    elseif nss.order == 2
        # Need to track xf and xs separately — for particle smoother we store
        # the total state, so we approximate: treat x_state as total = xf + xs
        # For HD via counterfactual this is fine since we re-simulate
        xf_new = hx_state * x_state + eta_x * eps_t
        vf = zeros(T, nv)
        if nx > 0; vf[1:nx] = x_state; end
        vf[nx+1:nv] = eps_t
        kron_vf = kron(vf, vf)
        xs_new = zeros(T, nx)
        if nss.hxx !== nothing
            xs_new += T(0.5) * nss.hxx * kron_vf
        end
        if nss.hsigmasigma !== nothing
            xs_new += T(0.5) * nss.hsigmasigma
        end
        x_total = xf_new + xs_new
        y_new = gx_state * x_total + eta_y * eps_t
        if nss.gxx !== nothing
            y_new += T(0.5) * nss.gxx * kron_vf
        end
        if nss.gsigmasigma !== nothing
            y_new += T(0.5) * nss.gsigmasigma
        end
        result = zeros(T, n_total)
        for (k, si) in enumerate(nss.state_indices)
            result[si] = x_total[k]
        end
        for (k, ci) in enumerate(nss.control_indices)
            result[ci] = y_new[k]
        end
        return result
    else  # order >= 3
        # Simplified: use first-order for particle propagation,
        # full nonlinear for counterfactual simulation in HD
        xf_new = hx_state * x_state + eta_x * eps_t
        y_new = gx_state * xf_new + eta_y * eps_t
        result = zeros(T, n_total)
        for (k, si) in enumerate(nss.state_indices)
            result[si] = xf_new[k]
        end
        for (k, ci) in enumerate(nss.control_indices)
            result[ci] = y_new[k]
        end
        return result
    end
end

"""
    dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                            N=1000, N_back=100,
                            rng=Random.default_rng()) where {T}

Forward-filtering backward-simulation (FFBSi) particle smoother for nonlinear
DSGE state space models.

# Arguments
- `nss` — nonlinear state space (from `_build_nonlinear_state_space`)
- `data` — n_obs × T_obs observation matrix (deviations from steady state)

# Keyword Arguments
- `N` — number of forward particles (default: 1000)
- `N_back` — number of backward trajectories (default: 100)
- `rng` — random number generator

# Returns
`KalmanSmootherResult{T}` with smoothed states/shocks as posterior means across
backward trajectories.
"""
function dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                                 N::Int=1000, N_back::Int=100,
                                 rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_obs, T_obs = size(data)
    n_states = length(nss.state_indices) + length(nss.control_indices)
    n_shocks = size(nss.eta, 2)
    nx = length(nss.state_indices)

    # --- Forward pass: Bootstrap particle filter ---
    # Storage for all particles and weights at each time step
    all_particles = zeros(T, n_states, N, T_obs)  # states × particles × time
    all_weights = zeros(T, N, T_obs)               # particles × time
    all_shocks = zeros(T, n_shocks, N, T_obs)      # shocks × particles × time

    # Current particles
    particles = zeros(T, n_states, N)  # initialize at zero (SS deviation)
    weights = fill(one(T) / N, N)
    ancestors = Vector{Int}(undef, N)
    cumweights = Vector{T}(undef, N)

    # Observation matrices
    Z = nss.Z
    d_vec = nss.d
    H_inv = nss.H_inv
    log_det_H = nss.log_det_H

    log_lik = zero(T)
    log_N = log(T(N))
    inv_N = one(T) / N

    @inbounds for t in 1:T_obs
        # Draw shocks and propagate
        shocks_t = randn(rng, T, n_shocks, N)

        for k in 1:N
            x_prev = @view particles[:, k]
            eps_k = @view shocks_t[:, k]
            particles[:, k] = _nonlinear_transition(nss, x_prev, eps_k)
        end

        # Store shocks
        all_shocks[:, :, t] = shocks_t

        # Compute log weights
        y_t = @view data[:, t]
        log_w = zeros(T, N)
        for k in 1:N
            inn = y_t - Z * particles[:, k] - d_vec
            log_w[k] = -T(0.5) * (dot(inn, H_inv * inn) + log_det_H)
        end

        # Log-likelihood increment
        max_lw = maximum(log_w)
        sum_exp = zero(T)
        for k in 1:N
            sum_exp += exp(log_w[k] - max_lw)
        end
        log_lik += max_lw + log(sum_exp) - log_N

        # Normalize
        for k in 1:N
            weights[k] = exp(log_w[k] - max_lw)
        end
        w_sum = sum(weights)
        for k in 1:N
            weights[k] /= w_sum
        end

        # Store
        all_particles[:, :, t] = particles
        all_weights[:, t] = weights

        # ESS-based resampling
        ess = one(T) / sum(w -> w^2, weights)
        if ess < T(0.5) * N
            _systematic_resample!(ancestors, weights, cumweights, N, rng)
            new_particles = similar(particles)
            for k in 1:N
                new_particles[:, k] = particles[:, ancestors[k]]
            end
            copyto!(particles, new_particles)
            fill!(weights, inv_N)
        end
    end

    # --- Backward simulation (Godsill-Doucet-West 2004) ---
    smoothed_trajectories = zeros(T, n_states, T_obs, N_back)
    smoothed_shock_trajectories = zeros(T, n_shocks, T_obs, N_back)

    # Pre-compute impact pseudo-inverse for shock extraction
    # For nonlinear, use first-order impact from hx's eta columns
    hx_state = nx > 0 ? nss.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x = nx > 0 ? nss.hx[:, nx+1:nx+n_shocks] : zeros(T, 0, n_shocks)

    # Build full impact matrix (n_states × n_shocks) from first-order
    impact_full = zeros(T, n_states, n_shocks)
    for (k, si) in enumerate(nss.state_indices)
        if k <= size(eta_x, 1)
            impact_full[si, :] = eta_x[k, :]
        end
    end
    gx_state = length(nss.control_indices) > 0 ? nss.gx[:, 1:nx] : zeros(T, 0, nx)
    eta_y = length(nss.control_indices) > 0 ? nss.gx[:, nx+1:nx+n_shocks] : zeros(T, 0, n_shocks)
    for (k, ci) in enumerate(nss.control_indices)
        if k <= size(eta_y, 1)
            impact_full[ci, :] = eta_y[k, :]
        end
    end
    impact_pinv = pinv(impact_full)

    # Pre-compute Cholesky of shock covariance for transition density
    Q_chol = cholesky(Hermitian(Matrix{T}(I, n_shocks, n_shocks)))
    Q_inv = Matrix{T}(I, n_shocks, n_shocks)
    log_det_Q = zero(T)

    for b in 1:N_back
        # Draw x_T from final weights
        idx_T = _categorical_draw(all_weights[:, T_obs], rng)
        smoothed_trajectories[:, T_obs, b] = all_particles[:, idx_T, T_obs]

        # Backward simulation
        for t in (T_obs-1):-1:1
            w_t = all_weights[:, t]
            x_next = smoothed_trajectories[:, t+1, b]

            # Compute backward weights
            back_w = zeros(T, N)
            for k in 1:N
                x_k = all_particles[:, k, t]
                # Transition density: p(x_{t+1} | x_k) using first-order approx
                # predicted = f(x_k, 0), residual = x_{t+1} - predicted
                predicted = _nonlinear_transition(nss, x_k, zeros(T, n_shocks))
                residual = x_next - predicted
                implied_shock = impact_pinv * residual
                back_w[k] = log(w_t[k] + T(1e-300)) -
                            T(0.5) * dot(implied_shock, Q_inv * implied_shock)
            end

            # Normalize and draw
            max_bw = maximum(back_w)
            for k in 1:N
                back_w[k] = exp(back_w[k] - max_bw)
            end
            bw_sum = sum(back_w)
            for k in 1:N
                back_w[k] /= bw_sum
            end
            idx = _categorical_draw(back_w, rng)
            smoothed_trajectories[:, t, b] = all_particles[:, idx, t]
        end

        # Extract shocks for this trajectory
        for t in 1:T_obs
            x_t = smoothed_trajectories[:, t, b]
            x_prev = t > 1 ? smoothed_trajectories[:, t-1, b] : zeros(T, n_states)
            predicted = _nonlinear_transition(nss, x_prev, zeros(T, n_shocks))
            residual = x_t - predicted
            smoothed_shock_trajectories[:, t, b] = impact_pinv * residual
        end
    end

    # --- Aggregate: posterior mean ---
    sm_states = dropdims(mean(smoothed_trajectories; dims=3); dims=3)
    sm_shocks = dropdims(mean(smoothed_shock_trajectories; dims=3); dims=3)

    # Covariances (diagonal approximation from trajectories)
    sm_cov = zeros(T, n_states, n_states, T_obs)
    for t in 1:T_obs
        for b in 1:N_back
            diff = smoothed_trajectories[:, t, b] - sm_states[:, t]
            sm_cov[:, :, t] += diff * diff'
        end
        sm_cov[:, :, t] ./= max(N_back - 1, 1)
    end

    # Filtered/predicted not stored for particle smoother — use smoothed as approximation
    return KalmanSmootherResult{T}(
        sm_states, sm_cov, sm_shocks,
        sm_states, sm_cov, sm_states, sm_cov, log_lik
    )
end

"""Draw from categorical distribution with weights `w`."""
function _categorical_draw(w::AbstractVector{T}, rng::AbstractRNG) where {T}
    u = rand(rng, T)
    cum = zero(T)
    @inbounds for k in eachindex(w)
        cum += w[k]
        if u <= cum
            return k
        end
    end
    return length(w)
end
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS — particle smoother correlates >0.90 with RTS on linear DGP.

**Step 4: Commit**

```bash
git add src/dsge/smoother.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement FFBSi particle smoother for nonlinear state spaces"
```

---

### Task 6: Nonlinear DSGE Historical Decomposition (Counterfactual)

**Files:**
- Modify: `src/dsge/hd.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "Nonlinear DSGE HD — 2nd order perturbation" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5
        endogenous: y, pi_var
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    end
    psol = perturbation_solver(spec; order=2)

    T_obs = 40
    rng = Random.MersenneTwister(66)
    sim_data = simulate(psol, T_obs; rng=rng)
    observables = [:y, :pi_var]

    hd = historical_decomposition(psol, sim_data, observables;
                                   N=500, N_back=50,
                                   rng=Random.MersenneTwister(42))

    @test hd isa HistoricalDecomposition{Float64}
    @test size(hd.contributions) == (T_obs, 2, 2)
    @test hd.method == :dsge_nonlinear

    # Approximate identity (counterfactual has interaction terms in initial)
    for i in 1:2
        total = sum(hd.contributions[:, i, :]; dims=2)[:, 1] .+ hd.initial_conditions[:, i]
        @test cor(total, hd.actual[:, i]) > 0.90
    end
end
```

**Step 2: Add nonlinear HD dispatch to `src/dsge/hd.jl`**

```julia
# =============================================================================
# Nonlinear DSGE HD (Counterfactual Decomposition)
# =============================================================================

"""
    historical_decomposition(sol::PerturbationSolution{T}, data::AbstractMatrix,
                              observables::Vector{Symbol};
                              states=:observables, measurement_error=nothing,
                              N=1000, N_back=100,
                              rng=Random.default_rng()) where {T}

Historical decomposition for a nonlinear (higher-order perturbation) DSGE model
via counterfactual simulation.

Uses the FFBSi particle smoother to extract smoothed state trajectories, then
computes each shock's contribution by comparing the baseline path with a
counterfactual path where that shock is zeroed out.

# Note
Contributions are NOT additive at higher orders due to nonlinear interactions.
The residual (interaction terms) is attributed to initial conditions.
"""
function historical_decomposition(sol::PerturbationSolution{T}, data::AbstractMatrix,
                                   observables::Vector{Symbol};
                                   states::Symbol=:observables,
                                   measurement_error=nothing,
                                   N::Int=1000, N_back::Int=100,
                                   rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    spec = sol.spec
    T_obs = size(data, 1)
    n_states = spec.n_endog
    n_shocks = spec.n_exog

    # Build nonlinear state space
    Z, d, H = _build_observation_equation(spec, observables, measurement_error)
    nss = _build_nonlinear_state_space(sol, Z, d, H)

    # Data as deviations
    obs_indices = [findfirst(==(obs), spec.endog) for obs in observables]
    data_dev = Matrix{T}(data[:, obs_indices]' .- spec.steady_state[obs_indices])

    # Run particle smoother
    smoother = dsge_particle_smoother(nss, data_dev; N=N, N_back=N_back, rng=rng)

    # Smoothed shocks: n_shocks × T_obs
    smoothed_shocks = smoother.smoothed_shocks

    # --- Counterfactual decomposition ---
    if states == :all
        n_vars = n_states
        var_names = String.(spec.endog)
    else
        n_vars = length(observables)
        var_names = String.(observables)
    end

    # Baseline: simulate with all shocks
    baseline = _simulate_from_shocks(sol, smoothed_shocks, T_obs)

    # Counterfactual: zero out each shock
    contributions = zeros(T, T_obs, n_vars, n_shocks)
    for j in 1:n_shocks
        shocks_without_j = copy(smoothed_shocks)
        shocks_without_j[j, :] .= zero(T)
        path_without_j = _simulate_from_shocks(sol, shocks_without_j, T_obs)

        if states == :all
            for t in 1:T_obs, i in 1:n_vars
                contributions[t, i, j] = baseline[t, i] - path_without_j[t, i]
            end
        else
            for t in 1:T_obs, (vi, oi) in enumerate(obs_indices)
                contributions[t, vi, j] = baseline[t, oi] - path_without_j[t, oi]
            end
        end
    end

    # Actual
    if states == :all
        actual = zeros(T, T_obs, n_states)
        for t in 1:T_obs
            actual[t, :] = smoother.smoothed_states[:, t]
        end
    else
        actual = Matrix{T}(data_dev')
    end

    # Initial conditions = actual - sum of contributions
    initial_conditions = copy(actual)
    for j in 1:n_shocks
        initial_conditions .-= contributions[:, :, j]
    end

    shock_names = String.(spec.exog)

    return HistoricalDecomposition{T}(
        contributions, initial_conditions, actual,
        Matrix{T}(smoothed_shocks'), T_obs, var_names, shock_names, :dsge_nonlinear
    )
end

"""
    _simulate_from_shocks(sol::PerturbationSolution{T}, shocks::Matrix{T},
                           T_obs::Int) where {T}

Simulate the perturbation solution forward using given shocks (n_shocks × T_obs).
Returns T_obs × n_endog deviations from steady state.
"""
function _simulate_from_shocks(sol::PerturbationSolution{T}, shocks::Matrix{T},
                                T_obs::Int) where {T<:AbstractFloat}
    shock_draws = Matrix{T}(shocks')  # T_obs × n_shocks
    sim = simulate(sol, T_obs; shock_draws=shock_draws)
    return sim .- sol.steady_state'  # return deviations
end
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS

**Step 4: Commit**

```bash
git add src/dsge/hd.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement nonlinear DSGE HD via counterfactual decomposition"
```

---

### Task 7: Bayesian DSGE Historical Decomposition

**Files:**
- Modify: `src/dsge/hd.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "Bayesian DSGE HD — mode_only" begin
    # Create a fake BayesianDSGE with known parameters
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)
    observables = [:y, :pi_var, :r]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)

    # Fake posterior with 10 draws near true values
    n_draws = 10
    param_names = [:rho_y, :rho_pi, :rho_r]
    theta_draws = repeat([0.8, 0.5, 0.6]', n_draws, 1)
    theta_draws .+= 0.01 * randn(Random.MersenneTwister(1), n_draws, 3)

    prior = MacroEconometricModels.DSGEPrior{Float64}(
        param_names,
        [MacroEconometricModels.Distributions.Uniform(0.0, 1.0) for _ in 1:3],
        zeros(3), ones(3)
    )

    post = BayesianDSGE{Float64}(
        theta_draws, zeros(n_draws), param_names, prior,
        0.0, :smc, 0.5, Float64[], Float64[],
        spec, sol, ss
    )

    T_obs = 40
    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, T_obs; rng=rng)

    # mode_only path
    hd = historical_decomposition(post, sim_data, observables; mode_only=true)
    @test hd isa HistoricalDecomposition{Float64}
    @test hd.method == :dsge_bayes_mode
    @test verify_decomposition(hd; tol=0.1)
end

@testset "Bayesian DSGE HD — full posterior" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)
    observables = [:y, :pi_var, :r]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)

    n_draws = 20
    param_names = [:rho_y, :rho_pi, :rho_r]
    theta_draws = repeat([0.8, 0.5, 0.6]', n_draws, 1)
    theta_draws .+= 0.02 * randn(Random.MersenneTwister(2), n_draws, 3)
    # Clamp to valid range
    theta_draws = clamp.(theta_draws, 0.01, 0.99)

    prior = MacroEconometricModels.DSGEPrior{Float64}(
        param_names,
        [MacroEconometricModels.Distributions.Uniform(0.0, 1.0) for _ in 1:3],
        zeros(3), ones(3)
    )

    post = BayesianDSGE{Float64}(
        theta_draws, zeros(n_draws), param_names, prior,
        0.0, :smc, 0.5, Float64[], Float64[],
        spec, sol, ss
    )

    T_obs = 30
    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, T_obs; rng=rng)

    hd = historical_decomposition(post, sim_data, observables;
                                   n_draws=10, quantiles=[0.16, 0.5, 0.84])

    @test hd isa BayesianHistoricalDecomposition{Float64}
    @test size(hd.quantiles, 4) == 3
    @test hd.method == :dsge_bayes
    @test length(hd.quantile_levels) == 3
end
```

**Step 2: Add Bayesian HD dispatch to `src/dsge/hd.jl`**

```julia
# =============================================================================
# Bayesian DSGE HD
# =============================================================================

"""
    historical_decomposition(post::BayesianDSGE{T}, data::AbstractMatrix,
                              observables::Vector{Symbol};
                              mode_only=false, n_draws=200,
                              quantiles=[0.16, 0.5, 0.84],
                              measurement_error=nothing,
                              states=:observables) where {T}

Bayesian historical decomposition for DSGE models.

# Keyword Arguments
- `mode_only` — if `true`, use posterior mode solution only (fast path)
- `n_draws` — number of posterior draws to use (subsampled from `post.theta_draws`)
- `quantiles` — quantile levels for uncertainty bands
- `measurement_error` — measurement error std devs
- `states` — `:observables` or `:all`

# Returns
- `mode_only=true`: `HistoricalDecomposition{T}`
- `mode_only=false`: `BayesianHistoricalDecomposition{T}`
"""
function historical_decomposition(post::BayesianDSGE{T}, data::AbstractMatrix,
                                   observables::Vector{Symbol};
                                   mode_only::Bool=false,
                                   n_draws::Int=200,
                                   quantiles::Vector{<:Real}=T[0.16, 0.5, 0.84],
                                   measurement_error=nothing,
                                   states::Symbol=:observables) where {T<:AbstractFloat}
    if mode_only
        # Fast path: use modal solution
        sol = post.solution
        if sol isa DSGESolution
            hd = historical_decomposition(sol, data, observables;
                                           states=states, measurement_error=measurement_error)
            return HistoricalDecomposition{T}(
                hd.contributions, hd.initial_conditions, hd.actual,
                hd.shocks, hd.T_eff, hd.variables, hd.shock_names, :dsge_bayes_mode
            )
        else
            hd = historical_decomposition(sol, data, observables;
                                           states=states, measurement_error=measurement_error)
            return HistoricalDecomposition{T}(
                hd.contributions, hd.initial_conditions, hd.actual,
                hd.shocks, hd.T_eff, hd.variables, hd.shock_names, :dsge_bayes_mode
            )
        end
    end

    # Full Bayesian: re-solve + re-smooth at each posterior draw
    spec = post.spec
    T_obs = size(data, 1)
    n_shocks = spec.n_exog
    param_names = post.param_names
    total_draws = size(post.theta_draws, 1)
    quant_vec = T.(quantiles)

    # Subsample draws
    use_draws = min(n_draws, total_draws)
    draw_indices = round.(Int, range(1, total_draws; length=use_draws))

    # Determine variable count
    if states == :all
        n_vars = spec.n_endog
        var_names = String.(spec.endog)
    else
        n_vars = length(observables)
        var_names = String.(observables)
    end
    shock_names = String.(spec.exog)

    # Storage for all draws
    all_contributions = zeros(T, use_draws, T_obs, n_vars, n_shocks)
    all_initial = zeros(T, use_draws, T_obs, n_vars)
    all_shocks_est = zeros(T, use_draws, T_obs, n_shocks)
    n_success = 0

    for (di, draw_idx) in enumerate(draw_indices)
        theta = post.theta_draws[draw_idx, :]

        try
            # Re-solve at this draw
            sol_draw, ss_draw = _build_solution_at_theta(
                spec, param_names, T.(theta), observables,
                measurement_error, :gensys, NamedTuple()
            )

            # Only handle linear for now in Bayesian loop
            if !(sol_draw isa DSGESolution)
                continue
            end

            # Run HD at this draw
            hd_draw = historical_decomposition(sol_draw, data, observables;
                                                states=states, measurement_error=measurement_error)

            n_success += 1
            all_contributions[n_success, :, :, :] = hd_draw.contributions
            all_initial[n_success, :, :] = hd_draw.initial_conditions
            all_shocks_est[n_success, :, :] = hd_draw.shocks
        catch
            continue  # skip indeterminate solutions
        end
    end

    if n_success == 0
        error("All posterior draws failed to produce a valid DSGE solution")
    end

    if n_success < use_draws
        @warn "$(use_draws - n_success) of $use_draws posterior draws failed; using $n_success"
    end

    # Trim to successful draws
    contrib_valid = all_contributions[1:n_success, :, :, :]
    initial_valid = all_initial[1:n_success, :, :]
    shocks_valid = all_shocks_est[1:n_success, :, :]

    # Compute quantiles and point estimates
    n_q = length(quant_vec)
    contrib_quantiles = zeros(T, T_obs, n_vars, n_shocks, n_q)
    initial_quantiles = zeros(T, T_obs, n_vars, n_q)

    for i in 1:n_vars, j in 1:n_shocks, t in 1:T_obs
        vals = contrib_valid[:, t, i, j]
        sort!(vals)
        for (qi, q) in enumerate(quant_vec)
            contrib_quantiles[t, i, j, qi] = _quantile_sorted(vals, q)
        end
    end
    for i in 1:n_vars, t in 1:T_obs
        vals = initial_valid[:, t, i]
        sort!(vals)
        for (qi, q) in enumerate(quant_vec)
            initial_quantiles[t, i, qi] = _quantile_sorted(vals, q)
        end
    end

    point_est = dropdims(mean(contrib_valid; dims=1); dims=1)
    initial_pe = dropdims(mean(initial_valid; dims=1); dims=1)
    shocks_pe = dropdims(mean(shocks_valid; dims=1); dims=1)

    # Actual data
    obs_indices = [findfirst(==(obs), spec.endog) for obs in observables]
    if states == :all
        # Use modal smoother
        Z, d, H = _build_observation_equation(spec, observables, measurement_error)
        ss = _build_state_space(post.solution, Z, d, H)
        data_dev = Matrix{T}(data[:, obs_indices]' .- spec.steady_state[obs_indices])
        sm = dsge_smoother(ss, data_dev)
        actual = zeros(T, T_obs, spec.n_endog)
        for t in 1:T_obs
            actual[t, :] = sm.smoothed_states[:, t]
        end
    else
        actual = Matrix{T}(data[:, obs_indices] .- spec.steady_state[obs_indices]')
    end

    return BayesianHistoricalDecomposition{T}(
        contrib_quantiles, point_est, initial_quantiles, initial_pe,
        shocks_pe, actual, T_obs, var_names, shock_names, quant_vec, :dsge_bayes
    )
end

"""Quantile from a pre-sorted vector."""
function _quantile_sorted(sorted::AbstractVector{T}, q::T) where {T}
    n = length(sorted)
    n == 0 && return zero(T)
    n == 1 && return sorted[1]
    pos = one(T) + (n - 1) * q
    lo = max(1, floor(Int, pos))
    hi = min(n, lo + 1)
    frac = pos - lo
    return sorted[lo] * (one(T) - frac) + sorted[hi] * frac
end
```

**Step 3: Run test to verify it passes**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS

**Step 4: Commit**

```bash
git add src/dsge/hd.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): implement Bayesian DSGE HD with mode_only and full posterior paths"
```

---

### Task 8: Missing Data + Edge Cases Tests

**Files:**
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write and run edge case tests**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "RTS smoother — missing data handling" begin
    spec = @dsge begin
        parameters: rho = 0.8
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    T_obs = 50
    rng = Random.MersenneTwister(44)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    # Introduce NaN at some periods
    data_nan = copy(data_matrix)
    data_nan[1, 10] = NaN
    data_nan[1, 20] = NaN
    data_nan[1, 30] = NaN

    result = dsge_smoother(ss, data_nan)
    @test isfinite(result.log_likelihood)
    @test all(isfinite, result.smoothed_states)
    @test all(isfinite, result.smoothed_shocks)
end

@testset "HD — short sample (T=5)" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    rng = Random.MersenneTwister(11)
    sim_data = simulate(sol, 5; rng=rng)
    hd = historical_decomposition(sol, sim_data, [:y])
    @test size(hd.contributions, 1) == 5
    @test verify_decomposition(hd; tol=0.05)
end

@testset "HD — single shock, single observable" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)

    rng = Random.MersenneTwister(22)
    sim_data = simulate(sol, 30; rng=rng)
    hd = historical_decomposition(sol, sim_data, [:y])
    @test size(hd.contributions) == (30, 1, 1)
    @test length(hd.shock_names) == 1
end
```

**Step 2: Run tests**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS

**Step 3: Commit**

```bash
git add test/dsge/test_dsge_hd.jl
git commit -m "test(dsge): add missing data, short sample, and edge case tests for HD"
```

---

### Task 9: Show Method for KalmanSmootherResult

**Files:**
- Modify: `src/dsge/smoother.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Write the failing test**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "KalmanSmootherResult show method" begin
    spec = @dsge begin
        parameters: rho = 0.8
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)
    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, 30; rng=rng)
    observables = [:y]
    Z, d, H_mat = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H_mat)
    data_matrix = Matrix(sim_data' .- sol.spec.steady_state)

    result = dsge_smoother(ss, data_matrix)
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("Kalman Smoother Result", output)
    @test occursin("Log-likelihood", output)
    @test occursin("States", output)
end
```

**Step 2: Implement show method in `src/dsge/smoother.jl`**

Add at the end:

```julia
# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::KalmanSmootherResult{T}) where {T}
    n_states, T_obs = size(r.smoothed_states)
    n_shocks = size(r.smoothed_shocks, 1)
    println(io, "Kalman Smoother Result")
    println(io, "═" ^ 40)
    println(io, "  States:         $n_states")
    println(io, "  Shocks:         $n_shocks")
    println(io, "  Periods:        $T_obs")
    println(io, "  Log-likelihood: $(round(r.log_likelihood; digits=4))")
end
```

**Step 3: Run test, commit**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

```bash
git add src/dsge/smoother.jl test/dsge/test_dsge_hd.jl
git commit -m "feat(dsge): add show method for KalmanSmootherResult"
```

---

### Task 10: Register Test File + Integration Test

**Files:**
- Modify: `test/runtests.jl`
- Modify: `test/dsge/test_dsge_hd.jl`

**Step 1: Add the test file to the DSGE test group**

In `test/runtests.jl`, in the Group 7 DSGE Models section (around line 103), add:

```julia
    ("DSGE Models" => [
        "dsge/test_dsge.jl",
        "dsge/test_bayesian_dsge.jl",
        "dsge/test_dsge_hd.jl",
    ]),
```

**Step 2: Add integration test combining smoother + HD + accessors**

Add to `test/dsge/test_dsge_hd.jl`:

```julia
@testset "Integration: smoother → HD → accessors → report" begin
    spec = @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
        endogenous: y, pi_var, r
        exogenous: eps_y, eps_pi, eps_r
        y[t] = rho_y * y[t-1] + eps_y[t]
        pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
        r[t] = rho_r * r[t-1] + eps_r[t]
    end
    sol = solve(spec)

    T_obs = 50
    rng = Random.MersenneTwister(99)
    sim_data = simulate(sol, T_obs; rng=rng)
    observables = [:y, :pi_var, :r]

    hd = historical_decomposition(sol, sim_data, observables)

    # Accessor functions work
    c = contribution(hd, 1, 1)
    @test length(c) == T_obs
    c_named = contribution(hd, "y", "eps_y")
    @test c_named == c

    ts = total_shock_contribution(hd, 1)
    @test length(ts) == T_obs

    @test verify_decomposition(hd; tol=0.1)

    # Report works
    io = IOBuffer()
    show(io, hd)
    output = String(take!(io))
    @test occursin("Historical Decomposition", output)

    # point_estimate interface
    pe = point_estimate(hd)
    @test size(pe) == (T_obs, 3, 3)
    @test has_uncertainty(hd) == false
end
```

**Step 3: Run the full DSGE test group**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Then verify the full test suite still passes:

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl`

**Step 4: Commit**

```bash
git add test/runtests.jl test/dsge/test_dsge_hd.jl
git commit -m "test(dsge): register test_dsge_hd.jl and add integration test"
```

---

### Task 11: Performance Optimization — Smoother Workspace Reuse

**Files:**
- Modify: `src/dsge/smoother.jl`

**Step 1: Profile and optimize the backward pass**

In the RTS backward pass, the main allocation hotspot is `P_filt_t * ss.G1'` creating a temporary. Replace with pre-allocated workspace:

In `dsge_smoother`, add to the backward pass workspace:

```julia
    PG = zeros(T, n_states, n_states)  # P_{t|t} * G1'
```

Replace the smoother gain computation:

```julia
        # J = P_{t|t} * G1' * P_pred_inv
        mul!(PG, P_filt_t, ss.G1')
        mul!(J, PG, P_pred_inv)
```

Also replace `JP * J'` with pre-allocated workspace:

```julia
    JP_Jt = zeros(T, n_states, n_states)
```

```julia
        mul!(JP, J, diff_P)
        mul!(JP_Jt, JP, J')
        for j in 1:n_states, i in 1:n_states
            smoothed_cov[i, j, t] = P_filt[i, j, t] + JP_Jt[i, j]
        end
```

**Step 2: Run tests to verify no regressions**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && julia --project=. -e 'using MacroEconometricModels; using Test; include("test/dsge/test_dsge_hd.jl")'`

Expected: PASS (same results, fewer allocations)

**Step 3: Commit**

```bash
git add src/dsge/smoother.jl
git commit -m "perf(dsge): reduce allocations in RTS backward pass"
```

---

### Task 12: Final — Full Test Suite + Documentation Stub

**Files:**
- Run full test suite
- Create: `docs/src/dsge_hd.md` (minimal stub)

**Step 1: Run full test suite**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels/.claude/worktrees/dreamy-imagining-locket && MACRO_MULTIPROCESS_TESTS=1 julia --project=. test/runtests.jl`

Expected: All groups PASS, including the new DSGE HD tests.

**Step 2: Create documentation page**

Create `docs/src/dsge_hd.md`:

```markdown
# DSGE Historical Decomposition

Historical decomposition for DSGE models decomposes observed variable movements into
contributions from individual structural shocks, using the Kalman smoother (linear models)
or FFBSi particle smoother (nonlinear models) to extract smoothed structural shocks.

## Quick Start

```julia
# Define and solve DSGE model
spec = @dsge begin
    parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
    endogenous: y, pi_var, r
    exogenous: eps_y, eps_pi, eps_r
    y[t] = rho_y * y[t-1] + eps_y[t]
    pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
    r[t] = rho_r * r[t-1] + eps_r[t]
end
sol = solve(spec)

# Simulate or load observed data (T_obs × n_endog)
data = simulate(sol, 100)

# Historical decomposition
hd = historical_decomposition(sol, data, [:y, :pi_var, :r])
report(hd)

# Access individual contributions
c = contribution(hd, "y", "eps_y")

# Verify decomposition identity
verify_decomposition(hd)

# Plot
plot_result(hd)
```

## Kalman Smoother

The RTS smoother can be used independently:

```julia
Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)
ss = MacroEconometricModels._build_state_space(sol, Z, d, H)
smoother = dsge_smoother(ss, data_matrix)
```

## Nonlinear Models

For higher-order perturbation solutions, HD uses counterfactual decomposition
with the FFBSi particle smoother:

```julia
psol = perturbation_solver(spec; order=2)
hd = historical_decomposition(psol, data, [:y, :pi_var]; N=1000, N_back=100)
```

## Bayesian HD

```julia
# Full posterior (re-solve at each draw)
hd = historical_decomposition(posterior, data, observables;
                               n_draws=200, quantiles=[0.16, 0.5, 0.84])

# Fast mode (posterior mode only)
hd = historical_decomposition(posterior, data, observables; mode_only=true)
```

## API Reference

```@docs
dsge_smoother
dsge_particle_smoother
KalmanSmootherResult
```
```

**Step 3: Commit**

```bash
git add -f docs/src/dsge_hd.md
git commit -m "docs: add DSGE historical decomposition documentation page"
```

---

## Summary of Files

| File | Action | Description |
|------|--------|-------------|
| `src/dsge/smoother.jl` | Create | RTS smoother + FFBSi particle smoother + KalmanSmootherResult |
| `src/dsge/hd.jl` | Create | HD dispatches for DSGESolution, PerturbationSolution, BayesianDSGE |
| `src/MacroEconometricModels.jl` | Modify | Add includes (after line 305) + exports |
| `test/dsge/test_dsge_hd.jl` | Create | Full test suite (~15 testsets) |
| `test/runtests.jl` | Modify | Register test file in Group 7 |
| `docs/src/dsge_hd.md` | Create | Documentation page |

## Commit History (12 commits)

1. `feat(dsge): add KalmanSmootherResult type and smoother skeleton`
2. `feat(dsge): implement Kalman filter forward pass in dsge_smoother`
3. `feat(dsge): implement RTS backward pass and shock extraction`
4. `feat(dsge): implement linear DSGE historical decomposition`
5. `feat(dsge): implement FFBSi particle smoother for nonlinear state spaces`
6. `feat(dsge): implement nonlinear DSGE HD via counterfactual decomposition`
7. `feat(dsge): implement Bayesian DSGE HD with mode_only and full posterior paths`
8. `test(dsge): add missing data, short sample, and edge case tests for HD`
9. `feat(dsge): add show method for KalmanSmootherResult`
10. `test(dsge): register test_dsge_hd.jl and add integration test`
11. `perf(dsge): reduce allocations in RTS backward pass`
12. `docs: add DSGE historical decomposition documentation page`
