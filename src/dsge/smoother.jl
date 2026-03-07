# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Kalman smoother and particle smoother for DSGE models.

Provides:
- `KalmanSmootherResult` — result type storing smoothed/filtered/predicted states, covariances, and shocks
- `dsge_smoother` — Rauch-Tung-Striebel (RTS) fixed-interval smoother for linear DSGE models
- `dsge_particle_smoother` — FFBSi particle smoother for nonlinear DSGE models

References:
- Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood estimates of linear
  dynamic systems. AIAA Journal, 3(8), 1445-1450.
- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for nonlinear
  time series. Journal of the American Statistical Association, 99(465), 156-168.
- Herbst, E. & Schorfheide, F. (2015). Bayesian Estimation of DSGE Models. Princeton University Press.
"""

using LinearAlgebra, Random

# =============================================================================
# Result type
# =============================================================================

"""
    KalmanSmootherResult{T<:AbstractFloat} <: AbstractAnalysisResult

Result of a Kalman smoother (or particle smoother) applied to a DSGE model.

# Fields
- `smoothed_states::Matrix{T}` — smoothed state means (n_states × T_obs)
- `smoothed_covariances::Array{T,3}` — smoothed state covariances (n_states × n_states × T_obs)
- `smoothed_shocks::Matrix{T}` — smoothed structural shocks (n_shocks × T_obs)
- `filtered_states::Matrix{T}` — filtered state means (n_states × T_obs)
- `filtered_covariances::Array{T,3}` — filtered state covariances (n_states × n_states × T_obs)
- `predicted_states::Matrix{T}` — one-step-ahead predicted state means (n_states × T_obs)
- `predicted_covariances::Array{T,3}` — one-step-ahead predicted state covariances (n_states × n_states × T_obs)
- `log_likelihood::T` — log-likelihood from the forward pass (prediction error decomposition)
"""
struct KalmanSmootherResult{T<:AbstractFloat} <: AbstractAnalysisResult
    smoothed_states::Matrix{T}
    smoothed_covariances::Array{T,3}
    smoothed_shocks::Matrix{T}
    filtered_states::Matrix{T}
    filtered_covariances::Array{T,3}
    predicted_states::Matrix{T}
    predicted_covariances::Array{T,3}
    log_likelihood::T
end

# =============================================================================
# Smoother stubs
# =============================================================================

"""
    dsge_smoother(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T}

Rauch-Tung-Striebel (RTS) fixed-interval smoother for linear DSGE state space models.

Runs a forward Kalman filter pass followed by a backward smoothing pass to produce
optimal state estimates conditional on the full sample.

# Arguments
- `ss` — `DSGEStateSpace{T}` with transition/observation matrices
- `data` — `n_obs × T_obs` matrix of observables (each column is one time period)

# Returns
A `KalmanSmootherResult{T}` containing smoothed states, covariances, shocks,
filtered quantities, predicted quantities, and log-likelihood.
"""
function dsge_smoother(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T<:AbstractFloat}
    n_obs, T_obs = size(data)
    n_states = size(ss.G1, 1)
    n_shocks = size(ss.impact, 2)

    # =====================================================================
    # Initialization
    # =====================================================================
    x = zeros(T, n_states)  # initial state mean (deviation from SS = 0)

    # Stationary covariance via Lyapunov equation; diffuse fallback
    RQR = ss.impact * ss.Q * ss.impact'
    P = try
        solve_lyapunov(ss.G1, ss.impact)
    catch
        T(10) * Matrix{T}(I, n_states, n_states)
    end

    # =====================================================================
    # Storage arrays for smoother
    # =====================================================================
    x_pred_store = zeros(T, n_states, T_obs)       # predicted states
    P_pred_store = zeros(T, n_states, n_states, T_obs)  # predicted covariances
    x_filt_store = zeros(T, n_states, T_obs)       # filtered states
    P_filt_store = zeros(T, n_states, n_states, T_obs)  # filtered covariances

    # =====================================================================
    # Pre-allocate workspace (same pattern as _kalman_loglikelihood)
    # =====================================================================
    # Prediction step
    x_pred = zeros(T, n_states)
    P_pred = zeros(T, n_states, n_states)
    G1P    = zeros(T, n_states, n_states)  # G1 * P

    # Innovation
    v = zeros(T, n_obs)          # innovation y_t - Z*x_pred - d
    Zx = zeros(T, n_obs)        # Z * x_pred

    # Innovation covariance F = Z * P_pred * Z' + H
    F = zeros(T, n_obs, n_obs)
    ZP = zeros(T, n_obs, n_states)   # Z * P_pred
    PZ = zeros(T, n_states, n_obs)   # P_pred * Z'

    # Kalman gain K = P_pred * Z' * F^{-1}
    K = zeros(T, n_states, n_obs)
    F_inv = zeros(T, n_obs, n_obs)
    F_inv_v = zeros(T, n_obs)

    # Update temporaries
    Kv = zeros(T, n_states)     # K * v
    KZP = zeros(T, n_states, n_states)  # K * Z * P_pred

    # Check if data has any NaN
    has_nan = any(isnan, data)

    ll = zero(T)
    log2pi = T(log(2 * T(pi)))

    # =====================================================================
    # Forward pass (Kalman filter) — mirrors _kalman_loglikelihood exactly
    # =====================================================================
    @inbounds for t in 1:T_obs
        # --- Prediction step ---
        # x_pred = G1 * x
        mul!(x_pred, ss.G1, x)
        # P_pred = G1 * P * G1' + impact * Q * impact'
        mul!(G1P, ss.G1, P)
        mul!(P_pred, G1P, ss.G1')
        @inbounds for j in 1:n_states, i in 1:n_states
            P_pred[i, j] += RQR[i, j]
        end

        # Store predicted quantities
        @inbounds for i in 1:n_states
            x_pred_store[i, t] = x_pred[i]
        end
        @inbounds for j in 1:n_states, i in 1:n_states
            P_pred_store[i, j, t] = P_pred[i, j]
        end

        # Get observation for this time step
        y_t = @view data[:, t]

        # Check for missing data in this period
        if has_nan && any(isnan, y_t)
            # --- Partial observation path ---
            obs_mask = .!isnan.(y_t)
            n_obs_t = count(obs_mask)

            if n_obs_t == 0
                # All missing: just propagate prediction, no likelihood contribution
                copyto!(x, x_pred)
                copyto!(P, P_pred)
                # Store filtered = predicted when no observations
                @inbounds for i in 1:n_states
                    x_filt_store[i, t] = x_pred[i]
                end
                @inbounds for j in 1:n_states, i in 1:n_states
                    P_filt_store[i, j, t] = P_pred[i, j]
                end
                continue
            end

            # Extract observed subset
            obs_idx = findall(obs_mask)
            Z_t = ss.Z[obs_idx, :]
            d_t = ss.d[obs_idx]
            H_t = ss.H[obs_idx, obs_idx]
            y_obs = y_t[obs_idx]

            # Innovation
            v_t = y_obs - Z_t * x_pred - d_t

            # Innovation covariance
            F_t = Z_t * P_pred * Z_t' + H_t
            F_t = (F_t + F_t') / 2  # symmetrize

            # Log-likelihood contribution
            F_chol = cholesky(Hermitian(F_t); check=false)
            if !issuccess(F_chol)
                # Fallback: add jitter
                F_t += T(1e-8) * I
                F_chol = cholesky(Hermitian(F_t))
            end
            log_det_F = logdet(F_chol)
            F_inv_t = inv(F_chol)
            F_inv_v_t = F_inv_t * v_t

            ll += -T(0.5) * (n_obs_t * log2pi + log_det_F + dot(v_t, F_inv_v_t))

            # Kalman gain
            K_t = P_pred * Z_t' * F_inv_t

            # Update
            x .= x_pred .+ K_t * v_t
            P .= P_pred .- K_t * Z_t * P_pred
            P .= (P .+ P') ./ 2  # symmetrize

            # Store filtered quantities
            @inbounds for i in 1:n_states
                x_filt_store[i, t] = x[i]
            end
            @inbounds for j in 1:n_states, i in 1:n_states
                P_filt_store[i, j, t] = P[i, j]
            end
        else
            # --- Full observation path (pre-allocated, zero allocation) ---

            # Innovation: v = y_t - Z * x_pred - d
            mul!(Zx, ss.Z, x_pred)
            @inbounds for i in 1:n_obs
                v[i] = y_t[i] - Zx[i] - ss.d[i]
            end

            # Innovation covariance: F = Z * P_pred * Z' + H
            mul!(ZP, ss.Z, P_pred)
            mul!(F, ZP, ss.Z')
            @inbounds for j in 1:n_obs, i in 1:n_obs
                F[i, j] += ss.H[i, j]
            end
            # Symmetrize F
            @inbounds for j in 1:n_obs, i in 1:j-1
                avg = (F[i, j] + F[j, i]) / 2
                F[i, j] = avg
                F[j, i] = avg
            end

            # Cholesky of F for stable inversion + log determinant
            F_chol = cholesky(Hermitian(F); check=false)
            if !issuccess(F_chol)
                @inbounds for i in 1:n_obs
                    F[i, i] += T(1e-8)
                end
                F_chol = cholesky(Hermitian(F))
            end
            log_det_F = logdet(F_chol)
            # F_inv = inv(F_chol) into pre-allocated buffer
            copyto!(F_inv, inv(F_chol))

            # F_inv * v
            mul!(F_inv_v, F_inv, v)

            # Log-likelihood contribution
            ll += -T(0.5) * (n_obs * log2pi + log_det_F + dot(v, F_inv_v))

            # Kalman gain: K = P_pred * Z' * F_inv
            mul!(PZ, P_pred, ss.Z')
            mul!(K, PZ, F_inv)

            # State update: x = x_pred + K * v
            mul!(Kv, K, v)
            @inbounds for i in 1:n_states
                x[i] = x_pred[i] + Kv[i]
            end

            # Covariance update: P = P_pred - K * Z * P_pred
            mul!(KZP, K, ZP)   # K * (Z * P_pred) — ZP already computed above
            @inbounds for j in 1:n_states, i in 1:n_states
                P[i, j] = P_pred[i, j] - KZP[i, j]
            end
            # Symmetrize P
            @inbounds for j in 1:n_states, i in 1:j-1
                avg = (P[i, j] + P[j, i]) / 2
                P[i, j] = avg
                P[j, i] = avg
            end

            # Store filtered quantities
            @inbounds for i in 1:n_states
                x_filt_store[i, t] = x[i]
            end
            @inbounds for j in 1:n_states, i in 1:n_states
                P_filt_store[i, j, t] = P[i, j]
            end
        end
    end

    # =====================================================================
    # Backward pass (RTS smoother)
    # =====================================================================
    x_smooth = zeros(T, n_states, T_obs)
    P_smooth = zeros(T, n_states, n_states, T_obs)

    # Initialize at T_obs: smoothed = filtered
    @inbounds for i in 1:n_states
        x_smooth[i, T_obs] = x_filt_store[i, T_obs]
    end
    @inbounds for j in 1:n_states, i in 1:n_states
        P_smooth[i, j, T_obs] = P_filt_store[i, j, T_obs]
    end

    # Pre-allocate backward pass workspace
    J_t = zeros(T, n_states, n_states)       # smoother gain
    P_filt_t = zeros(T, n_states, n_states)  # P_{t|t}
    P_pred_tp1 = zeros(T, n_states, n_states) # P_{t+1|t}
    P_smooth_tp1 = zeros(T, n_states, n_states) # P_{t+1|T}
    x_diff = zeros(T, n_states)              # x_{t+1|T} - x_{t+1|t}
    P_diff = zeros(T, n_states, n_states)    # P_{t+1|T} - P_{t+1|t}
    JP = zeros(T, n_states, n_states)        # J_t * P_diff
    P_filt_G1t = zeros(T, n_states, n_states)  # P_{t|t} * G1'

    @inbounds for t in (T_obs - 1):-1:1
        # Load filtered covariance at t
        for j in 1:n_states, i in 1:n_states
            P_filt_t[i, j] = P_filt_store[i, j, t]
        end
        # Load predicted covariance at t+1
        for j in 1:n_states, i in 1:n_states
            P_pred_tp1[i, j] = P_pred_store[i, j, t + 1]
        end

        # Smoother gain: J_t = P_{t|t} * G1' * inv(P_{t+1|t})
        mul!(P_filt_G1t, P_filt_t, ss.G1')
        # Use Cholesky for stable inversion of P_{t+1|t}
        P_pred_chol = cholesky(Hermitian(P_pred_tp1); check=false)
        if !issuccess(P_pred_chol)
            # Fallback: add jitter
            for i in 1:n_states
                P_pred_tp1[i, i] += T(1e-8)
            end
            P_pred_chol = cholesky(Hermitian(P_pred_tp1))
        end
        P_pred_inv = inv(P_pred_chol)
        mul!(J_t, P_filt_G1t, Matrix{T}(P_pred_inv))

        # Smoothed state: x_{t|T} = x_{t|t} + J_t * (x_{t+1|T} - x_{t+1|t})
        for i in 1:n_states
            x_diff[i] = x_smooth[i, t + 1] - x_pred_store[i, t + 1]
        end
        # x_smooth[:, t] = x_filt[:, t] + J_t * x_diff
        mul!(Kv, J_t, x_diff)  # reuse Kv workspace
        for i in 1:n_states
            x_smooth[i, t] = x_filt_store[i, t] + Kv[i]
        end

        # Smoothed covariance: P_{t|T} = P_{t|t} + J_t * (P_{t+1|T} - P_{t+1|t}) * J_t'
        for j in 1:n_states, i in 1:n_states
            P_smooth_tp1[i, j] = P_smooth[i, j, t + 1]
        end
        for j in 1:n_states, i in 1:n_states
            P_diff[i, j] = P_smooth_tp1[i, j] - P_pred_store[i, j, t + 1]
        end
        mul!(JP, J_t, P_diff)
        mul!(P_smooth_tp1, JP, J_t')  # reuse P_smooth_tp1 as temporary
        for j in 1:n_states, i in 1:n_states
            P_smooth[i, j, t] = P_filt_t[i, j] + P_smooth_tp1[i, j]
        end
        # Symmetrize P_{t|T}
        for j in 1:n_states, i in 1:j-1
            avg = (P_smooth[i, j, t] + P_smooth[j, i, t]) / 2
            P_smooth[i, j, t] = avg
            P_smooth[j, i, t] = avg
        end
    end

    # =====================================================================
    # Shock extraction: ε_t = pinv(impact) * (x_{t|T} - G1 * x_{t-1|T})
    # =====================================================================
    smoothed_shocks = zeros(T, n_shocks, T_obs)
    impact_pinv = pinv(ss.impact)
    x_prev = zeros(T, n_states)  # x_{0|T} = 0
    G1x = zeros(T, n_states)     # workspace for G1 * x_prev

    @inbounds for t in 1:T_obs
        mul!(G1x, ss.G1, x_prev)
        for i in 1:n_states
            x_diff[i] = x_smooth[i, t] - G1x[i]
        end
        # smoothed_shocks[:, t] = impact_pinv * x_diff
        mul!(@view(smoothed_shocks[:, t]), impact_pinv, x_diff)
        # Update x_prev for next iteration
        for i in 1:n_states
            x_prev[i] = x_smooth[i, t]
        end
    end

    return KalmanSmootherResult{T}(
        x_smooth, P_smooth, smoothed_shocks,
        x_filt_store, P_filt_store,
        x_pred_store, P_pred_store,
        ll
    )
end

"""
    dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                            n_particles::Int=1000, rng::AbstractRNG=Random.default_rng()) where {T}

Forward-filtering backward-simulation (FFBSi) particle smoother for nonlinear DSGE models.

Uses a bootstrap particle filter forward pass followed by backward simulation to
produce smoothed state trajectories for higher-order perturbation solutions.

# Arguments
- `nss` — `NonlinearStateSpace{T}` from a higher-order perturbation solution
- `data` — `n_obs × T_obs` matrix of observables
- `n_particles` — number of particles (default: 1000)
- `rng` — random number generator (default: `Random.default_rng()`)

# Returns
A `KalmanSmootherResult{T}` with smoothed states and shocks (covariances from particle
approximation).
"""
function dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                                 n_particles::Int=1000,
                                 rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    error("dsge_particle_smoother is not yet implemented — see Task 5 of the DSGE HD plan")
end
