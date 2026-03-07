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

# =============================================================================
# Helper: single-particle nonlinear (pruned) transition
# =============================================================================

"""
    _nonlinear_transition(nss::NonlinearStateSpace{T}, xf_prev::AbstractVector{T},
                           xs_prev::AbstractVector{T}, xrd_prev::AbstractVector{T},
                           eps_t::AbstractVector{T}) where {T}

Evaluate one step of the pruned nonlinear transition for a single particle.

Returns `(xf_new, xs_new, xrd_new, x_total, y_t)` — the updated first-order state,
second-order correction, third-order correction, total state, and control vector.

Uses the same formulas as `simulate(::PerturbationSolution, ...)` in pruning.jl.
"""
function _nonlinear_transition(nss::NonlinearStateSpace{T},
                                xf_prev::AbstractVector{T},
                                xs_prev::AbstractVector{T},
                                xrd_prev::AbstractVector{T},
                                eps_t::AbstractVector{T}) where {T<:AbstractFloat}
    nx = length(nss.state_indices)
    ny = length(nss.control_indices)
    n_eps = size(nss.eta, 2)
    nv = nx + n_eps

    # Extract first-order blocks
    hx_state = @view nss.hx[:, 1:nx]       # nx x nx
    eta_x = @view nss.hx[:, nx+1:nv]       # nx x n_eps
    gx_state = @view nss.gx[:, 1:nx]       # ny x nx
    eta_y = @view nss.gx[:, nx+1:nv]       # ny x n_eps

    # First-order state
    xf_new = hx_state * xf_prev + eta_x * eps_t

    # Build augmented vector vf = [xf_prev; eps_t]
    vf = zeros(T, nv)
    if nx > 0
        vf[1:nx] = xf_prev
    end
    vf[nx+1:nv] = eps_t

    xs_new = zeros(T, nx)
    xrd_new = zeros(T, nx)

    if nss.order >= 2
        kron_vf = kron(vf, vf)

        # Second-order state correction
        xs_new = hx_state * xs_prev
        if nss.hxx !== nothing && !isempty(kron_vf)
            xs_new += T(0.5) * nss.hxx * kron_vf
        end
        if nss.hsigmasigma !== nothing
            xs_new += T(0.5) * nss.hsigmasigma
        end

        if nss.order >= 3
            # Build vs = [xs_prev; 0]
            vs = zeros(T, nv)
            if nx > 0
                vs[1:nx] = xs_prev
            end
            kron_vf_vs = kron(vf, vs)
            kron3_vf = kron(vf, kron_vf)

            xrd_new = hx_state * xrd_prev
            if nss.hxx !== nothing && !isempty(kron_vf_vs)
                xrd_new += nss.hxx * kron_vf_vs
            end
            if nss.hxxx !== nothing && !isempty(kron3_vf)
                xrd_new += (one(T) / T(6)) * nss.hxxx * kron3_vf
            end
            if nss.hsigmax !== nothing
                xrd_new += T(0.5) * nss.hsigmax * vf
            end
            if nss.hsigmasigmasigma !== nothing
                xrd_new += (one(T) / T(6)) * nss.hsigmasigmasigma
            end
        end
    end

    # Total state
    x_total = xf_new + xs_new + xrd_new

    # Controls
    y_t = gx_state * x_total + eta_y * eps_t

    if nss.order >= 2 && nss.gxx !== nothing
        # Build vf_new = [xf_new; eps_t] for control Kronecker products
        vf_new = zeros(T, nv)
        if nx > 0
            vf_new[1:nx] = xf_new
        end
        vf_new[nx+1:nv] = eps_t
        kron_vf_new = kron(vf_new, vf_new)

        if nss.order >= 3
            vs_new = zeros(T, nv)
            if nx > 0
                vs_new[1:nx] = xs_new
            end
            kron_vfvs_new = kron(vf_new, vs_new)
            y_t += T(0.5) * nss.gxx * (kron_vf_new + T(2) * kron_vfvs_new)
        else
            y_t += T(0.5) * nss.gxx * kron_vf_new
        end

        if nss.gsigmasigma !== nothing
            y_t += T(0.5) * nss.gsigmasigma
        end

        if nss.order >= 3
            vf_new = zeros(T, nv)
            if nx > 0
                vf_new[1:nx] = xf_new
            end
            vf_new[nx+1:nv] = eps_t
            kron_vf_new2 = kron(vf_new, vf_new)
            kron3_vf_new = kron(vf_new, kron_vf_new2)

            if nss.gxxx !== nothing
                y_t += (one(T) / T(6)) * nss.gxxx * kron3_vf_new
            end
            if nss.gsigmax !== nothing
                y_t += T(0.5) * nss.gsigmax * vf_new
            end
            if nss.gsigmasigmasigma !== nothing
                y_t += (one(T) / T(6)) * nss.gsigmasigmasigma
            end
        end
    end

    return xf_new, xs_new, xrd_new, x_total, y_t
end

"""
    _nonlinear_transition_full(nss::NonlinearStateSpace{T}, xf_prev, xs_prev, xrd_prev,
                                eps_t) -> full_state_vector

Evaluate one step and return the full endogenous vector (states + controls) assembled
at the correct indices.
"""
function _nonlinear_transition_full(nss::NonlinearStateSpace{T},
                                     xf_prev::AbstractVector{T},
                                     xs_prev::AbstractVector{T},
                                     xrd_prev::AbstractVector{T},
                                     eps_t::AbstractVector{T}) where {T<:AbstractFloat}
    xf_new, xs_new, xrd_new, x_total, y_t = _nonlinear_transition(nss, xf_prev, xs_prev, xrd_prev, eps_t)
    nx = length(nss.state_indices)
    ny = length(nss.control_indices)
    n_endog = nx + ny
    full = zeros(T, n_endog)
    for (k, si) in enumerate(nss.state_indices)
        full[si] = x_total[k]
    end
    for (k, ci) in enumerate(nss.control_indices)
        full[ci] = y_t[k]
    end
    return full, xf_new, xs_new, xrd_new
end

# =============================================================================
# Helper: categorical draw
# =============================================================================

"""
    _categorical_draw(w::AbstractVector{T}, rng::AbstractRNG) where {T}

Draw one index from a categorical distribution with weights `w` (must sum to 1).
Uses a simple linear scan.
"""
function _categorical_draw(w::AbstractVector{T}, rng::AbstractRNG) where {T<:AbstractFloat}
    u = rand(rng, T)
    cumw = zero(T)
    @inbounds for i in eachindex(w)
        cumw += w[i]
        if u <= cumw
            return i
        end
    end
    return length(w)  # fallback for floating-point edge case
end

# =============================================================================
# FFBSi particle smoother
# =============================================================================

"""
    dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                            N::Int=1000, N_back::Int=100,
                            rng::AbstractRNG=Random.default_rng()) where {T}

Forward-filtering backward-simulation (FFBSi) particle smoother for nonlinear DSGE models.

Uses a bootstrap particle filter forward pass followed by Godsill-Doucet-West (2004)
backward simulation to produce smoothed state trajectories for higher-order perturbation
solutions.

# Arguments
- `nss` — `NonlinearStateSpace{T}` from a higher-order perturbation solution
- `data` — `n_obs × T_obs` matrix of observables (deviations from steady state)
- `N` — number of forward particles (default: 1000)
- `N_back` — number of backward trajectories (default: 100)
- `rng` — random number generator (default: `Random.default_rng()`)

# Returns
A `KalmanSmootherResult{T}` with smoothed states and shocks (covariances from particle
approximation).

# References
- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for nonlinear
  time series. JASA, 99(465), 156-168.
"""
function dsge_particle_smoother(nss::NonlinearStateSpace{T}, data::Matrix{T};
                                 N::Int=1000, N_back::Int=100,
                                 rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_obs, T_obs = size(data)
    nx = length(nss.state_indices)
    ny = length(nss.control_indices)
    n_endog = nx + ny
    n_eps = size(nss.eta, 2)
    nv = nx + n_eps

    # Extract first-order blocks for impact matrix
    eta_x = @view nss.hx[:, nx+1:nv]       # nx x n_eps
    eta_y = @view nss.gx[:, nx+1:nv]       # ny x n_eps

    # Build the full impact matrix (n_endog x n_eps)
    impact_full = zeros(T, n_endog, n_eps)
    for (k, si) in enumerate(nss.state_indices)
        impact_full[si, :] = eta_x[k, :]
    end
    for (k, ci) in enumerate(nss.control_indices)
        impact_full[ci, :] = eta_y[k, :]
    end
    impact_pinv = pinv(impact_full)

    # =====================================================================
    # Forward pass — Bootstrap Particle Filter with storage
    # =====================================================================

    # Allocate PF workspace
    ws = _allocate_pf_workspace(T, n_endog, n_obs, n_eps, N;
                                 nv=nv, nx=nx, order=nss.order)

    # Storage for all particles and weights at each time step (for backward pass)
    # Store full endogenous particles and pruned components
    particles_store = zeros(T, n_endog, N, T_obs)
    weights_store = zeros(T, N, T_obs)
    # Also store pruned state components for each particle at each t
    xf_store = zeros(T, nx, N, T_obs)
    xs_store = zeros(T, nx, N, T_obs)
    xrd_store = nss.order >= 3 ? zeros(T, nx, N, T_obs) : nothing

    # Initialize particles at zero (deviation from SS)
    fill!(ws.particles, zero(T))
    fill!(ws.particles_fo, zero(T))
    if ws.particles_so !== nothing
        fill!(ws.particles_so, zero(T))
    end
    if ws.particles_to !== nothing
        fill!(ws.particles_to, zero(T))
    end
    inv_N = one(T) / N
    fill!(ws.weights, inv_N)
    fill!(ws.log_weights, -log(T(N)))

    log_lik = zero(T)
    log_N = log(T(N))

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate through pruned nonlinear transition
        _pf_transition_pruned!(ws, nss)

        # Compute log weights
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, nss.Z, nss.d, nss.H_inv, nss.log_det_H)

        # Accumulate log-likelihood
        log_lik += _logsumexp(ws.log_weights) - log_N

        # Normalize weights
        _normalize_log_weights!(ws.weights, ws.log_weights)

        # Store particles and weights BEFORE resampling
        for k in 1:N
            weights_store[k, t] = ws.weights[k]
            for s in 1:n_endog
                particles_store[s, k, t] = ws.particles[s, k]
            end
            for i in 1:nx
                xf_store[i, k, t] = ws.particles_fo[i, k]
            end
            if ws.particles_so !== nothing
                for i in 1:nx
                    xs_store[i, k, t] = ws.particles_so[i, k]
                end
            end
            if xrd_store !== nothing && ws.particles_to !== nothing
                for i in 1:nx
                    xrd_store[i, k, t] = ws.particles_to[i, k]
                end
            end
        end

        # ESS-based adaptive resampling
        ess = zero(T)
        for k in 1:N
            ess += ws.weights[k] * ws.weights[k]
        end
        ess = one(T) / ess

        if ess < T(0.5) * N
            _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            _pf_resample_pruned!(ws, N, nx, nss.order)
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end
    end

    # =====================================================================
    # Backward simulation (Godsill-Doucet-West 2004)
    # =====================================================================

    # For transition density evaluation: first-order hx_state block
    hx_state = nx > 0 ? Matrix{T}(nss.hx[:, 1:nx]) : zeros(T, 0, 0)

    # Pre-compute transition impact pseudo-inverse (for shock extraction from state only)
    # The first-order impact on states: eta_x (nx x n_eps)
    eta_x_mat = Matrix{T}(eta_x)
    eta_x_pinv = pinv(eta_x_mat)

    # Backward trajectories: store full states
    back_states = zeros(T, n_endog, N_back, T_obs)
    back_shocks = zeros(T, n_eps, N_back, T_obs)

    # Temporary backward weights
    bw = zeros(T, N)

    for b in 1:N_back
        # Draw x_T from final weights
        idx_T = _categorical_draw(@view(weights_store[:, T_obs]), rng)
        back_states[:, b, T_obs] = particles_store[:, idx_T, T_obs]

        # Backward pass: t = T_obs-1 down to 1
        for t in (T_obs - 1):-1:1
            x_tp1 = @view back_states[:, b, t + 1]

            # Extract the state part of x_{t+1}
            x_tp1_states = zeros(T, nx)
            for (k, si) in enumerate(nss.state_indices)
                x_tp1_states[k] = x_tp1[si]
            end

            # Compute backward weights: w_tilde_k = w_t^k * p(x_{t+1} | x_t^k)
            # Transition density approximation: first-order
            # x_{t+1}_states approx = hx_state * xf_t^k + eta_x * eps
            # implied_shock = eta_x_pinv * (x_{t+1}_states - hx_state * xf_t^k)
            # log p = -0.5 * ||implied_shock||^2 (standard normal prior on shocks)
            max_log_bw = T(-Inf)
            for k in 1:N
                xf_k = @view xf_store[:, k, t]
                predicted = hx_state * xf_k
                residual = x_tp1_states - predicted
                implied_shock = eta_x_pinv * residual
                log_trans = -T(0.5) * dot(implied_shock, implied_shock)
                bw[k] = log(max(weights_store[k, t], T(1e-300))) + log_trans
                if bw[k] > max_log_bw
                    max_log_bw = bw[k]
                end
            end

            # Normalize backward weights (in log space then exponentiate)
            bw_sum = zero(T)
            for k in 1:N
                bw[k] = exp(bw[k] - max_log_bw)
                bw_sum += bw[k]
            end
            if bw_sum > zero(T)
                for k in 1:N
                    bw[k] /= bw_sum
                end
            else
                # Fallback: uniform
                fill!(bw, inv_N)
            end

            # Draw ancestor
            idx = _categorical_draw(bw, rng)
            back_states[:, b, t] = particles_store[:, idx, t]
        end

        # Extract shocks from backward trajectory
        # eps_t = impact_pinv * (full_state_t - f(full_state_{t-1}, 0))
        # For t=1: x_prev = 0
        prev_xf = zeros(T, nx)
        prev_xs = zeros(T, nx)
        prev_xrd = zeros(T, nx)

        for t in 1:T_obs
            x_t = @view back_states[:, b, t]

            # Get the deterministic prediction (zero shocks) from previous state
            # Assemble deterministic full state
            det_full = zeros(T, n_endog)
            xf_det, xs_det, xrd_det, x_total_det, y_det = _nonlinear_transition(nss, prev_xf, prev_xs, prev_xrd, zeros(T, n_eps))
            for (k, si) in enumerate(nss.state_indices)
                det_full[si] = x_total_det[k]
            end
            for (k, ci) in enumerate(nss.control_indices)
                det_full[ci] = y_det[k]
            end

            # Implied shock
            resid = Vector{T}(x_t) - det_full
            implied_eps = impact_pinv * resid
            back_shocks[:, b, t] = implied_eps

            # Update previous states for next step
            # Recover pruned components from the shock we just extracted
            xf_new, xs_new, xrd_new, _, _ = _nonlinear_transition(nss, prev_xf, prev_xs, prev_xrd, implied_eps)
            prev_xf = xf_new
            prev_xs = xs_new
            prev_xrd = xrd_new
        end
    end

    # =====================================================================
    # Aggregate: posterior mean of smoothed states and shocks
    # =====================================================================
    smoothed_states = zeros(T, n_endog, T_obs)
    smoothed_shocks = zeros(T, n_eps, T_obs)

    for t in 1:T_obs
        for b in 1:N_back
            for s in 1:n_endog
                smoothed_states[s, t] += back_states[s, b, t]
            end
            for j in 1:n_eps
                smoothed_shocks[j, t] += back_shocks[j, b, t]
            end
        end
        smoothed_states[:, t] ./= N_back
        smoothed_shocks[:, t] ./= N_back
    end

    # =====================================================================
    # Compute covariances from particle approximation
    # =====================================================================
    smoothed_cov = zeros(T, n_endog, n_endog, T_obs)
    for t in 1:T_obs
        for b in 1:N_back
            diff = back_states[:, b, t] - smoothed_states[:, t]
            for j in 1:n_endog, i in 1:n_endog
                smoothed_cov[i, j, t] += diff[i] * diff[j]
            end
        end
        smoothed_cov[:, :, t] ./= (N_back - 1)
    end

    # Use weighted forward filter means as filtered/predicted quantities
    filtered_states = zeros(T, n_endog, T_obs)
    filtered_cov = zeros(T, n_endog, n_endog, T_obs)
    predicted_states = zeros(T, n_endog, T_obs)
    predicted_cov = zeros(T, n_endog, n_endog, T_obs)

    for t in 1:T_obs
        for k in 1:N
            w = weights_store[k, t]
            for s in 1:n_endog
                filtered_states[s, t] += w * particles_store[s, k, t]
            end
        end
    end
    # Approximate filtered covariance
    for t in 1:T_obs
        for k in 1:N
            w = weights_store[k, t]
            diff = particles_store[:, k, t] - filtered_states[:, t]
            for j in 1:n_endog, i in 1:n_endog
                filtered_cov[i, j, t] += w * diff[i] * diff[j]
            end
        end
    end

    # Predicted = filtered from previous step propagated (approximate: use filtered)
    copyto!(predicted_states, filtered_states)
    copyto!(predicted_cov, filtered_cov)

    return KalmanSmootherResult{T}(
        smoothed_states, smoothed_cov, smoothed_shocks,
        filtered_states, filtered_cov,
        predicted_states, predicted_cov,
        log_lik
    )
end
