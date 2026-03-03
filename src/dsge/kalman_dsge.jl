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
Observation equation builder and DSGE-specific Kalman filter for Bayesian estimation.

Provides:
- `_build_observation_equation` — maps observables to state space selection matrices
- `_build_state_space` — constructs `DSGEStateSpace` from `DSGESolution` + observation eq
- `_build_nonlinear_state_space` — constructs `NonlinearStateSpace` from `PerturbationSolution`
- `_kalman_loglikelihood` — prediction error decomposition log-likelihood (performance-critical)

References:
- Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. Ch. 13.
- Herbst, E. & Schorfheide, F. (2015). Bayesian Estimation of DSGE Models. Princeton University Press.
"""

using LinearAlgebra

# =============================================================================
# Observation equation builder
# =============================================================================

"""
    _build_observation_equation(spec::DSGESpec{T}, observables::Vector{Symbol},
                                 measurement_error) where {T}

Build the observation equation matrices (Z, d, H) mapping state variables to observables.

The observation equation is: `y_t = Z * x_t + d + v_t`, where `v_t ~ N(0, H)`.

# Arguments
- `spec` — DSGE model specification
- `observables` — symbols identifying which endogenous variables are observed
- `measurement_error` — vector of measurement error standard deviations, or `nothing`
  for default (1e-4 * I)

# Returns
- `Z::Matrix{T}` — n_obs x n_states selection matrix (ones at observable positions)
- `d::Vector{T}` — n_obs steady-state intercept at observable indices
- `H::Matrix{T}` — n_obs x n_obs measurement error covariance
"""
function _build_observation_equation(spec::DSGESpec{T}, observables::Vector{Symbol},
                                      measurement_error) where {T<:AbstractFloat}
    n_states = spec.n_endog
    n_obs = length(observables)
    n_obs == 0 && throw(ArgumentError("observables must be non-empty"))

    # Find indices of observables in spec.endog
    obs_indices = Int[]
    for obs in observables
        idx = findfirst(==(obs), spec.endog)
        idx === nothing && throw(ArgumentError(
            "Observable :$obs not found in model endogenous variables: $(spec.endog)"))
        push!(obs_indices, idx)
    end

    # Build Z: selection matrix with 1s at observable positions
    Z = zeros(T, n_obs, n_states)
    for (i, j) in enumerate(obs_indices)
        Z[i, j] = one(T)
    end

    # d: steady-state values at observable indices (zeros if not yet computed)
    if isempty(spec.steady_state)
        d = zeros(T, n_obs)
    else
        d = T[spec.steady_state[j] for j in obs_indices]
    end

    # H: measurement error covariance
    if measurement_error === nothing
        H = T(1e-4) * Matrix{T}(I, n_obs, n_obs)
    else
        me = Vector{T}(measurement_error)
        length(me) == n_obs || throw(ArgumentError(
            "measurement_error length ($(length(me))) must match observables ($n_obs)"))
        H = diagm(me .^ 2)
    end

    return Z, d, H
end

# =============================================================================
# State space constructors
# =============================================================================

"""
    _build_state_space(sol::DSGESolution{T}, Z::Matrix{T}, d::Vector{T},
                        H::Matrix{T}) where {T}

Construct a `DSGEStateSpace{T}` from a linear DSGE solution and observation equation matrices.

Shock covariance Q defaults to identity (unit-variance structural shocks).
"""
function _build_state_space(sol::DSGESolution{T}, Z::Matrix{T}, d::Vector{T},
                             H::Matrix{T}) where {T<:AbstractFloat}
    n_shocks = size(sol.impact, 2)
    Q = Matrix{T}(I, n_shocks, n_shocks)
    DSGEStateSpace{T}(sol.G1, sol.impact, Z, d, H, Q)
end

"""
    _build_nonlinear_state_space(sol::PerturbationSolution{T}, Z::Matrix{T},
                                  d::Vector{T}, H::Matrix{T}) where {T}

Construct a `NonlinearStateSpace{T}` from a higher-order perturbation solution and
observation equation matrices. Passes all coefficient matrices (1st/2nd/3rd order)
through to the NonlinearStateSpace constructor.
"""
function _build_nonlinear_state_space(sol::PerturbationSolution{T}, Z::Matrix{T},
                                       d::Vector{T}, H::Matrix{T}) where {T<:AbstractFloat}
    NonlinearStateSpace{T}(
        sol.hx, sol.gx, sol.eta, sol.steady_state,
        sol.state_indices, sol.control_indices, sol.order,
        sol.hxx, sol.gxx, sol.hσσ, sol.gσσ,
        sol.hxxx, sol.gxxx, sol.hσσx, sol.gσσx, sol.hσσσ, sol.gσσσ,
        Z, d, H
    )
end

# =============================================================================
# Kalman filter log-likelihood — prediction error decomposition
# =============================================================================

"""
    _kalman_loglikelihood(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T}

Compute the log-likelihood of `data` under the linear state space model via the
prediction error decomposition (Harvey 1989, Hamilton 1994 Ch. 13).

# State space model
    x_{t+1} = G1 * x_t + impact * varepsilon_t,    varepsilon_t ~ N(0, Q)
    y_t     = Z * x_t + d + v_t,                    v_t ~ N(0, H)

# Arguments
- `ss` — `DSGEStateSpace{T}` with transition/observation matrices and cached inverses
- `data` — `n_obs x T_obs` matrix (each column is one time period)

# Returns
Scalar log-likelihood (sum over t of log p(y_t | y_{1:t-1})).

# Implementation notes
- **Initialization**: stationary distribution via `solve_lyapunov`; falls back to
  diffuse initialization (P0 = 10 * I) if Lyapunov equation fails (e.g., unit root).
- **Pre-allocation**: all workspace matrices allocated once before the loop.
- **BLAS**: uses `mul!` for matrix-vector and matrix-matrix products (Level 2/3 BLAS).
- **Missing data**: NaN entries in `data` are handled by reducing the observation
  dimension for that time step (rows with NaN are dropped).
- **Symmetry**: P is symmetrized after each update to prevent numerical drift.
"""
function _kalman_loglikelihood(ss::DSGEStateSpace{T}, data::Matrix{T}) where {T<:AbstractFloat}
    n_obs, T_obs = size(data)
    n_states = size(ss.G1, 1)
    n_shocks = size(ss.impact, 2)

    # --- Initialization ---
    x = zeros(T, n_states)  # initial state mean (deviation from SS = 0)

    # Stationary covariance via Lyapunov equation; diffuse fallback
    RQR = ss.impact * ss.Q * ss.impact'
    P = try
        solve_lyapunov(ss.G1, ss.impact)
    catch
        T(10) * Matrix{T}(I, n_states, n_states)
    end

    # --- Pre-allocate workspace (zero inner-loop allocation for full obs) ---
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
    KF = zeros(T, n_states, n_obs)  # K * F  (not needed, but K * Z * P is)
    KZP = zeros(T, n_states, n_states)  # K * Z * P_pred (for Joseph form or direct)

    # Check if data has any NaN
    has_nan = any(isnan, data)

    ll = zero(T)
    log2pi = T(log(2 * T(pi)))

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
        end
    end

    return ll
end
