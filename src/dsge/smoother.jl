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
    error("dsge_smoother is not yet implemented — see Task 2/3 of the DSGE HD plan")
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
