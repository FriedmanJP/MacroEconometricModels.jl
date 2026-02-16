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
Kalman filter/smoother utilities for dynamic factor models.

These utilities are used by the dynamic factor model estimation.
"""

using LinearAlgebra

# =============================================================================
# Factor Model Forecast Result
# =============================================================================

"""
    FactorForecast{T<:AbstractFloat}

Result of factor model forecasting with optional confidence intervals.

Fields: factors, observables, factors_lower, factors_upper, observables_lower, observables_upper,
factors_se, observables_se, horizon, conf_level, ci_method.

When `ci_method == :none`, CI and SE fields are zero matrices.
"""
struct FactorForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    factors::Matrix{T}            # h × r factor forecasts
    observables::Matrix{T}        # h × N observable forecasts
    factors_lower::Matrix{T}      # h × r lower CI for factors
    factors_upper::Matrix{T}      # h × r upper CI for factors
    observables_lower::Matrix{T}  # h × N lower CI for observables
    observables_upper::Matrix{T}  # h × N upper CI for observables
    factors_se::Matrix{T}         # h × r standard errors for factors
    observables_se::Matrix{T}     # h × N standard errors for observables
    horizon::Int
    conf_level::T
    ci_method::Symbol             # :none, :theoretical, :bootstrap, :simulation
end

function Base.show(io::IO, fc::FactorForecast{T}) where {T}
    h, r = size(fc.factors)
    N = size(fc.observables, 2)
    ci_str = fc.ci_method == :none ? "none" : "$(fc.ci_method) ($(round(100*fc.conf_level, digits=1))%)"
    data = Any[
        "Horizon"     h;
        "Factors"     r;
        "Observables" N;
        "CI method"   ci_str
    ]
    _pretty_table(io, data;
        title = "Factor Forecast",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# Shared Utilities
# =============================================================================

"""Standardize matrix: subtract mean, divide by std."""
function _standardize(X::AbstractMatrix{T}) where {T}
    μ, σ = mean(X, dims=1), max.(std(X, dims=1), T(1e-10))
    (X .- μ) ./ σ
end

# =============================================================================
# Kalman Filter/Smoother for Dynamic Factor Model
# =============================================================================

"""
    _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p) -> (a_smooth, P_smooth, Pt_smooth, loglik)

Kalman filter and smoother for state-space form of dynamic factor model.

State-space representation:
- Observation: Y_t = Z * α_t + ε_t, ε_t ~ N(0, H)
- State: α_t = T * α_{t-1} + η_t, η_t ~ N(0, Q)

Where:
- α_t = [F_t', F_{t-1}', ..., F_{t-p+1}']' (stacked factors)
- Z = [Λ, 0, ..., 0] (observation matrix)
- T = companion matrix for factor VAR
- Q = [Sigma_eta, 0; 0, 0] (state innovation covariance)
- H = Sigma_e (observation noise covariance)

Returns smoothed state estimates, covariances, and log-likelihood.
"""
function _kalman_smoother_dfm(Y::AbstractMatrix{T}, Λ::AbstractMatrix{T}, A::Vector{Matrix{T}},
    Sigma_eta::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, r::Int, p::Int
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    # Build state-space matrices
    Z = zeros(T, N, state_dim); Z[:, 1:r] = Λ
    T_mat = zeros(T, state_dim, state_dim)
    for lag in 1:p
        T_mat[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (T_mat[(r+1):state_dim, 1:(state_dim-r)] = I(state_dim - r))

    Q = zeros(T, state_dim, state_dim); Q[1:r, 1:r] = Sigma_eta
    H = Sigma_e

    # Initialize from unconditional distribution
    a0, P0 = zeros(T, state_dim), _compute_unconditional_covariance(T_mat, Q, state_dim)

    # Forward pass: Kalman filter
    a_filt = zeros(T, T_obs, state_dim)
    P_filt = zeros(T, T_obs, state_dim, state_dim)
    a_pred = zeros(T, T_obs, state_dim)
    P_pred = zeros(T, T_obs, state_dim, state_dim)
    loglik, a_t, P_t = zero(T), a0, P0

    for t in 1:T_obs
        # Prediction step
        a_pred[t, :] = T_mat * a_t
        P_pred[t, :, :] = T_mat * P_t * T_mat' + Q

        # Update step
        v_t = Y[t, :] - Z * a_pred[t, :]
        F_t = Symmetric(Z * P_pred[t, :, :] * Z' + H)
        F_inv = try inv(F_t) catch; pinv(F_t) end
        K_t = P_pred[t, :, :] * Z' * F_inv

        a_filt[t, :] = a_pred[t, :] + K_t * v_t
        P_filt[t, :, :] = (I(state_dim) - K_t * Z) * P_pred[t, :, :]

        # Log-likelihood contribution
        det_F = det(F_t)
        det_F > 0 && (loglik -= 0.5 * (N * log(2π) + log(det_F) + v_t' * F_inv * v_t))
        a_t, P_t = a_filt[t, :], P_filt[t, :, :]
    end

    # Backward pass: Kalman smoother
    a_smooth = zeros(T, T_obs, state_dim)
    P_smooth = zeros(T, T_obs, state_dim, state_dim)
    Pt_smooth = zeros(T, T_obs-1, state_dim, state_dim)

    a_smooth[T_obs, :], P_smooth[T_obs, :, :] = a_filt[T_obs, :], P_filt[T_obs, :, :]

    for t in (T_obs-1):-1:1
        P_pred_inv = try inv(Symmetric(P_pred[t+1, :, :])) catch; pinv(Symmetric(P_pred[t+1, :, :])) end
        J_t = P_filt[t, :, :] * T_mat' * P_pred_inv
        a_smooth[t, :] = a_filt[t, :] + J_t * (a_smooth[t+1, :] - a_pred[t+1, :])
        P_smooth[t, :, :] = P_filt[t, :, :] + J_t * (P_smooth[t+1, :, :] - P_pred[t+1, :, :]) * J_t'
        t < T_obs && (Pt_smooth[t, :, :] = J_t * P_smooth[t+1, :, :])
    end

    a_smooth, P_smooth, Pt_smooth, loglik
end

# =============================================================================
# Unconditional Covariance Computation
# =============================================================================

"""
    _compute_unconditional_covariance(T_mat, Q, state_dim; max_iter=1000, tol=1e-10)

Compute unconditional covariance of state vector by solving the discrete Lyapunov equation:
P = T * P * T' + Q

For stationary systems, iterates until convergence. For non-stationary systems,
returns a large diagonal matrix as fallback.
"""
function _compute_unconditional_covariance(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T},
    state_dim::Int; max_iter::Int=1000, tol::Float64=1e-10
) where {T<:AbstractFloat}
    # Check stationarity
    maximum(abs.(eigvals(T_mat))) >= 1.0 && return Matrix{T}(10.0 * I(state_dim))

    # Iterate Lyapunov equation
    P = Matrix{T}(I(state_dim))
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        norm(P_new - P) < tol * norm(P) && return Symmetric(P_new)
        P = P_new
    end
    Symmetric(P)
end

# =============================================================================
# Factor Forecast Helpers (shared across Static FM, DFM, GDFM)
# =============================================================================

"""
    _factor_forecast_var_theoretical(A, Sigma_eta, r, p, h) -> Vector{Matrix{T}}

Compute h-step forecast error covariance for factor VAR(p) via VMA(∞) representation.

Returns vector of h covariance matrices (r × r each): MSE_h = Σ_{j=0}^{h-1} Ψ_j Σ_η Ψ_j'.
"""
function _factor_forecast_var_theoretical(A::Vector{<:AbstractMatrix{T}}, Sigma_eta::AbstractMatrix{T},
    r::Int, p::Int, h::Int) where {T<:AbstractFloat}

    state_dim = r * p
    # Build companion matrix
    C = zeros(T, state_dim, state_dim)
    for lag in 1:p
        C[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (C[(r+1):end, 1:(r*(p-1))] = I(r * (p - 1)))

    # Build state-level Q
    Q = zeros(T, state_dim, state_dim)
    Q[1:r, 1:r] = Sigma_eta

    # Selector: first r rows of companion state
    J = zeros(T, r, state_dim)
    J[1:r, 1:r] = I(r)

    # Accumulate MSE via VMA representation
    mse = Vector{Matrix{T}}(undef, h)
    C_power = Matrix{T}(I, state_dim, state_dim)
    cumul = zeros(T, r, r)
    for step in 1:h
        Psi = J * C_power
        cumul += Psi * Q * Psi'
        mse[step] = copy(cumul)
        C_power = C_power * C
    end
    mse
end

"""
    _factor_forecast_obs_se(factor_mse, Lambda, Sigma_e, h) -> Matrix{T}

Compute observable forecast standard errors.

Var(X_{T+h} error) = Λ * MSE_factor_h * Λ' + Σ_e. Returns h × N matrix of SEs.
"""
function _factor_forecast_obs_se(factor_mse::Vector{Matrix{T}}, Lambda::AbstractMatrix{T},
    Sigma_e::AbstractMatrix{T}, h::Int) where {T<:AbstractFloat}

    N = size(Lambda, 1)
    obs_se = Matrix{T}(undef, h, N)
    for step in 1:h
        obs_var = Lambda * factor_mse[step] * Lambda' + Sigma_e
        obs_se[step, :] = sqrt.(max.(diag(obs_var), zero(T)))
    end
    obs_se
end

"""
    _factor_forecast_bootstrap(F_last, A, resids, Sigma_e, Lambda, h, r, p, n_boot, conf_level) -> tuple

Residual bootstrap for factor forecast CIs. Resamples factor VAR residuals,
simulates factor paths, projects to observables, computes percentile CIs.

Returns (f_lo, f_hi, o_lo, o_hi, f_se, o_se).
"""
function _factor_forecast_bootstrap(F_last::Vector{Vector{T}}, A::Vector{<:AbstractMatrix{T}},
    resids::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, Lambda::AbstractMatrix{T},
    h::Int, r::Int, p::Int, n_boot::Int, conf_level::T) where {T<:AbstractFloat}

    N = size(Lambda, 1)
    T_resid = size(resids, 1)
    L_e = safe_cholesky(Sigma_e)

    F_boot = zeros(T, n_boot, h, r)
    X_boot = zeros(T, n_boot, h, N)

    for b in 1:n_boot
        for step in 1:h
            # VAR forecast with resampled innovation
            F_h = sum(A[lag] * (step - lag >= 1 ? F_boot[b, step - lag, :] : F_last[lag - step + 1]) for lag in 1:p)
            boot_idx = rand(1:T_resid)
            F_boot[b, step, :] = F_h + resids[boot_idx, :]
            X_boot[b, step, :] = Lambda * F_boot[b, step, :] + L_e * randn(T, N)
        end
    end

    α_lo = (1 - conf_level) / 2
    α_hi = 1 - α_lo
    f_lo = T[quantile(F_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:r]
    f_hi = T[quantile(F_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:r]
    o_lo = T[quantile(X_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:N]
    o_hi = T[quantile(X_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:N]
    f_se = T[std(F_boot[:, hh, j]) for hh in 1:h, j in 1:r]
    o_se = T[std(X_boot[:, hh, j]) for hh in 1:h, j in 1:N]

    (f_lo, f_hi, o_lo, o_hi, f_se, o_se)
end

"""
    _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, X_original)

In-place unstandardization of observable forecasts using mean/std of original data.
"""
function _unstandardize_factor_forecast!(X_fc::Matrix{T}, X_lo::Matrix{T}, X_hi::Matrix{T},
    X_se::Matrix{T}, X_original::AbstractMatrix{T}) where {T<:AbstractFloat}

    μ = vec(mean(X_original, dims=1))
    σ = max.(vec(std(X_original, dims=1)), T(1e-10))
    X_fc .= X_fc .* σ' .+ μ'
    X_lo .= X_lo .* σ' .+ μ'
    X_hi .= X_hi .* σ' .+ μ'
    X_se .= X_se .* σ'
    nothing
end

"""
    _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_level, ci_method) -> FactorForecast{T}

Construct a FactorForecast from components.
"""
function _build_factor_forecast(F_fc::Matrix{T}, X_fc::Matrix{T},
    F_lo::Matrix{T}, F_hi::Matrix{T}, X_lo::Matrix{T}, X_hi::Matrix{T},
    F_se::Matrix{T}, X_se::Matrix{T}, h::Int, conf_level::T, ci_method::Symbol) where {T<:AbstractFloat}

    FactorForecast{T}(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_level, ci_method)
end
