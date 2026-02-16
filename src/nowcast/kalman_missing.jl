"""
Kalman filter and smoother with missing data (NaN-aware).

Handles arbitrary NaN patterns per time step by eliminating missing rows
from the observation equation before each update step. Based on the approach
in Bańbura & Modugno (2014).
"""

# =============================================================================
# Missing Data Handler
# =============================================================================

"""
    _miss_data(y, C, R) -> (y_obs, C_obs, R_obs, obs_idx)

Eliminate rows corresponding to NaN observations.

Returns reduced observation vector, loadings, noise covariance, and
the indices of observed (non-NaN) entries.
"""
function _miss_data(y::AbstractVector{T}, C::AbstractMatrix{T},
                    R::AbstractMatrix{T}) where {T<:AbstractFloat}
    obs_idx = findall(!isnan, y)
    if isempty(obs_idx)
        return T[], Matrix{T}(undef, 0, size(C, 2)), Matrix{T}(undef, 0, 0), Int[]
    end
    y_obs = y[obs_idx]
    C_obs = C[obs_idx, :]
    R_obs = R[obs_idx, obs_idx]
    return y_obs, C_obs, R_obs, obs_idx
end

# =============================================================================
# Kalman Filter with Missing Data
# =============================================================================

"""
    _kalman_filter_missing(y, A, C, Q, R, x0, P0) -> (x_pred, P_pred, x_filt, P_filt, loglik)

Forward Kalman filter handling NaN observations.

At each time step, rows with NaN are removed from the observation equation.
If all observations are missing, the update step is skipped (prior = posterior).

# Arguments
- `y::Matrix{T}` — N × T_obs observation matrix (columns = time steps)
- `A::Matrix{T}` — state transition matrix (state_dim × state_dim)
- `C::Matrix{T}` — observation matrix (N × state_dim)
- `Q::Matrix{T}` — state noise covariance
- `R::Matrix{T}` — observation noise covariance
- `x0::Vector{T}` — initial state mean
- `P0::Matrix{T}` — initial state covariance

# Returns
- `x_pred` — predicted states (state_dim × T_obs)
- `P_pred` — predicted covariances
- `x_filt` — filtered states
- `P_filt` — filtered covariances
- `loglik` — log-likelihood
"""
function _kalman_filter_missing(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                                C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                                R::AbstractMatrix{T}, x0::AbstractVector{T},
                                P0::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    x_pred = zeros(T, state_dim, T_obs)
    P_pred = zeros(T, state_dim, state_dim, T_obs)
    x_filt = zeros(T, state_dim, T_obs)
    P_filt = zeros(T, state_dim, state_dim, T_obs)
    loglik = zero(T)

    x_t = copy(x0)
    P_t = Matrix{T}(P0)

    for t in 1:T_obs
        # Prediction step
        x_pred[:, t] = A * x_t
        P_pred[:, :, t] = A * P_t * A' + Q

        # Handle missing data
        y_obs, C_obs, R_obs, obs_idx = _miss_data(y[:, t], C, R)

        if isempty(obs_idx)
            # All observations missing — skip update
            x_filt[:, t] = x_pred[:, t]
            P_filt[:, :, t] = P_pred[:, :, t]
        else
            # Innovation
            v_t = y_obs - C_obs * x_pred[:, t]
            F_t = Symmetric(C_obs * P_pred[:, :, t] * C_obs' + R_obs)
            F_inv = try inv(F_t) catch; pinv(Matrix{T}(F_t)) end
            K_t = P_pred[:, :, t] * C_obs' * F_inv

            # Update
            x_filt[:, t] = x_pred[:, t] + K_t * v_t
            P_filt[:, :, t] = (I(state_dim) - K_t * C_obs) * P_pred[:, :, t]

            # Log-likelihood contribution
            n_obs = length(obs_idx)
            det_F = det(F_t)
            if det_F > 0
                loglik -= T(0.5) * (n_obs * log(T(2π)) + log(det_F) + v_t' * F_inv * v_t)
            end
        end

        x_t = x_filt[:, t]
        P_t = P_filt[:, :, t]
    end

    return x_pred, P_pred, x_filt, P_filt, loglik
end

# =============================================================================
# Kalman Smoother with Missing Data
# =============================================================================

"""
    _kalman_smoother_missing(y, A, C, Q, R, x0, P0) -> (x_smooth, P_smooth, PP_smooth, loglik)

Kalman smoother (Harvey 1989 fixed-interval) with NaN handling.

Runs forward filter then backward smoother pass.

# Returns
- `x_smooth` — smoothed states (state_dim × T_obs)
- `P_smooth` — smoothed covariances (state_dim × state_dim × T_obs)
- `PP_smooth` — cross-covariances E[x_t x_{t-1}'] (state_dim × state_dim × T_obs)
- `loglik` — log-likelihood
"""
function _kalman_smoother_missing(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                                  C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                                  R::AbstractMatrix{T}, x0::AbstractVector{T},
                                  P0::AbstractMatrix{T}) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    # Forward pass
    x_pred, P_pred, x_filt, P_filt, loglik = _kalman_filter_missing(
        y, A, C, Q, R, x0, P0)

    # Backward pass
    x_smooth = zeros(T, state_dim, T_obs)
    P_smooth = zeros(T, state_dim, state_dim, T_obs)
    PP_smooth = zeros(T, state_dim, state_dim, T_obs)

    x_smooth[:, T_obs] = x_filt[:, T_obs]
    P_smooth[:, :, T_obs] = P_filt[:, :, T_obs]

    for t in (T_obs - 1):-1:1
        P_pred_inv = try
            inv(Symmetric(P_pred[:, :, t + 1]))
        catch
            pinv(Matrix{T}(P_pred[:, :, t + 1]))
        end
        J_t = P_filt[:, :, t] * A' * P_pred_inv

        x_smooth[:, t] = x_filt[:, t] + J_t * (x_smooth[:, t + 1] - x_pred[:, t + 1])
        P_smooth[:, :, t] = P_filt[:, :, t] + J_t * (P_smooth[:, :, t + 1] - P_pred[:, :, t + 1]) * J_t'

        # Cross-covariance for EM sufficient statistics
        PP_smooth[:, :, t + 1] = P_smooth[:, :, t + 1] * J_t' + x_smooth[:, t + 1] * x_smooth[:, t]'
    end

    # First time step cross-covariance (using initial conditions)
    P0_mat = Matrix{T}(P0)
    P_pred_1_inv = try
        inv(Symmetric(P_pred[:, :, 1]))
    catch
        pinv(Matrix{T}(P_pred[:, :, 1]))
    end
    J_0 = P0_mat * A' * P_pred_1_inv
    PP_smooth[:, :, 1] = P_smooth[:, :, 1] * J_0' + x_smooth[:, 1] * x0'

    return x_smooth, P_smooth, PP_smooth, loglik
end

# =============================================================================
# Kalman Smoother with Lagged Cross-Covariances
# =============================================================================

"""
    _kalman_smoother_lag(y, A, C, Q, R, x0, P0, k) -> (x_smooth, P_smooth, Plag, loglik)

Extended Kalman smoother computing lagged cross-covariances up to lag k.

Needed for news decomposition (Bańbura & Modugno 2014).

# Returns
- `x_smooth` — smoothed states
- `P_smooth` — smoothed covariances
- `Plag` — vector of k cross-covariance arrays: Plag[j] = E[x_t x_{t-j}'] - E[x_t]E[x_{t-j}]'
- `loglik` — log-likelihood
"""
function _kalman_smoother_lag(y::AbstractMatrix{T}, A::AbstractMatrix{T},
                              C::AbstractMatrix{T}, Q::AbstractMatrix{T},
                              R::AbstractMatrix{T}, x0::AbstractVector{T},
                              P0::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat}
    N, T_obs = size(y)
    state_dim = length(x0)

    # Forward pass
    x_pred, P_pred, x_filt, P_filt, loglik = _kalman_filter_missing(
        y, A, C, Q, R, x0, P0)

    # Standard backward smoother
    x_smooth = zeros(T, state_dim, T_obs)
    P_smooth = zeros(T, state_dim, state_dim, T_obs)
    J = zeros(T, state_dim, state_dim, T_obs)

    x_smooth[:, T_obs] = x_filt[:, T_obs]
    P_smooth[:, :, T_obs] = P_filt[:, :, T_obs]

    for t in (T_obs - 1):-1:1
        P_pred_inv = try
            inv(Symmetric(P_pred[:, :, t + 1]))
        catch
            pinv(Matrix{T}(P_pred[:, :, t + 1]))
        end
        J[:, :, t] = P_filt[:, :, t] * A' * P_pred_inv
        x_smooth[:, t] = x_filt[:, t] + J[:, :, t] * (x_smooth[:, t + 1] - x_pred[:, t + 1])
        P_smooth[:, :, t] = P_filt[:, :, t] + J[:, :, t] * (P_smooth[:, :, t + 1] - P_pred[:, :, t + 1]) * J[:, :, t]'
    end

    # Compute lagged cross-covariances
    # Plag[j][s,s',t] = Cov(x_t, x_{t-j})
    Plag = Vector{Array{T,3}}(undef, k)
    for j in 1:k
        Plag[j] = zeros(T, state_dim, state_dim, T_obs)
    end

    # Lag 1: P_{t,t-1|T} = P_{t|T} * J_{t-1}'
    # But we also need the full recursion for higher lags
    for t in 2:T_obs
        Plag[1][:, :, t] = P_smooth[:, :, t] * J[:, :, t - 1]'
    end

    # Higher lags via recursion: P_{t,t-j|T} = P_{t,t-1|T} * inv(P_{t-1|T}) * P_{t-1,t-j|T}
    # Simplified: P_{t,t-j} = J_{t-1} * P_{t-1,t-j+1}  (backward recursion)
    for j in 2:k
        for t in (j + 1):T_obs
            Plag[j][:, :, t] = J[:, :, t - 1] * Plag[j - 1][:, :, t - 1]
        end
    end

    return x_smooth, P_smooth, Plag, loglik
end
