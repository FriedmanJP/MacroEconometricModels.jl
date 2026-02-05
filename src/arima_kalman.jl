"""
State-space representation and Kalman filter for ARMA models.

Uses Harvey's (1993) state-space form for exact MLE.
"""

using LinearAlgebra

# =============================================================================
# State-Space Matrices (Harvey's Form)
# =============================================================================

"""
    _arma_state_space(c, phi, theta, sigma2, p, q) -> (Z, T, R, Q, H, state_dim)

Build state-space matrices for ARMA(p,q) model using Harvey's representation.

The state-space form is:
- Observation: yₜ = c + Z αₜ
- State: αₜ₊₁ = T αₜ + R ηₜ, ηₜ ~ N(0, Q)

where the state vector αₜ = [aₜ, aₜ₋₁, ..., aₜ₋ᵣ₊₁]' has dimension r = max(p, q+1).

# Arguments
- `c`: Intercept
- `phi`: AR coefficients [φ₁, ..., φₚ]
- `theta`: MA coefficients [θ₁, ..., θq]
- `sigma2`: Innovation variance

# Returns
Tuple (Z, T, R, Q, H, state_dim) of state-space matrices and state dimension.
"""
function _arma_state_space(c::T, phi::Vector{T}, theta::Vector{T}, sigma2::T, p::Int, q::Int) where {T<:AbstractFloat}
    r = max(p, q + 1)

    # Observation matrix: Z = [1, θ₁, ..., θ_{r-1}]
    Z = zeros(T, 1, r)
    Z[1, 1] = one(T)
    for j in 1:min(q, r-1)
        Z[1, j+1] = theta[j]
    end

    # Transition matrix: companion form with AR coefficients
    T_mat = zeros(T, r, r)
    for i in 1:min(p, r)
        T_mat[1, i] = phi[i]
    end
    if r > 1
        T_mat[2:r, 1:r-1] = I(r - 1)
    end

    # Selection matrix: R = [1, 0, ..., 0]'
    R = zeros(T, r, 1)
    R[1, 1] = one(T)

    # State innovation variance
    Q = fill(sigma2, 1, 1)

    # Observation noise (zero for pure ARMA)
    H = zeros(T, 1, 1)

    Z, T_mat, R, Q, H, r
end

# =============================================================================
# Stationarity and Invertibility Checks
# =============================================================================

"""
    _is_stationary(phi) -> Bool

Check if AR polynomial is stationary (all roots outside unit circle).
Uses companion matrix eigenvalue check: |λᵢ| < 1 for all eigenvalues.
"""
function _is_stationary(phi::Vector{T}) where {T<:AbstractFloat}
    isempty(phi) && return true
    p = length(phi)

    # Build companion matrix
    if p == 1
        return abs(phi[1]) < one(T)
    end

    F = zeros(T, p, p)
    F[1, :] = phi
    F[2:p, 1:p-1] = I(p - 1)

    maximum(abs.(eigvals(F))) < one(T)
end

"""
    _is_invertible(theta) -> Bool

Check if MA polynomial is invertible (all roots outside unit circle).
"""
function _is_invertible(theta::Vector{T}) where {T<:AbstractFloat}
    isempty(theta) && return true
    q = length(theta)

    if q == 1
        return abs(theta[1]) < one(T)
    end

    # Build companion matrix for MA polynomial
    F = zeros(T, q, q)
    F[1, :] = theta
    F[2:q, 1:q-1] = I(q - 1)

    maximum(abs.(eigvals(F))) < one(T)
end

# =============================================================================
# Kalman Filter for Log-Likelihood
# =============================================================================

"""
    _kalman_filter_arma(y, c, phi, theta, sigma2) -> (loglik, residuals, fitted)

Compute log-likelihood and residuals for ARMA model via Kalman filter.

Uses the exact likelihood based on prediction error decomposition.
Initializes with unconditional (stationary) distribution when possible,
falls back to diffuse initialization for non-stationary parameters.

# Arguments
- `y`: Observed time series
- `c`: Intercept
- `phi`: AR coefficients
- `theta`: MA coefficients
- `sigma2`: Innovation variance

# Returns
- `loglik`: Log-likelihood value
- `residuals`: One-step-ahead prediction errors
- `fitted`: One-step-ahead predictions
"""
function _kalman_filter_arma(y::Vector{T}, c::T, phi::Vector{T}, theta::Vector{T}, sigma2::T) where {T<:AbstractFloat}
    n = length(y)
    p, q = length(phi), length(theta)

    # Handle edge case: white noise
    if p == 0 && q == 0
        residuals = y .- c
        fitted = fill(c, n)
        ss = sum(abs2, residuals)
        loglik = -T(n/2) * log(T(2π)) - T(n/2) * log(sigma2) - ss / (2 * sigma2)
        return loglik, residuals, fitted
    end

    # Build state-space matrices
    Z, T_mat, R, Q, H, r = _arma_state_space(c, phi, theta, sigma2, p, q)

    # Initialize state
    a, P = _initialize_state(T_mat, R, Q, r, T)

    # Storage
    residuals = zeros(T, n)
    fitted = zeros(T, n)
    loglik = zero(T)

    @inbounds for t in 1:n
        # Prediction error
        y_pred = c + dot(Z, a)
        fitted[t] = y_pred
        v = y[t] - y_pred

        # Prediction error variance
        F = Z * P * Z' .+ H
        f = F[1, 1]

        # Skip if variance is too small (numerical issues)
        if f < T(1e-12)
            residuals[t] = v
            a = T_mat * a
            P = T_mat * P * T_mat' + R * Q * R'
            continue
        end

        # Log-likelihood contribution
        loglik -= T(0.5) * (log(T(2π)) + log(f) + v^2 / f)
        residuals[t] = v

        # Kalman gain
        K = T_mat * P * Z' / f

        # Update state
        a = T_mat * a + K * v
        P = T_mat * P * T_mat' + R * Q * R' - K * (K' * f)

        # Ensure symmetry
        P = (P + P') / 2
    end

    loglik, residuals, fitted
end

"""
    _initialize_state(T_mat, R, Q, r, ::Type{T}) -> (a, P)

Initialize Kalman filter state using unconditional distribution.
Falls back to diffuse initialization if system is non-stationary.
"""
function _initialize_state(T_mat::Matrix{T}, R::Matrix{T}, Q::Matrix{T}, r::Int, ::Type{T}) where {T<:AbstractFloat}
    # Initial state mean
    a = zeros(T, r)

    # Check if stationary
    max_eig = maximum(abs.(eigvals(T_mat)))
    if max_eig >= one(T)
        # Diffuse initialization for non-stationary
        P = Matrix{T}(1e6 * I(r))
        return a, P
    end

    # Solve discrete Lyapunov equation: P = T P T' + R Q R'
    RQR = R * Q * R'
    P = _solve_lyapunov(T_mat, RQR, r; tol=T(1e-10))

    a, P
end

"""
    _solve_lyapunov(T_mat, Q, n; max_iter=1000, tol=1e-10)

Solve discrete Lyapunov equation P = T P T' + Q by iteration.
"""
function _solve_lyapunov(T_mat::Matrix{T}, Q::Matrix{T}, n::Int; max_iter::Int=1000, tol::T=T(1e-10)) where {T<:AbstractFloat}
    P = Matrix{T}(I(n))
    for _ in 1:max_iter
        P_new = T_mat * P * T_mat' + Q
        if norm(P_new - P) < tol * max(norm(P), one(T))
            return Symmetric(P_new)
        end
        P = P_new
    end
    Symmetric(P)
end

# =============================================================================
# Negative Log-Likelihood for Optimization
# =============================================================================

"""
    _unpack_arma_params(params, p, q; include_intercept=true, has_log_sigma2=false)

Unpack a parameter vector into (c, phi, theta, sigma2_or_nothing).

When `has_log_sigma2=true`, the last element is treated as log(σ²) and
the returned fourth element is `exp(log_sigma2)`. Otherwise the fourth
element is `nothing`.
"""
function _unpack_arma_params(params::Vector{T}, p::Int, q::Int;
                              include_intercept::Bool=true,
                              has_log_sigma2::Bool=false) where {T}
    idx = 1
    if include_intercept
        c = params[idx]
        idx += 1
    else
        c = zero(T)
    end

    phi = p > 0 ? params[idx:idx+p-1] : T[]
    idx += p

    theta = q > 0 ? params[idx:idx+q-1] : T[]
    idx += q

    sigma2 = has_log_sigma2 ? exp(params[idx]) : nothing

    (c, phi, theta, sigma2)
end

"""
    _arma_negloglik(params, y, p, q; include_intercept=true) -> T

Compute negative log-likelihood for optimization.

Parameters are packed as [c, φ₁, ..., φₚ, θ₁, ..., θq, log(σ²)].
Uses log(σ²) for unconstrained optimization of variance.

Returns large penalty for non-stationary/non-invertible parameters.
"""
function _arma_negloglik(params::Vector{T}, y::Vector{T}, p::Int, q::Int; include_intercept::Bool=true) where {T<:AbstractFloat}
    c, phi, theta, sigma2 = _unpack_arma_params(params, p, q;
                                                  include_intercept=include_intercept,
                                                  has_log_sigma2=true)

    # Penalty for non-stationarity or non-invertibility
    penalty = T(1e10)
    !_is_stationary(phi) && return penalty
    !_is_invertible(theta) && return penalty

    # Compute log-likelihood via Kalman filter
    loglik, _, _ = _kalman_filter_arma(y, c, phi, theta, sigma2)

    # Handle numerical issues
    isnan(loglik) && return penalty
    isinf(loglik) && return penalty

    -loglik
end

# =============================================================================
# Residuals Computation
# =============================================================================

"""
    _compute_arma_residuals(y, c, phi, theta) -> residuals

Compute residuals using recursive filtering.
εₜ = yₜ - c - Σφᵢyₜ₋ᵢ - Σθⱼεₜ₋ⱼ

Initializes past residuals to zero.
"""
function _compute_arma_residuals(y::Vector{T}, c::T, phi::Vector{T}, theta::Vector{T}) where {T<:AbstractFloat}
    n = length(y)
    p, q = length(phi), length(theta)
    m = max(p, q)

    residuals = zeros(T, n)

    @inbounds for t in 1:n
        # AR component
        ar_part = zero(T)
        for i in 1:min(p, t-1)
            ar_part += phi[i] * y[t-i]
        end

        # MA component
        ma_part = zero(T)
        for j in 1:min(q, t-1)
            ma_part += theta[j] * residuals[t-j]
        end

        residuals[t] = y[t] - c - ar_part - ma_part
    end

    residuals
end
