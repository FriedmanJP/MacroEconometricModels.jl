# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

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
    _build_observation_equation(spec::ModelSpec{T}, observables::Vector{Symbol},
                                 measurement_error) where {T}

Build the observation equation matrices (Z, d, H) mapping state variables to observables.

The observation equation is: `y_t = Z * x_t + d + v_t`, where `v_t ~ N(0, H)`.

# Arguments
- `spec` — DSGE model specification
- `observables` — symbols identifying which endogenous variables are observed
- `measurement_error` — vector of measurement error standard deviations, or `nothing`
  for zero measurement error (the model-implied covariance is used; requires
  `n_obs ≤ n_shocks`, else a `StochasticSingularityError` is thrown)

# Returns
- `Z::Matrix{T}` — n_obs x n_states selection matrix (ones at observable positions)
- `d::Vector{T}` — n_obs steady-state intercept at observable indices
- `H::Matrix{T}` — n_obs x n_obs measurement error covariance
"""
function _build_observation_equation(spec::ModelSpec{T}, observables::Vector{Symbol},
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

    # H: measurement error covariance. Default is ZERO (use the model covariance); a
    # nonzero default would mask stochastic singularity when n_obs > n_shocks.
    if measurement_error === nothing
        H = zeros(T, n_obs, n_obs)
        n_shocks = spec.n_exog
        if n_obs > n_shocks
            throw(StochasticSingularityError(
                "$n_obs observables exceed $n_shocks structural shocks; the model-implied " *
                "observation covariance is singular and the likelihood is ill-defined. " *
                "Add measurement error (measurement_error=:auto or a vector of SDs) or " *
                "reduce the number of observables."))
        end
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

"""
    _build_projection_state_space(sol::ProjectionSolution, Z, d, H)

Build a `ProjectionStateSpace` from a projection/PFI solution.
Computes the impact matrix once from the linearized solution and
pre-computes scaling constants for the Chebyshev domain.
"""
function _build_projection_state_space(sol::ProjectionSolution{T}, Z::Matrix{T},
                                        d::Vector{T}, H::Matrix{T}) where {T<:AbstractFloat}
    # Reuse cached companion-QZ first-order shock-impact matrix from ProjectionSolution
    impact = Matrix{T}(sol.impact)

    # Pre-compute scaling constants: z_i = x_i * scale_i + shift_i
    nx = length(sol.state_indices)
    scale_vec = Vector{T}(undef, nx)
    shift_vec = Vector{T}(undef, nx)
    for i in 1:nx
        a = sol.state_bounds[i, 1]; b = sol.state_bounds[i, 2]
        scale_vec[i] = T(2) / (b - a)
        shift_vec[i] = -(a + b) / (b - a)
    end

    ProjectionStateSpace{T}(sol.coefficients, sol.multi_indices,
                              maximum(sol.multi_indices),
                              sol.steady_state, sol.state_indices,
                              sol.control_indices, impact, sol.state_bounds,
                              scale_vec, shift_vec, Z, d, H)
end

# =============================================================================
# Lock-free pure-Julia small-matrix kernels for multi-threaded Kalman (#947)
# =============================================================================
# Avoids OpenBLAS `USE_LOCKING` mutex serialization on small n×n (≤16) matrices
# when called concurrently inside `Threads.@threads` / `_parallel_for_worksteal`.

const _KALMAN_SMALL_DIM = 16

@inline function _small_mul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat}
    m, k_dim = size(A)
    n = size(B, 2)
    fill!(C, zero(T))
    @inbounds for j in 1:n, k in 1:k_dim
        bkj = B[k, j]
        @simd for i in 1:m
            C[i, j] += A[i, k] * bkj
        end
    end
    return C
end

@inline function _small_mul_Bt!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat}
    m, k_dim = size(A)
    n = size(B, 1)
    fill!(C, zero(T))
    @inbounds for j in 1:n, k in 1:k_dim
        bjk = B[j, k]
        @simd for i in 1:m
            C[i, j] += A[i, k] * bjk
        end
    end
    return C
end

@inline function _small_mul_vec!(y::Vector{T}, A::Matrix{T}, x::Vector{T}) where {T<:AbstractFloat}
    m, n = size(A)
    fill!(y, zero(T))
    @inbounds for j in 1:n
        xj = x[j]
        @simd for i in 1:m
            y[i] += A[i, j] * xj
        end
    end
    return y
end

@inline function _small_cholesky!(L::Matrix{T}, A::Matrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    log_det = zero(T)
    @inbounds for j in 1:n
        s = A[j, j]
        for k in 1:(j - 1)
            ljk = L[j, k]
            s -= ljk * ljk
        end
        if !(s > zero(T)) || !isfinite(s)
            return false, T(-Inf)
        end
        ljj = sqrt(s)
        L[j, j] = ljj
        log_det += log(ljj)
        inv_ljj = one(T) / ljj
        for i in (j + 1):n
            s_ij = A[i, j]
            for k in 1:(j - 1)
                s_ij -= L[i, k] * L[j, k]
            end
            L[i, j] = s_ij * inv_ljj
        end
    end
    return true, T(2) * log_det
end

@inline function _small_chol_solve_vec!(x::Vector{T}, L::Matrix{T}, b::Vector{T}) where {T<:AbstractFloat}
    n = length(b)
    @inbounds for i in 1:n
        s = b[i]
        for k in 1:(i - 1)
            s -= L[i, k] * x[k]
        end
        x[i] = s / L[i, i]
    end
    @inbounds for i in n:-1:1
        s = x[i]
        for k in (i + 1):n
            s -= L[k, i] * x[k]
        end
        x[i] = s / L[i, i]
    end
    return x
end

@inline function _small_chol_rdiv!(K::Matrix{T}, PZ::Matrix{T}, L::Matrix{T}) where {T<:AbstractFloat}
    m, p = size(PZ)
    @inbounds for j in 1:p
        inv_ljj = one(T) / L[j, j]
        @simd for i in 1:m
            K[i, j] = PZ[i, j]
        end
        for k in 1:(j - 1)
            ljk = L[j, k]
            @simd for i in 1:m
                K[i, j] -= ljk * K[i, k]
            end
        end
        @simd for i in 1:m
            K[i, j] *= inv_ljj
        end
    end
    @inbounds for j in p:-1:1
        inv_ljj = one(T) / L[j, j]
        for k in (j + 1):p
            lkj = L[k, j]
            @simd for i in 1:m
                K[i, j] -= lkj * K[i, k]
            end
        end
        @simd for i in 1:m
            K[i, j] *= inv_ljj
        end
    end
    return K
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
- **Initialization**: stationary distribution via `solve_lyapunov` when all transition
  eigenvalues have modulus `< 1`. For unit roots (any `|eig(G1)| ≥ 1 - 1e-6`), the
  initial covariance uses a finite Lyapunov solution on the stationary subspace plus
  a large diffuse variance (κ = 1e6) on the nonstationary subspace, via
  `_diffuse_initial_covariance` (Durbin & Koopman 2012, ch. 5), emitting a one-time
  warning — never a silent `P0 = 10*I`. Non-stability exceptions (`MethodError`,
  `DimensionMismatch`, etc.) propagate rather than being swallowed.
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

    # Stationary covariance via Lyapunov equation; eigenvalue-partitioned diffuse
    # initialization for unit roots (never a silent P0=10I; non-stability errors propagate).
    use_small = (n_states <= _KALMAN_SMALL_DIM) && (n_obs <= _KALMAN_SMALL_DIM) &&
                (n_shocks <= _KALMAN_SMALL_DIM)
    RQR = if use_small
        RQ = Matrix{T}(undef, n_states, n_shocks)
        _small_mul!(RQ, ss.impact, ss.Q)
        out = Matrix{T}(undef, n_states, n_states)
        _small_mul_Bt!(out, RQ, ss.impact)
        out
    else
        ss.impact * ss.Q * ss.impact'
    end
    P = if maximum(abs, eigvals(ss.G1)) >= one(T) - T(1e-6)
        _diffuse_initial_covariance(ss.G1, RQR)
    else
        solve_lyapunov(ss.G1, ss.impact)
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
    F_L = zeros(T, n_obs, n_obs)        # lower Cholesky factor for small-matrix path
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

    @inbounds for t in 1:T_obs
        # --- Prediction step ---
        if use_small
            _small_mul_vec!(x_pred, ss.G1, x)
            _small_mul!(G1P, ss.G1, P)
            _small_mul_Bt!(P_pred, G1P, ss.G1)
        else
            mul!(x_pred, ss.G1, x)
            mul!(G1P, ss.G1, P)
            mul!(P_pred, G1P, ss.G1')
        end
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
        elseif use_small
            # --- Lock-free pure-Julia small-matrix full observation path (#947) ---
            _small_mul_vec!(Zx, ss.Z, x_pred)
            @inbounds for i in 1:n_obs
                v[i] = y_t[i] - Zx[i] - ss.d[i]
            end

            _small_mul!(ZP, ss.Z, P_pred)
            _small_mul_Bt!(F, ZP, ss.Z)
            @inbounds for j in 1:n_obs, i in 1:n_obs
                F[i, j] += ss.H[i, j]
            end
            @inbounds for j in 1:n_obs, i in 1:j-1
                avg = (F[i, j] + F[j, i]) / 2
                F[i, j] = avg
                F[j, i] = avg
            end

            ok, log_det_F = _small_cholesky!(F_L, F)
            if !ok
                @inbounds for i in 1:n_obs
                    F[i, i] += T(1e-8)
                end
                ok, log_det_F = _small_cholesky!(F_L, F)
                if !ok
                    F_chol = cholesky(Hermitian(F))
                    log_det_F = logdet(F_chol)
                    copyto!(F_L, F_chol.L)
                end
            end

            _small_chol_solve_vec!(F_inv_v, F_L, v)
            quad = zero(T)
            @inbounds @simd for i in 1:n_obs
                quad += v[i] * F_inv_v[i]
            end
            ll += -T(0.5) * (n_obs * log2pi + log_det_F + quad)

            # Kalman gain: PZ = P_pred * Z', K = PZ * F^{-1}
            _small_mul_Bt!(PZ, P_pred, ss.Z)
            _small_chol_rdiv!(K, PZ, F_L)

            # State update: x = x_pred + K * v
            _small_mul_vec!(Kv, K, v)
            @inbounds @simd for i in 1:n_states
                x[i] = x_pred[i] + Kv[i]
            end

            # Covariance update: P = P_pred - K * ZP
            _small_mul!(KZP, K, ZP)
            @inbounds for j in 1:n_states, i in 1:n_states
                P[i, j] = P_pred[i, j] - KZP[i, j]
            end
            @inbounds for j in 1:n_states, i in 1:j-1
                avg = (P[i, j] + P[j, i]) / 2
                P[i, j] = avg
                P[j, i] = avg
            end
        else
            # --- Full observation path (BLAS/LAPACK for larger systems) ---

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
