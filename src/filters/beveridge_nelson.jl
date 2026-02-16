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
Beveridge-Nelson (1981) trend-cycle decomposition.

Decomposes an I(1) process into a permanent (random walk + drift) component
and a stationary transitory component using the ARIMA representation.

Reference: Beveridge, Stephen, and Charles R. Nelson. 1981.
"A New Approach to Decomposition of Economic Time Series into Permanent and
Transitory Components with Particular Attention to Measurement of the
'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151–174.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    beveridge_nelson(y::AbstractVector; method=:arima, p=:auto, q=:auto, max_terms=500, cycle_order=2) -> BeveridgeNelsonResult

Compute the Beveridge-Nelson decomposition of time series `y`.

Assumes `y` is I(1) and decomposes it into:
```math
y_t = \\tau_t + c_t
```
where ``\\tau_t`` is a random walk with drift (permanent component)
and ``c_t`` is a stationary transitory component.

# Methods
- `:arima` (default): Classic BN via ARIMA representation (Beveridge & Nelson, 1981)
- `:statespace`: Correlated UC model via MLE + Kalman smoother (Morley, Nelson & Zivot, 2003)

# Arguments
- `y::AbstractVector`: Time series data (assumed I(1), length ≥ 10)

# Keywords
- `method::Symbol=:arima`: Decomposition method
- `p`: AR order for ARMA model of Δy (method=:arima). `:auto` uses `auto_arima`
- `q`: MA order for ARMA model of Δy (method=:arima). `:auto` uses `auto_arima`
- `max_terms::Int=500`: Maximum ψ-weights for MA(∞) truncation (method=:arima)
- `cycle_order::Int=2`: AR order for cyclical component (method=:statespace, 1 or 2)

# Returns
- `BeveridgeNelsonResult{T}` with fields `permanent`, `transitory`, `drift`,
  `long_run_multiplier`, `arima_order`, `T_obs`

# Examples
```julia
# Classic ARIMA-based BN decomposition
y = cumsum(randn(200)) + 0.3 * sin.(2π * (1:200) / 20)
result = beveridge_nelson(y)
result = beveridge_nelson(y; p=2, q=1)  # manual ARMA order

# Morley (2002) correlated UC model
result = beveridge_nelson(y; method=:statespace)
result = beveridge_nelson(y; method=:statespace, cycle_order=1)
```

# References
- Beveridge, S., & Nelson, C. R. (1981). *JME* 7(2): 151–174.
- Morley, J. C., Nelson, C. R., & Zivot, E. (2003). *REStat* 85(2): 235–243.
"""
function beveridge_nelson(y::AbstractVector{T}; method::Symbol=:arima, p=:auto, q=:auto,
                          max_terms::Int=500, cycle_order::Int=2) where {T<:AbstractFloat}
    T_obs = length(y)
    T_obs < 10 && throw(ArgumentError("BN decomposition requires at least 10 observations, got $T_obs"))

    yv = Vector{T}(y)

    if method == :statespace
        return _beveridge_nelson_statespace(yv; ar_order=cycle_order)
    elseif method != :arima
        throw(ArgumentError("method must be :arima or :statespace, got :$method"))
    end

    max_terms < 1 && throw(ArgumentError("max_terms must be positive, got $max_terms"))

    # First differences
    dy = diff(yv)
    drift = mean(dy)

    # Determine ARMA order for Δy (already stationary, so max_d=0)
    if p === :auto || q === :auto
        sel = auto_arima(dy; max_p=6, max_q=6, max_d=0)
        p_use = p === :auto ? ar_order(sel) : Int(p)
        q_use = q === :auto ? ma_order(sel) : Int(q)
    else
        p_use = Int(p)
        q_use = Int(q)
    end

    # Handle pure white noise case (p=0, q=0)
    if p_use == 0 && q_use == 0
        # psi(1) = 1, transitory = 0
        permanent = copy(yv)
        transitory = zeros(T, T_obs)
        return BeveridgeNelsonResult(permanent, transitory, drift, one(T),
                                     (0, 1, 0), T_obs)
    end

    # Fit ARMA model to Δy
    if p_use > 0 && q_use > 0
        arma_model = estimate_arma(dy, p_use, q_use)
        phi = arma_model.phi
        theta = arma_model.theta
        resid = arma_model.residuals
    elseif p_use > 0
        ar_model = estimate_ar(dy, p_use)
        phi = ar_model.phi
        theta = T[]
        resid = ar_model.residuals
    else
        ma_model = estimate_ma(dy, q_use)
        phi = T[]
        theta = ma_model.theta
        resid = ma_model.residuals
    end

    # Compute ψ-weights: MA(∞) representation coefficients
    psi = _compute_psi_weights(phi, theta, max_terms)

    # Long-run multiplier: ψ(1) = 1 + Σ ψ_j
    long_run = one(T) + sum(psi)

    # Cumulative ψ-star weights: ψ*_j = Σ_{i=j}^∞ ψ_i
    # (truncated at max_terms)
    psi_star = zeros(T, max_terms)
    psi_star[max_terms] = psi[max_terms]
    @inbounds for j in (max_terms - 1):-1:1
        psi_star[j] = psi[j] + psi_star[j + 1]
    end

    # Transitory component: c_t = -Σ_{j=1}^∞ ψ*_j ε_{t-j+1}
    # We only have residuals from p_use+1:end (length of resid)
    n_resid = length(resid)

    # Align residuals with the original series
    # The ARMA residuals correspond to Δy, which starts at index 2 of y
    # With p lags, residuals start at index p_use+1 of Δy = index p_use+2 of y
    transitory = zeros(T, T_obs)
    start_idx = T_obs - n_resid + 1  # index in y where residuals start

    @inbounds for t in start_idx:T_obs
        ct = zero(T)
        for j in 1:min(max_terms, t - start_idx + 1)
            resid_idx = (t - start_idx + 1) - j + 1
            if resid_idx >= 1 && resid_idx <= n_resid
                ct -= psi_star[j] * resid[resid_idx]
            end
        end
        transitory[t] = ct
    end

    # Permanent component: τ_t = y_t - c_t
    permanent = yv .- transitory

    BeveridgeNelsonResult(permanent, transitory, drift, long_run,
                           (p_use, 1, q_use), T_obs)
end

# =============================================================================
# Morley (2002) State-Space BN Decomposition
# =============================================================================

"""
State-space Beveridge-Nelson via correlated UC model (Morley, Nelson & Zivot 2003).

State vector: x_t = [τ_t, c_t, c_{t-1}]  (for AR(2) cycle)
Measurement: y_t = [1, 1, 0] x_t
Transition: x_t = [μ, 0, 0] + [[1, 0, 0]; [0, φ₁, φ₂]; [0, 1, 0]] x_{t-1} + [[η_t]; [ε_t]; [0]]
Cov([η_t; ε_t]) = [[σ²_η, σ_ηε]; [σ_ηε, σ²_ε]]
"""
function _beveridge_nelson_statespace(y::Vector{T}; ar_order::Int=2) where {T<:AbstractFloat}
    T_obs = length(y)
    (ar_order < 1 || ar_order > 2) && throw(ArgumentError("ar_order must be 1 or 2, got $ar_order"))

    # Parameter vector: [μ, φ₁, (φ₂ if ar_order==2), log(σ²_η), log(σ²_ε), atanh(ρ)]
    n_phi = ar_order
    n_params = 1 + n_phi + 3  # μ, φ's, log(σ²_η), log(σ²_ε), atanh(ρ)

    # Initial guess
    dy = diff(y)
    mu0 = mean(dy)
    var_dy = var(dy)
    theta0 = zeros(T, n_params)
    theta0[1] = mu0
    if ar_order >= 1; theta0[2] = T(0.5); end
    if ar_order >= 2; theta0[3] = T(-0.2); end
    theta0[end-2] = log(var_dy / 4)  # log(σ²_η)
    theta0[end-1] = log(var_dy / 4)  # log(σ²_ε)
    theta0[end] = T(0.0)             # atanh(ρ) = 0 → ρ = 0

    function neg_loglik(theta)
        params = _unpack_uc_params(theta, ar_order)
        isnothing(params) && return T(1e10)
        mu, phi, sig2_eta, sig2_eps, rho = params

        # Check stationarity of cycle
        if ar_order == 1
            abs(phi[1]) >= one(T) && return T(1e10)
        elseif ar_order == 2
            (abs(phi[2]) >= one(T) || phi[1] + phi[2] >= one(T) || phi[2] - phi[1] >= one(T)) && return T(1e10)
        end

        ll = _uc_kalman_loglik(y, mu, phi, sig2_eta, sig2_eps, rho, ar_order)
        isfinite(ll) ? -ll : T(1e10)
    end

    # Optimize
    result = Optim.optimize(neg_loglik, theta0, Optim.NelderMead(),
                            Optim.Options(iterations=5000, f_reltol=T(1e-8)))
    theta_hat = Optim.minimizer(result)

    mu, phi, sig2_eta, sig2_eps, rho = _unpack_uc_params(theta_hat, ar_order)

    # Run Kalman smoother to extract states
    tau, cyc = _uc_kalman_smoother(y, mu, phi, sig2_eta, sig2_eps, rho, ar_order)

    drift = mu
    long_run = one(T)  # UC model: no direct long-run multiplier like ARIMA BN

    BeveridgeNelsonResult(tau, cyc, drift, long_run, (ar_order, 0, 0), T_obs)
end

"""Unpack and transform UC model parameters."""
function _unpack_uc_params(theta::AbstractVector{T}, ar_order::Int) where {T}
    n_phi = ar_order
    mu = theta[1]
    phi = theta[2:(1+n_phi)]
    log_sig2_eta = theta[end-2]
    log_sig2_eps = theta[end-1]
    atanh_rho = theta[end]

    sig2_eta = exp(log_sig2_eta)
    sig2_eps = exp(log_sig2_eps)
    rho = tanh(atanh_rho)

    # Bounds check
    (sig2_eta <= zero(T) || sig2_eps <= zero(T)) && return nothing

    (mu, phi, sig2_eta, sig2_eps, rho)
end

"""Kalman filter log-likelihood for the UC model."""
function _uc_kalman_loglik(y::Vector{T}, mu::T, phi::Vector{T},
                           sig2_eta::T, sig2_eps::T, rho::T,
                           ar_order::Int) where {T<:AbstractFloat}
    T_obs = length(y)
    ns = 1 + ar_order

    # Build system matrices
    Z = zeros(T, 1, ns)          # measurement
    Z[1, 1] = one(T)             # τ
    Z[1, 2] = one(T)             # c

    TT = zeros(T, ns, ns)        # transition
    TT[1, 1] = one(T)            # τ_t = τ_{t-1} + ...
    TT[2, 2] = phi[1]            # c_t = φ₁c_{t-1} + ...
    if ar_order == 2
        TT[2, 3] = phi[2]        # c_t = ... + φ₂c_{t-2}
        TT[3, 2] = one(T)        # c_{t-1} = c_{t-1}
    end

    c_vec = zeros(T, ns)          # state intercept
    c_vec[1] = mu

    # Shock covariance (only τ and c have shocks)
    sig_eta_eps = rho * sqrt(sig2_eta * sig2_eps)
    Q_cov = zeros(T, ns, ns)
    Q_cov[1, 1] = sig2_eta
    Q_cov[2, 2] = sig2_eps
    Q_cov[1, 2] = sig_eta_eps
    Q_cov[2, 1] = sig_eta_eps

    # Initialize state
    a = zeros(T, ns)
    a[1] = y[1]

    P = zeros(T, ns, ns)
    P[1, 1] = T(1e6)  # diffuse prior for τ
    if ar_order == 1
        denom = max(one(T) - phi[1]^2, T(0.01))
        P[2, 2] = sig2_eps / denom
    elseif ar_order == 2
        A_c = zeros(T, 2, 2)
        A_c[1, 1] = phi[1]; A_c[1, 2] = phi[2]
        A_c[2, 1] = one(T)
        Q_c = zeros(T, 2, 2)
        Q_c[1, 1] = sig2_eps
        P_c = _solve_lyapunov_2x2(A_c, Q_c)
        P[2:3, 2:3] = P_c
    end

    # Kalman filter
    ll = zero(T)
    for t in 1:T_obs
        # Prediction error
        v = y[t] - dot(Z[1, :], a)
        F = dot(Z[1, :], P * Z[1, :])

        F < T(1e-12) && return T(-Inf)

        # Log-likelihood contribution
        ll -= T(0.5) * (log(T(2π)) + log(F) + v^2 / F)

        # Kalman gain
        K = (TT * P * Z') / F

        # Update
        a = c_vec + TT * a + K[:, 1] * v
        P = TT * P * TT' + Q_cov - K * F * K'
        P = (P + P') / 2  # symmetrize
    end

    ll
end

"""Solve 2×2 discrete Lyapunov equation P = A P A' + Q."""
function _solve_lyapunov_2x2(A::Matrix{T}, Q::Matrix{T}) where {T}
    AA = kron(A, A)
    I4 = Matrix{T}(I, 4, 4)
    det_check = det(I4 - AA)
    abs(det_check) < T(1e-10) && return Matrix{T}(I, 2, 2) * T(1e4)
    p_vec = (I4 - AA) \ vec(Q)
    P = reshape(p_vec, 2, 2)
    (P + P') / 2
end

"""Kalman smoother for UC model — extract smoothed trend and cycle."""
function _uc_kalman_smoother(y::Vector{T}, mu::T, phi::Vector{T},
                              sig2_eta::T, sig2_eps::T, rho::T,
                              ar_order::Int) where {T<:AbstractFloat}
    T_obs = length(y)
    ns = 1 + ar_order

    # Build system matrices (same as in loglik)
    Z = zeros(T, 1, ns)
    Z[1, 1] = one(T); Z[1, 2] = one(T)

    TT = zeros(T, ns, ns)
    TT[1, 1] = one(T)
    TT[2, 2] = phi[1]
    if ar_order == 2
        TT[2, 3] = phi[2]
        TT[3, 2] = one(T)
    end

    c_vec = zeros(T, ns)
    c_vec[1] = mu

    sig_eta_eps = rho * sqrt(sig2_eta * sig2_eps)
    Q_cov = zeros(T, ns, ns)
    Q_cov[1, 1] = sig2_eta
    Q_cov[2, 2] = sig2_eps
    Q_cov[1, 2] = sig_eta_eps
    Q_cov[2, 1] = sig_eta_eps

    # Forward pass (Kalman filter) — store predictions
    a_pred = zeros(T, T_obs + 1, ns)
    P_pred = zeros(T, ns, ns, T_obs + 1)
    a_filt = zeros(T, T_obs, ns)
    P_filt = zeros(T, ns, ns, T_obs)

    # Initialize
    a_pred[1, 1] = y[1]
    P_pred[1, 1, 1] = T(1e6)
    if ar_order == 1
        P_pred[2, 2, 1] = sig2_eps / max(one(T) - phi[1]^2, T(0.01))
    elseif ar_order == 2
        A_c = zeros(T, 2, 2)
        A_c[1, 1] = phi[1]; A_c[1, 2] = phi[2]
        A_c[2, 1] = one(T)
        Q_c = zeros(T, 2, 2)
        Q_c[1, 1] = sig2_eps
        P_c = _solve_lyapunov_2x2(A_c, Q_c)
        P_pred[2:3, 2:3, 1] = P_c
    end

    for t in 1:T_obs
        a_t = a_pred[t, :]
        P_t = P_pred[:, :, t]

        # Innovation
        v = y[t] - dot(Z[1, :], a_t)
        F = dot(Z[1, :], P_t * Z[1, :])
        F = max(F, T(1e-12))

        # Kalman gain
        K_t = P_t * Z[1, :] / F

        # Filtered state
        a_filt[t, :] = a_t + K_t * v
        P_filt[:, :, t] = P_t - K_t * Z[1, :]' * P_t
        P_filt[:, :, t] = (P_filt[:, :, t] + P_filt[:, :, t]') / 2

        # Predicted next state
        if t < T_obs
            a_pred[t+1, :] = c_vec + TT * a_filt[t, :]
            P_pred[:, :, t+1] = TT * P_filt[:, :, t] * TT' + Q_cov
            P_pred[:, :, t+1] = (P_pred[:, :, t+1] + P_pred[:, :, t+1]') / 2
        end
    end

    # Backward pass (RTS smoother)
    a_smooth = zeros(T, T_obs, ns)
    a_smooth[T_obs, :] = a_filt[T_obs, :]

    for t in (T_obs-1):-1:1
        P_f = P_filt[:, :, t]
        P_p = TT * P_f * TT' + Q_cov
        P_p = (P_p + P_p') / 2

        # Smoother gain
        P_p_inv = robust_inv(P_p)
        J_t = P_f * TT' * P_p_inv

        a_smooth[t, :] = a_filt[t, :] + J_t * (a_smooth[t+1, :] - c_vec - TT * a_filt[t, :])
    end

    # Extract trend and cycle
    tau = a_smooth[:, 1]
    cyc = a_smooth[:, 2]

    (tau, cyc)
end

# Float64 fallback for non-float input
beveridge_nelson(y::AbstractVector; kwargs...) = beveridge_nelson(Float64.(y); kwargs...)
