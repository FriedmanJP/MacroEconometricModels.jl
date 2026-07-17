# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Large BVAR nowcasting with GLP-style priors (Cimadomo et al. 2022).

Estimates a mixed-frequency BVAR with Normal-Inverse-Wishart prior.
Hyperparameters (lambda, theta, miu, alpha) optimized via marginal
log-likelihood maximization.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_bvar(Y, nM, nQ; lags=5, thresh=1e-6, max_iter=200,
                lambda0=0.2, theta0=1.0, miu0=1.0, alpha0=2.0) -> NowcastBVAR{T}

Estimate a large BVAR for mixed-frequency nowcasting.

The first `nM` columns are monthly variables; the next `nQ` columns are
quarterly (observed every 3rd month). The BVAR is estimated on the complete
(non-NaN) portion, then the Kalman smoother fills the ragged edge.

# Arguments
- `Y::AbstractMatrix` — T_obs × N data matrix (NaN for missing)
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables

# Keyword Arguments
- `lags::Int=5` — number of lags
- `thresh::Real=1e-6` — optimization convergence threshold
- `max_iter::Int=200` — max optimization iterations
- `lambda0::Real=0.2` — initial overall shrinkage
- `theta0::Real=1.0` — initial cross-variable shrinkage
- `miu0::Real=1.0` — initial sum-of-coefficients weight
- `alpha0::Real=2.0` — initial co-persistence weight

# Returns
`NowcastBVAR{T}` with smoothed data and posterior parameters.

# References
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F. & Sokol, A. (2022).
  Nowcasting with Large Bayesian Vector Autoregressions.
"""
function nowcast_bvar(Y::AbstractMatrix, nM::Int, nQ::Int;
                      lags::Int=5, thresh::Real=1e-6, max_iter::Int=200,
                      lambda0::Real=0.2, theta0::Real=1.0,
                      miu0::Real=1.0, alpha0::Real=2.0)
    T_obs, N = size(Y)
    N == nM + nQ || throw(ArgumentError("nM ($nM) + nQ ($nQ) must equal number of columns ($N)"))
    lags >= 1 || throw(ArgumentError("lags must be >= 1, got $lags"))

    Tf = eltype(Y) <: AbstractFloat ? eltype(Y) : Float64
    Ymat = Matrix{Tf}(Y)

    # Find the last complete row (no NaN)
    t_complete = T_obs
    while t_complete > 0 && any(isnan, Ymat[t_complete, :])
        t_complete -= 1
    end

    # Need at least lags+1 complete rows for VAR construction
    if t_complete < lags + 2
        t_complete = T_obs
    end

    # Fill NaN in balanced panel (quarterly columns have NaN at non-quarter months)
    Ybal = copy(Ymat[1:t_complete, :])
    for j in 1:N
        col = Ybal[:, j]
        valid_mask = .!isnan.(col)
        if any(valid_mask) && !all(valid_mask)
            m = mean(col[valid_mask])
            for i in 1:t_complete
                if isnan(Ybal[i, j])
                    Ybal[i, j] = m
                end
            end
        elseif !any(valid_mask)
            Ybal[:, j] .= zero(Tf)
        end
    end

    # Compute AR(1) residual standard deviations for prior scaling
    sigma_ar = zeros(Tf, N)
    for j in 1:N
        col = Ybal[:, j]
        valid = filter(!isnan, col)
        if length(valid) > 2
            y_dep = valid[2:end]
            y_lag = valid[1:end-1]
            b = dot(y_lag, y_dep) / max(dot(y_lag, y_lag), Tf(1e-10))
            resid_j = y_dep - b * y_lag
            sigma_ar[j] = max(std(resid_j), Tf(1e-6))
        else
            sigma_ar[j] = one(Tf)
        end
    end

    # Optimize hyperparameters via marginal log-likelihood
    par0 = [log(Tf(lambda0)), log(Tf(theta0)), log(Tf(miu0)), log(Tf(alpha0))]

    obj = par -> begin
        val = -_bvar_log_ml(par, Ybal, lags, sigma_ar)
        isfinite(val) ? val : Tf(1e10)
    end

    result = Optim.optimize(obj, par0, Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, f_reltol=Tf(thresh)))

    par_opt = Optim.minimizer(result)
    lambda_opt = exp(par_opt[1])
    theta_opt = exp(par_opt[2])
    miu_opt = exp(par_opt[3])
    alpha_opt = exp(par_opt[4])

    # The log-hyperparameter box is |par| ≤ 5; a hit at the corner (λ = exp(5) ≈ 148.4)
    # means the marginal-likelihood optimizer diverged to the boundary rather than an
    # interior optimum — flag it (the box is the documented cause and is NOT altered, which
    # would perturb the whole nowcast pipeline). (B4/T173)
    converged = !any(x -> abs(x) >= Tf(5) - Tf(1e-3), par_opt)

    # Estimate BVAR with optimal hyperparameters
    beta, sigma, ml = _bvar_estimate(Ybal, lags, sigma_ar,
                                      lambda_opt, theta_opt, miu_opt, alpha_opt)

    # Use Kalman smoother to fill missing data
    X_sm = _bvar_smooth_missing(Ymat, beta, sigma, lags, t_complete)

    NowcastBVAR{Tf}(X_sm, beta, sigma, lambda_opt, theta_opt, miu_opt,
                     alpha_opt, lags, ml, nM, nQ, Ymat, converged)
end

# =============================================================================
# BVAR Marginal Log-Likelihood
# =============================================================================

"""Compute log marginal likelihood for Normal-IW BVAR."""
function _bvar_log_ml(par::AbstractVector{T}, Y::Matrix{T}, lags::Int,
                      sigma_ar::Vector{T}) where {T<:AbstractFloat}
    any(x -> abs(x) > T(5), par) && return -T(1e10)
    lambda = exp(par[1])
    theta = exp(par[2])
    miu = exp(par[3])
    alpha = exp(par[4])

    _, _, ml = _bvar_estimate(Y, lags, sigma_ar, lambda, theta, miu, alpha)
    return isfinite(ml) ? ml : -T(1e10)
end

# =============================================================================
# BVAR Estimation with Minnesota-style Prior
# =============================================================================

"""
    _bvar_estimate(Y, lags, sigma_ar, lambda, theta, miu, alpha) -> (beta, sigma, logml)

Estimate BVAR with Normal-Inverse-Wishart prior.

Uses Minnesota-style dummy observations for prior implementation.
"""
function _bvar_estimate(Y::Matrix{T}, lags::Int, sigma_ar::Vector{T},
                        lambda::T, theta::T, miu::T, alpha::T) where {T<:AbstractFloat}
    T_obs, N = size(Y)

    # Construct VAR matrices
    Y_dep = Y[(lags + 1):end, :]
    T_eff = size(Y_dep, 1)
    X_reg = ones(T, T_eff, 1)  # intercept
    for lag in 1:lags
        X_reg = hcat(X_reg, Y[(lags + 1 - lag):(end - lag), :])
    end

    k = size(X_reg, 2)  # 1 + N*lags

    # Minnesota prior: dummy observations
    # Prior mean: random walk for each variable
    Y_d, X_d = _bvar_dummy_obs(Y[1:lags, :], lags, sigma_ar, lambda, theta, miu, alpha)

    # Stack data + dummy observations
    Y_star = vcat(Y_dep, Y_d)
    X_star = vcat(X_reg, X_d)

    if !all(isfinite, Y_star) || !all(isfinite, X_star)
        return zeros(T, k, N), Matrix{T}(I(N)), -T(1e10)
    end

    # OLS on augmented system (= posterior mode of Normal-IW)
    XtX = X_star' * X_star
    XtX_reg = XtX + T(1e-6) * I(k)
    beta = XtX_reg \ (X_star' * Y_star)

    if !all(isfinite, beta)
        return zeros(T, k, N), Matrix{T}(I(N)), -T(1e10)
    end

    # Residuals and posterior sigma
    resid = Y_star - X_star * beta
    sigma = (resid' * resid) / T(size(Y_star, 1) - k)
    sigma = (sigma + sigma') / T(2)

    # Log marginal likelihood (Normal-IW closed form approximation)
    resid_data = Y_dep - X_reg * beta
    SSR = resid_data' * resid_data
    ld = logdet_safe(sigma)
    sigma_inv = Matrix{T}(robust_inv(sigma + T(1e-6) * I(N); silent=true))
    logml = -T(0.5) * T_eff * N * log(T(2π)) -
            T(0.5) * T_eff * ld -
            T(0.5) * tr(sigma_inv * SSR)

    if !isfinite(logml)
        logml = -T(1e10)
    end

    return beta, sigma, logml
end

"""
    _bvar_dummy_obs(Y0, lags, sigma_ar, lambda, theta, miu, alpha) -> (Y_d, X_d)

Construct Minnesota prior dummy observations.

- `lambda`: overall shrinkage
- `theta`: cross-variable shrinkage (1 = same as own, higher = more shrinkage)
- `miu`: sum-of-coefficients (unit root prior)
- `alpha`: co-persistence (common stochastic trend prior)
"""
function _bvar_dummy_obs(Y0::AbstractMatrix{T}, lags::Int, sigma_ar::Vector{T},
                         lambda::T, theta::T, miu::T, alpha::T) where {T<:AbstractFloat}
    N = size(Y0, 2)
    k = 1 + N * lags

    # Mean of initial observations (NaN-safe for mixed-frequency data)
    y_bar = zeros(T, N)
    for j in 1:N
        col = Y0[:, j]
        valid = filter(!isnan, col)
        y_bar[j] = isempty(valid) ? zero(T) : mean(valid)
    end

    dummy_Y = Matrix{T}(undef, 0, N)
    dummy_X = Matrix{T}(undef, 0, k)

    # 1. Minnesota tightness dummies (Litterman 1986)
    for lag in 1:lags
        Y_d = zeros(T, N, N)
        X_d = zeros(T, N, k)
        for i in 1:N
            # Diagonal (own-lag): standard Minnesota dummy
            Y_d[i, i] = sigma_ar[i] / (lambda * T(lag)^T(2))
            X_d[i, 1 + (lag - 1) * N + i] = sigma_ar[i] / (lambda * T(lag)^T(2))
            # Off-diagonal (cross-variable): shrunk by theta
            for j in 1:N
                if i != j
                    X_d[i, 1 + (lag - 1) * N + j] = sigma_ar[i] / (theta * lambda * T(lag)^T(2))
                end
            end
        end
        dummy_Y = vcat(dummy_Y, Y_d)
        dummy_X = vcat(dummy_X, X_d)
    end

    # 2. Sum-of-coefficients prior (unit root)
    if miu > 0
        Y_d = zeros(T, N, N)
        X_d = zeros(T, N, k)
        for i in 1:N
            Y_d[i, i] = y_bar[i] / miu
            for lag in 1:lags
                X_d[i, 1 + (lag - 1) * N + i] = y_bar[i] / miu
            end
        end
        dummy_Y = vcat(dummy_Y, Y_d)
        dummy_X = vcat(dummy_X, X_d)
    end

    # 3. Co-persistence prior (common stochastic trend)
    if alpha > 0
        Y_d = y_bar' / alpha
        X_d = zeros(T, 1, k)
        X_d[1, 1] = one(T) / alpha  # intercept
        for lag in 1:lags
            X_d[1, (1 + (lag - 1) * N + 1):(1 + lag * N)] = y_bar' / alpha
        end
        dummy_Y = vcat(dummy_Y, Y_d)
        dummy_X = vcat(dummy_X, X_d)
    end

    return dummy_Y, dummy_X
end

# =============================================================================
# Kalman Smoother for Ragged Edge
# =============================================================================

"""
    _bvar_smooth_missing(Y, beta, sigma, lags, t_complete) -> Matrix

Fill the missing entries of the panel `Y` (interior NaNs and the ragged edge) with a genuine
Kalman smoother. The estimated BVAR(`lags`) is cast in companion state-space form
(state `[y_t; …; y_{t-lags+1}]`, transition from the lag blocks of `beta`, state-noise `sigma`,
observation `C = [I 0]`, tiny measurement ridge) and the missing-data filter/RTS smoother from
`kalman_missing.jl` is run on the mean-centred panel. Because that smoother drops only the
missing rows each period, contemporaneously OBSERVED variables update the unobserved states
through the state covariance — so a released series informs the fill of an unreleased one,
which the previous interpolation + deterministic-projection routine ignored. `t_complete` is
retained for signature compatibility; the smoother fills every missing entry uniformly.
"""
function _bvar_smooth_missing(Y::Matrix{T}, beta::Matrix{T}, sigma::Matrix{T},
                               lags::Int, t_complete::Int) where {T<:AbstractFloat}
    T_obs, N = size(Y)
    sd = N * lags

    # Companion state-space form of the BVAR. beta is (1 + N*lags) × N (row 1 = intercept,
    # then lag blocks); B[i][j,m] = coefficient of y_{t-i}[m] in equation j.
    c = Vector{T}(beta[1, :])
    B = [permutedims(Matrix{T}(beta[(2 + (i - 1) * N):(1 + i * N), :])) for i in 1:lags]
    A = zeros(T, sd, sd)
    for i in 1:lags
        A[1:N, ((i - 1) * N + 1):(i * N)] = B[i]
    end
    lags > 1 && (A[(N + 1):sd, 1:(sd - N)] = Matrix{T}(I, sd - N, sd - N))
    Q = zeros(T, sd, sd); Q[1:N, 1:N] = T(0.5) * (sigma + sigma')
    C = zeros(T, N, sd); C[1:N, 1:N] = Matrix{T}(I, N, N)
    R = Matrix{T}(I, N, N) * T(1e-8)              # observed series measured (nearly) exactly

    # Centre by the steady-state mean so the centred VAR is intercept-free (mean-zero);
    # fall back to per-column observed means if I - ΣBᵢ is near-singular (near unit root).
    mu_emp = [begin v = filter(!isnan, @view Y[:, j]); isempty(v) ? zero(T) : T(mean(v)) end for j in 1:N]
    Imb = Matrix{T}(I, N, N) - sum(B)
    mu = let ss = try Imb \ c catch; fill(T(Inf), N) end
        (all(isfinite, ss) && maximum(abs, ss) < T(1e6)) ? ss : mu_emp
    end

    Yc = Y .- mu'                                  # NaN stays NaN → dropped per period by _miss_data
    x0 = zeros(T, sd)
    P0 = _compute_unconditional_covariance(A, Q, sd)
    all(isfinite, P0) || (P0 = Matrix{T}(I, sd, sd) * T(1e6))
    x_smooth, _, _, _ = _kalman_smoother_missing(Matrix{T}(Yc'), A, C, Q, R, x0, P0)

    # Fill only the missing entries with the (un-centred) smoothed current-period state.
    X_sm = copy(Y)
    @inbounds for t in 1:T_obs, j in 1:N
        isnan(Y[t, j]) && (X_sm[t, j] = mu[j] + x_smooth[j, t])
    end
    X_sm
end
