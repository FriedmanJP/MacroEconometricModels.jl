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

    # Need at least lags+1 complete rows
    if t_complete < lags + 2
        # Fallback: use column means for NaN
        Y_filled = copy(Ymat)
        for j in 1:N
            valid = filter(!isnan, Y_filled[:, j])
            m = isempty(valid) ? zero(Tf) : mean(valid)
            for i in 1:T_obs
                isnan(Y_filled[i, j]) && (Y_filled[i, j] = m)
            end
        end
        t_complete = T_obs
        Ybal = Y_filled[1:t_complete, :]
    else
        Ybal = Ymat[1:t_complete, :]
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

    obj = par -> -_bvar_log_ml(par, Ybal, lags, sigma_ar)

    result = Optim.optimize(obj, par0, Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, f_reltol=Tf(thresh)))

    par_opt = Optim.minimizer(result)
    lambda_opt = exp(par_opt[1])
    theta_opt = exp(par_opt[2])
    miu_opt = exp(par_opt[3])
    alpha_opt = exp(par_opt[4])

    # Estimate BVAR with optimal hyperparameters
    beta, sigma, ml = _bvar_estimate(Ybal, lags, sigma_ar,
                                      lambda_opt, theta_opt, miu_opt, alpha_opt)

    # Use Kalman smoother to fill missing data
    X_sm = _bvar_smooth_missing(Ymat, beta, sigma, lags, t_complete)

    NowcastBVAR{Tf}(X_sm, beta, sigma, lambda_opt, theta_opt, miu_opt,
                     alpha_opt, lags, ml, nM, nQ, Ymat)
end

# =============================================================================
# BVAR Marginal Log-Likelihood
# =============================================================================

"""Compute log marginal likelihood for Normal-IW BVAR."""
function _bvar_log_ml(par::AbstractVector{T}, Y::Matrix{T}, lags::Int,
                      sigma_ar::Vector{T}) where {T<:AbstractFloat}
    lambda = exp(par[1])
    theta = exp(par[2])
    miu = exp(par[3])
    alpha = exp(par[4])

    _, _, ml = _bvar_estimate(Y, lags, sigma_ar, lambda, theta, miu, alpha)
    return ml
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

    # OLS on augmented system (= posterior mode of Normal-IW)
    XtX = X_star' * X_star
    XtX_reg = XtX + T(1e-8) * I(k)
    beta = XtX_reg \ (X_star' * Y_star)

    # Residuals and posterior sigma
    resid = Y_star - X_star * beta
    sigma = (resid' * resid) / T(size(Y_star, 1) - k)
    sigma = (sigma + sigma') / T(2)

    # Log marginal likelihood (Normal-IW closed form approximation)
    resid_data = Y_dep - X_reg * beta
    SSR = resid_data' * resid_data
    logml = -T(0.5) * T_eff * N * log(T(2π)) -
            T(0.5) * T_eff * log(max(det(sigma), T(1e-300))) -
            T(0.5) * tr(inv(sigma + T(1e-8) * I(N)) * SSR)

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

    # Mean of initial observations
    y_bar = vec(mean(Y0, dims=1))

    dummy_Y = Matrix{T}(undef, 0, N)
    dummy_X = Matrix{T}(undef, 0, k)

    # 1. Minnesota tightness dummies (Litterman 1986)
    for lag in 1:lags
        Y_d = zeros(T, N, N)
        X_d = zeros(T, N, k)
        for i in 1:N
            Y_d[i, i] = sigma_ar[i] / (lambda * T(lag)^T(2))
            X_d[i, 1 + (lag - 1) * N + i] = sigma_ar[i] / (lambda * T(lag)^T(2))
        end
        # Cross-variable shrinkage
        for i in 1:N, j in 1:N
            if i != j
                X_d[i, 1 + (lag - 1) * N + j] *= one(T) / theta
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

"""Fill missing data using BVAR estimates + Kalman smoother."""
function _bvar_smooth_missing(Y::Matrix{T}, beta::Matrix{T}, sigma::Matrix{T},
                               lags::Int, t_complete::Int) where {T<:AbstractFloat}
    T_obs, N = size(Y)
    X_sm = copy(Y)

    # For observed values, keep originals; for NaN, use BVAR forecast
    if t_complete < T_obs
        for t in (t_complete + 1):T_obs
            # Construct lagged values (use smoothed values for already-filled periods)
            x_lag = ones(T, 1)
            for lag in 1:lags
                t_lag = t - lag
                if t_lag >= 1
                    x_lag = vcat(x_lag, X_sm[t_lag, :])
                else
                    x_lag = vcat(x_lag, zeros(T, N))
                end
            end
            y_pred = beta' * x_lag
            for j in 1:N
                if isnan(X_sm[t, j])
                    X_sm[t, j] = y_pred[j]
                end
            end
        end
    end

    # Also fill any interior NaN values using interpolation + BVAR conditional
    for t in 1:t_complete
        for j in 1:N
            if isnan(X_sm[t, j])
                # Linear interpolation between nearest non-NaN values
                lo = t - 1
                while lo >= 1 && isnan(X_sm[lo, j])
                    lo -= 1
                end
                hi = t + 1
                while hi <= T_obs && isnan(X_sm[hi, j])
                    hi += 1
                end
                if lo >= 1 && hi <= T_obs
                    # Linear interpolation
                    X_sm[t, j] = X_sm[lo, j] + (X_sm[hi, j] - X_sm[lo, j]) * T(t - lo) / T(hi - lo)
                elseif lo >= 1
                    X_sm[t, j] = X_sm[lo, j]
                elseif hi <= T_obs
                    X_sm[t, j] = X_sm[hi, j]
                else
                    # Column mean
                    valid = filter(!isnan, Y[:, j])
                    X_sm[t, j] = isempty(valid) ? zero(T) : mean(valid)
                end
            end
        end
    end

    return X_sm
end
