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
Minnesota prior dummy observations and marginal likelihood for Bayesian VAR.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Dummy Observation Generation
# =============================================================================

"""
    gen_dummy_obs(Y, p, hyper) -> (Y_dummy, X_dummy)

Generate Minnesota prior dummy observations.
Hyperparameters: tau (tightness), decay, lambda (sum-of-coef), mu (co-persistence), omega.
"""
function gen_dummy_obs(Y::AbstractMatrix{T}, p::Int, hyper::MinnesotaHyperparameters) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    k = 1 + n * p

    sigmas = [univariate_ar_variance(@view Y[:, i]) for i in 1:n]
    y_bar = vec(mean(@view(Y[1:min(p, T_obs), :]), dims=1))

    tau, d, lambda, mu, omega = T(hyper.tau), T(hyper.decay), T(hyper.lambda), T(hyper.mu), T(hyper.omega)

    # Collect dummy blocks
    blocks_Y, blocks_X = Matrix{T}[], Matrix{T}[]

    # AR coefficient prior (diagonal shrinkage)
    push!(blocks_Y, _ar_prior_Y(n, p, sigmas, tau, d))
    push!(blocks_X, _ar_prior_X(n, p, k, sigmas, tau, d))

    # Sum-of-coefficients prior
    lambda > 0 && (push!(blocks_Y, _soc_Y(n, y_bar, lambda)); push!(blocks_X, _soc_X(n, p, k, y_bar, lambda)))

    # Dummy initial observation (co-persistence)
    mu > 0 && (push!(blocks_Y, _dio_Y(n, y_bar, mu)); push!(blocks_X, _dio_X(n, p, k, y_bar, mu)))

    # Covariance prior
    omega > 0 && (push!(blocks_Y, diagm(sigmas)); push!(blocks_X, zeros(T, n, k)))

    vcat(blocks_Y...), vcat(blocks_X...)
end

@float_fallback gen_dummy_obs Y

# Helper functions for dummy observation blocks
function _ar_prior_Y(n, p, sigmas::Vector{T}, tau, d) where {T}
    Y = zeros(T, n * p, n)
    row = 0
    @inbounds for lag in 1:p, i in 1:n
        row += 1
        lag == 1 && (Y[row, i] = sigmas[i] * T(lag)^d / tau)
    end
    Y
end

function _ar_prior_X(n, p, k, sigmas::Vector{T}, tau, d) where {T}
    X = zeros(T, n * p, k)
    row = 0
    @inbounds for lag in 1:p, i in 1:n
        row += 1
        X[row, 1 + (lag-1)*n + i] = sigmas[i] * T(lag)^d / tau
    end
    X
end

function _soc_Y(n, y_bar::Vector{T}, lambda) where {T}
    diagm(y_bar ./ lambda)
end

function _soc_X(n, p, k, y_bar::Vector{T}, lambda) where {T}
    X = zeros(T, n, k)
    @inbounds for i in 1:n, lag in 1:p
        X[i, 1 + (lag-1)*n + i] = y_bar[i] / lambda
    end
    X
end

_dio_Y(n, y_bar::Vector{T}, mu) where {T} = reshape(y_bar ./ mu, 1, n)

function _dio_X(n, p, k, y_bar::Vector{T}, mu) where {T}
    X = zeros(T, 1, k)
    X[1, 1] = one(T) / mu
    @inbounds for lag in 1:p
        X[1, (2 + (lag-1)*n):(1 + lag*n)] .= y_bar ./ mu
    end
    X
end

# =============================================================================
# Marginal Likelihood
# =============================================================================

"""
    log_marginal_likelihood(Y, p, hyper) -> T

Closed-form log marginal likelihood for BVAR with Minnesota prior.
"""
function log_marginal_likelihood(Y::AbstractMatrix{T}, p::Int, hyper::MinnesotaHyperparameters) where {T<:AbstractFloat}
    n = size(Y, 2)

    Y_d, X_d = gen_dummy_obs(Y, p, hyper)
    T_d = size(Y_d, 1)

    Y_eff, X = construct_var_matrices(Y, p)
    T_eff, k = size(Y_eff, 1), size(X, 2)

    # Augmented data
    Y_aug, X_aug = vcat(Y_d, Y_eff), vcat(X_d, X)
    K_post, K_prior = X_aug'X_aug, X_d'X_d

    # OLS on augmented and prior data
    B_aug = robust_inv(K_post) * (X_aug' * Y_aug)
    B_prior = robust_inv(K_prior) * (X_d' * Y_d)
    S_post = (Y_aug - X_aug * B_aug)' * (Y_aug - X_aug * B_aug)
    S_prior = (Y_d - X_d * B_prior)' * (Y_d - X_d * B_prior)

    nu_prior = T_d - k
    nu_prior <= n - 1 && (@warn "Prior dof too low"; return T(-Inf))
    nu_post = T_eff + nu_prior

    T(0.5) * n * (logdet_safe(K_prior) - logdet_safe(K_post)) -
    T(0.5) * nu_post * logdet_safe(S_post) +
    T(0.5) * nu_prior * logdet_safe(S_prior)
end

@float_fallback log_marginal_likelihood Y

# =============================================================================
# Hyperparameter Optimization
# =============================================================================

"""Optimize tau via grid search on marginal likelihood."""
function optimize_hyperparameters(Y::AbstractMatrix{T}, p::Int;
    grid_size::Int=20, tau_range::Tuple{Real,Real}=(0.01, 10.0)
) where {T<:AbstractFloat}
    taus = range(T(tau_range[1]), T(tau_range[2]), length=grid_size)
    best_tau, best_ml = T(tau_range[1]), T(-Inf)

    for tau in taus
        ml = log_marginal_likelihood(Y, p, MinnesotaHyperparameters(; tau))
        ml > best_ml && (best_ml = ml; best_tau = tau)
    end
    MinnesotaHyperparameters(; tau=best_tau)
end

@float_fallback optimize_hyperparameters Y

"""Full grid search over tau, lambda, mu."""
function optimize_hyperparameters_full(Y::AbstractMatrix{T}, p::Int;
    tau_grid=range(T(0.1), T(5.0), length=10),
    lambda_grid=T.([1.0, 5.0, 10.0]),
    mu_grid=T.([1.0, 2.0, 5.0])
) where {T<:AbstractFloat}

    best_hyper = MinnesotaHyperparameters(; tau=first(tau_grid), lambda=first(lambda_grid), mu=first(mu_grid))
    best_ml = T(-Inf)

    for tau in tau_grid, lambda in lambda_grid, mu in mu_grid
        hyper = MinnesotaHyperparameters(; tau=T(tau), lambda=T(lambda), mu=T(mu))
        ml = log_marginal_likelihood(Y, p, hyper)
        ml > best_ml && (best_ml = ml; best_hyper = hyper)
    end
    best_hyper, best_ml
end
