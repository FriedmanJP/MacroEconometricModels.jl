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
Propensity Score LP (Angrist et al. 2018).
"""

using LinearAlgebra, Statistics, Distributions

"""
    estimate_propensity_score(treatment::AbstractVector{Bool}, X::AbstractMatrix{T};
                              method::Symbol=:logit) -> Vector{T}

Estimate propensity scores P(D=1|X) via logit or probit.
"""
function estimate_propensity_score(treatment::AbstractVector{Bool}, X::AbstractMatrix{T};
                                   method::Symbol=:logit) where {T<:AbstractFloat}
    n = length(treatment)
    X_aug = hcat(ones(T, n), X)
    k = size(X_aug, 2)
    y = T.(treatment)
    beta = zeros(T, k)

    for iter in 1:50
        eta = X_aug * beta
        p, dp = if method == :logit
            (@. 1 / (1 + exp(-eta)), @. 1 / (1 + exp(-eta)) * (1 - 1 / (1 + exp(-eta))))
        elseif method == :probit
            (@. cdf(Normal(), eta), @. pdf(Normal(), eta))
        else
            throw(ArgumentError("method must be :logit or :probit"))
        end

        p = clamp.(p, T(1e-10), T(1 - 1e-10))
        dp = max.(dp, T(1e-10))

        score = X_aug' * (y .- p)
        hessian = X_aug' * Diagonal(dp) * X_aug
        delta = hessian \ score

        maximum(abs.(delta)) < 1e-8 && break
        beta .+= delta
    end

    eta = X_aug * beta
    p = method == :logit ? (@. 1 / (1 + exp(-eta))) : (@. cdf(Normal(), eta))
    clamp.(p, T(1e-10), T(1 - 1e-10))
end

estimate_propensity_score(treatment::AbstractVector{<:Integer}, X::AbstractMatrix{T};
                          method::Symbol=:logit) where {T<:AbstractFloat} =
    estimate_propensity_score(treatment .!= 0, X; method=method)

"""
    inverse_propensity_weights(treatment::AbstractVector{Bool}, propensity::AbstractVector{T};
                               trimming::Tuple{T,T}=(T(0.01), T(0.99)), normalize::Bool=true) -> Vector{T}

Compute IPW weights with optional trimming.
"""
function inverse_propensity_weights(treatment::AbstractVector{Bool}, propensity::AbstractVector{T};
                                    trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                                    normalize::Bool=true) where {T<:AbstractFloat}
    p_trimmed = clamp.(propensity, trimming[1], trimming[2])
    weights = [t ? one(T) / p : one(T) / (one(T) - p) for (t, p) in zip(treatment, p_trimmed)]

    if normalize
        for (idx, mask) in [(findall(treatment), true), (findall(.!treatment), false)]
            isempty(idx) && continue
            w_sum = sum(weights[idx])
            weights[idx] ./= w_sum
            weights[idx] .*= length(idx)
        end
    end
    weights
end

"""
    estimate_propensity_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                           covariates::AbstractMatrix{T}, horizon::Int; ...) -> PropensityLPModel{T}

Estimate LP with Inverse Propensity Weighting (Angrist et al. 2018).
"""
function estimate_propensity_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                                covariates::AbstractMatrix{T}, horizon::Int;
                                ps_method::Symbol=:logit,
                                trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                                lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                                cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    @assert length(treatment) == T_obs

    propensity = estimate_propensity_score(treatment, covariates; method=ps_method)
    ipw_weights = inverse_propensity_weights(treatment, propensity; trimming=trimming)
    config = PropensityScoreConfig{T}(ps_method, trimming, true)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)
    n_cov = size(covariates, 2)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_vec = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)
    ate = Matrix{T}(undef, horizon + 1, n_response)
    ate_se = Matrix{T}(undef, horizon + 1, n_response)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)
        D_h = treatment[t_start:t_end]
        w_h = ipw_weights[t_start:t_end]

        # Build regressor matrix: [1, D, lagged_Y, covariates]
        k = 2 + n * lags + n_cov
        X_h = Matrix{T}(undef, T_h, k)

        @inbounds for (i, t) in enumerate(t_start:t_end)
            X_h[i, 1] = one(T)
            X_h[i, 2] = D_h[i] ? one(T) : zero(T)
            col = 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = Y[t - lag, var]
                col += 1
            end
            for cov in 1:n_cov
                X_h[i, col] = covariates[t, cov]
                col += 1
            end
        end

        # Weighted least squares
        W = Diagonal(w_h)
        XtWX_inv = robust_inv(X_h' * W * X_h)
        B_h = XtWX_inv * (X_h' * W * Y_h)
        U_h = Y_h - X_h * B_h

        B[h + 1] = B_h
        residuals_store[h + 1] = U_h

        for (j, _) in enumerate(response_vars)
            ate[h + 1, j] = B_h[2, j]
        end

        # Robust covariance with weights (horizon-aware for NW)
        V_h = zeros(T, k * n_response, k * n_response)
        for eq in 1:n_response
            weighted_u = sqrt.(w_h) .* @view(U_h[:, eq])
            weighted_X = sqrt.(w_h) .* X_h
            V_eq = if cov_estimator isa NeweyWestEstimator && cov_estimator.bandwidth == 0 && h > 0
                auto_bw = optimal_bandwidth_nw(weighted_u)
                effective_bw = max(auto_bw, h + 1)
                newey_west(weighted_X, weighted_u; bandwidth=effective_bw,
                          kernel=cov_estimator.kernel, prewhiten=cov_estimator.prewhiten)
            else
                robust_vcov(weighted_X, weighted_u, cov_estimator)
            end
            idx = ((eq-1)*k + 1):(eq*k)
            V_h[idx, idx] .= V_eq
        end
        vcov_vec[h + 1] = V_h

        for (j, _) in enumerate(response_vars)
            ate_se[h + 1, j] = sqrt(V_h[(j - 1) * k + 2, (j - 1) * k + 2])
        end
    end

    PropensityLPModel{T}(Matrix{T}(Y), treatment, response_vars, Matrix{T}(covariates),
                         horizon, propensity, ipw_weights, B, residuals_store, vcov_vec,
                         ate, ate_se, config, T_eff, cov_estimator)
end

estimate_propensity_lp(Y::AbstractMatrix, treatment::AbstractVector, covariates::AbstractMatrix,
                       horizon::Int; kwargs...) =
    estimate_propensity_lp(Float64.(Y), Bool.(treatment .!= 0), Float64.(covariates), horizon; kwargs...)

"""
    doubly_robust_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                     covariates::AbstractMatrix{T}, horizon::Int; ...) -> PropensityLPModel{T}

Doubly robust LP estimator combining IPW and regression adjustment.
"""
function doubly_robust_lp(Y::AbstractMatrix{T}, treatment::AbstractVector{Bool},
                          covariates::AbstractMatrix{T}, horizon::Int;
                          ps_method::Symbol=:logit, trimming::Tuple{T,T}=(T(0.01), T(0.99)),
                          lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                          cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_response = length(response_vars)
    n_cov = size(covariates, 2)

    propensity = estimate_propensity_score(treatment, covariates; method=ps_method)
    p_trimmed = clamp.(propensity, trimming[1], trimming[2])
    ipw_weights = inverse_propensity_weights(treatment, propensity; trimming=trimming)
    config = PropensityScoreConfig{T}(ps_method, trimming, true)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_vec = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)
    ate = Matrix{T}(undef, horizon + 1, n_response)
    ate_se = Matrix{T}(undef, horizon + 1, n_response)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)
        D_h = treatment[t_start:t_end]
        p_h = p_trimmed[t_start:t_end]

        # Covariates for outcome regression
        k_cov = 1 + n * lags + n_cov
        X_cov = Matrix{T}(undef, T_h, k_cov)
        @inbounds for (i, t) in enumerate(t_start:t_end)
            X_cov[i, 1] = one(T)
            col = 2
            for lag in 1:lags, var in 1:n
                X_cov[i, col] = Y[t - lag, var]
                col += 1
            end
            for cov in 1:n_cov
                X_cov[i, col] = covariates[t, cov]
                col += 1
            end
        end

        treated_idx = findall(D_h)
        control_idx = findall(.!D_h)

        ate_dr = Vector{T}(undef, n_response)
        ate_se_h = Vector{T}(undef, n_response)

        for (j, _) in enumerate(response_vars)
            y_j = Y_h[:, j]

            # Outcome regressions
            mu1 = length(treated_idx) > k_cov ?
                  X_cov * (robust_inv(X_cov[treated_idx, :]' * X_cov[treated_idx, :]) *
                           (X_cov[treated_idx, :]' * y_j[treated_idx])) :
                  fill(mean(y_j[treated_idx]), T_h)

            mu0 = length(control_idx) > k_cov ?
                  X_cov * (robust_inv(X_cov[control_idx, :]' * X_cov[control_idx, :]) *
                           (X_cov[control_idx, :]' * y_j[control_idx])) :
                  fill(mean(y_j[control_idx]), T_h)

            # Doubly robust estimator
            psi = [D_h[i] ? (y_j[i] - mu1[i]) / p_h[i] + mu1[i] - mu0[i] :
                            mu1[i] - (y_j[i] - mu0[i]) / (1 - p_h[i]) - mu0[i] for i in 1:T_h]

            ate_dr[j] = mean(psi)
            ate_se_h[j] = std(psi) / sqrt(T_h)
        end

        ate[h + 1, :] = ate_dr
        ate_se[h + 1, :] = ate_se_h

        # Store regression coefficients (IPW version)
        k = 2 + n * lags + n_cov
        X_full = hcat(ones(T, T_h), T.(D_h), X_cov[:, 2:end])
        W = Diagonal(ipw_weights[t_start:t_end])
        B[h + 1] = robust_inv(X_full' * W * X_full) * (X_full' * W * Y_h)
        residuals_store[h + 1] = Y_h - X_full * B[h + 1]

        vcov_vec[h + 1] = zeros(T, k * n_response, k * n_response)
        for (j, _) in enumerate(response_vars)
            vcov_vec[h + 1][(j-1)*k + 2, (j-1)*k + 2] = ate_se_h[j]^2
        end
    end

    PropensityLPModel{T}(Matrix{T}(Y), treatment, response_vars, Matrix{T}(covariates),
                         horizon, propensity, ipw_weights, B, residuals_store, vcov_vec,
                         ate, ate_se, config, T_eff, cov_estimator)
end

"""
    propensity_irf(model::PropensityLPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract treatment effect (ATE) IRF from PropensityLPModel.
"""
function propensity_irf(model::PropensityLPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = model.ate .- z .* model.ate_se
    ci_upper = model.ate .+ z .* model.ate_se

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(model.ate, ci_lower, ci_upper, model.ate_se, model.horizon,
                         response_names, "Treatment Effect (ATE)", cov_type_sym, T(conf_level))
end

"""
    propensity_diagnostics(model::PropensityLPModel{T}) -> NamedTuple

Propensity score diagnostics (overlap, balance).
"""
function propensity_diagnostics(model::PropensityLPModel{T}) where {T<:AbstractFloat}
    p = model.propensity_scores
    w = model.ipw_weights
    treated_idx = findall(model.treatment)
    control_idx = findall(.!model.treatment)

    ps_summary = (treated=(min=minimum(p[treated_idx]), max=maximum(p[treated_idx]),
                           mean=mean(p[treated_idx]), std=std(p[treated_idx])),
                  control=(min=minimum(p[control_idx]), max=maximum(p[control_idx]),
                           mean=mean(p[control_idx]), std=std(p[control_idx])))

    overlap_min = max(minimum(p[treated_idx]), minimum(p[control_idx]))
    overlap_max = min(maximum(p[treated_idx]), maximum(p[control_idx]))
    overlap = (common_support=(overlap_min, overlap_max),
               treated_in_support=mean((p[treated_idx] .>= overlap_min) .& (p[treated_idx] .<= overlap_max)),
               control_in_support=mean((p[control_idx] .>= overlap_min) .& (p[control_idx] .<= overlap_max)))

    X = model.covariates
    n_cov = size(X, 2)
    balance_raw = Vector{T}(undef, n_cov)
    balance_weighted = Vector{T}(undef, n_cov)

    for j in 1:n_cov
        x = X[:, j]
        pooled_sd = sqrt((var(x[treated_idx]) + var(x[control_idx])) / 2)
        balance_raw[j] = (mean(x[treated_idx]) - mean(x[control_idx])) / pooled_sd

        w_t, w_c = w[treated_idx], w[control_idx]
        mean_t_w = sum(w_t .* x[treated_idx]) / sum(w_t)
        mean_c_w = sum(w_c .* x[control_idx]) / sum(w_c)
        balance_weighted[j] = (mean_t_w - mean_c_w) / pooled_sd
    end

    (propensity_summary=ps_summary, overlap=overlap,
     balance=(raw_smd=balance_raw, weighted_smd=balance_weighted,
              max_raw=maximum(abs.(balance_raw)), max_weighted=maximum(abs.(balance_weighted))))
end
