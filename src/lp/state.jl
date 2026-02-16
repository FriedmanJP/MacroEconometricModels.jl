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
State-dependent LP (Auerbach & Gorodnichenko 2013).
"""

using LinearAlgebra, Statistics, Distributions

"""
    logistic_transition(z::AbstractVector{T}, gamma::T, c::T) -> Vector{T}

Logistic transition function: F(z) = exp(-γ(z - c)) / (1 + exp(-γ(z - c)))
"""
logistic_transition(z::AbstractVector{T}, gamma::T, c::T) where {T<:AbstractFloat} =
    @. exp(-gamma * (z - c)) / (1 + exp(-gamma * (z - c)))

logistic_transition(z::T, gamma::T, c::T) where {T<:AbstractFloat} =
    exp(-gamma * (z - c)) / (1 + exp(-gamma * (z - c)))

"""
    exponential_transition(z::AbstractVector{T}, gamma::T, c::T) -> Vector{T}

Exponential (symmetric) transition: F(z) = 1 - exp(-γ(z - c)²)
"""
exponential_transition(z::AbstractVector{T}, gamma::T, c::T) where {T<:AbstractFloat} =
    @. 1 - exp(-gamma * (z - c)^2)

"""
    indicator_transition(z::AbstractVector{T}, c::T) -> Vector{T}

Sharp indicator transition: F(z) = 1 if z ≥ c, else 0
"""
indicator_transition(z::AbstractVector{T}, c::T) where {T<:AbstractFloat} = T.(z .>= c)

"""
    estimate_transition_params(state_var::AbstractVector{T}, Y::AbstractMatrix{T},
                               shock_var::Int; method::Symbol=:nlls, ...) -> NamedTuple

Estimate smooth transition parameters (γ, c).
"""
function estimate_transition_params(state_var::AbstractVector{T}, Y::AbstractMatrix{T},
                                    shock_var::Int; method::Symbol=:nlls,
                                    gamma_init::T=T(1.5),
                                    c_init::Union{T,Symbol}=:median) where {T<:AbstractFloat}
    @assert length(state_var) == size(Y, 1)

    c0 = c_init isa Symbol ? (c_init == :median ? median(state_var) :
                               c_init == :mean ? mean(state_var) : zero(T)) : c_init

    compute_ssr(gamma, c) = begin
        F = logistic_transition(state_var, gamma, c)
        T_obs, n = size(Y)
        ssr = zero(T)
        for t in 2:T_obs
            shock_t = Y[t-1, shock_var]
            for var in 1:n
                pred = (1 - F[t-1]) * shock_t + F[t-1] * shock_t
                ssr += (Y[t, var] - pred)^2
            end
        end
        ssr
    end

    if method == :grid_search
        gamma_grid = T.(range(0.5, 5.0, length=20))
        c_grid = T.(quantile(state_var, range(0.1, 0.9, length=20)))

        best_gamma, best_c, best_ssr = gamma_init, c0, T(Inf)
        for gamma in gamma_grid, c in c_grid
            ssr = compute_ssr(gamma, c)
            if ssr < best_ssr
                best_ssr, best_gamma, best_c = ssr, gamma, c
            end
        end
        (gamma=best_gamma, c=best_c, F_values=logistic_transition(state_var, best_gamma, best_c),
         convergence_info=(method=:grid_search, ssr=best_ssr))
    else  # :nlls
        gamma_curr, c_curr = gamma_init, c0
        for iter in 1:50
            gamma_prev, c_prev = gamma_curr, c_curr

            # Optimize γ
            for gamma in T.(range(max(0.1, gamma_curr - 1), gamma_curr + 1, length=10))
                compute_ssr(gamma, c_curr) < compute_ssr(gamma_curr, c_curr) && (gamma_curr = gamma)
            end

            # Optimize c
            for c in T.(quantile(state_var, range(0.1, 0.9, length=15)))
                compute_ssr(gamma_curr, c) < compute_ssr(gamma_curr, c_curr) && (c_curr = c)
            end

            abs(gamma_curr - gamma_prev) < 1e-4 && abs(c_curr - c_prev) < 1e-4 &&
                return (gamma=gamma_curr, c=c_curr, F_values=logistic_transition(state_var, gamma_curr, c_curr),
                        convergence_info=(method=:nlls, iterations=iter, converged=true))
        end
        (gamma=gamma_curr, c=c_curr, F_values=logistic_transition(state_var, gamma_curr, c_curr),
         convergence_info=(method=:nlls, iterations=50, converged=false))
    end
end

"""
    estimate_state_lp(Y::AbstractMatrix{T}, shock_var::Int, state_var::AbstractVector{T},
                      horizon::Int; gamma::Union{T,Symbol}=:estimate, ...) -> StateLPModel{T}

Estimate state-dependent LP (Auerbach & Gorodnichenko 2013).
"""
function estimate_state_lp(Y::AbstractMatrix{T}, shock_var::Int, state_var::AbstractVector{T},
                           horizon::Int; gamma::Union{T,Symbol}=:estimate,
                           threshold::Union{T,Symbol}=:estimate,
                           lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                           cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    @assert length(state_var) == T_obs

    # Estimate or set transition parameters
    gamma_val, c_val, F_values = if gamma == :estimate || threshold == :estimate
        result = estimate_transition_params(state_var, Y, shock_var;
                                            c_init=threshold isa Symbol ? :median : threshold)
        (gamma == :estimate ? result.gamma : gamma,
         threshold == :estimate ? result.c : (threshold isa Symbol ? result.c : threshold),
         logistic_transition(state_var, gamma == :estimate ? result.gamma : gamma,
                             threshold == :estimate ? result.c : (threshold isa Symbol ? result.c : threshold)))
    else
        c_fixed = threshold isa Symbol ? (threshold == :median ? median(state_var) :
                                           threshold == :mean ? mean(state_var) : zero(T)) : threshold
        (gamma, c_fixed, logistic_transition(state_var, gamma, c_fixed))
    end

    state_transition = StateTransition(state_var, gamma_val, c_val, :logistic)
    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)

    B_expansion = Vector{Matrix{T}}(undef, horizon + 1)
    B_recession = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_expansion = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_recession = Vector{Matrix{T}}(undef, horizon + 1)
    vcov_diff = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    k_per_regime = 2 + n * lags

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

        # Build state-dependent regressor matrix
        k_total = 2 * k_per_regime
        X_h = Matrix{T}(undef, T_h, k_total)

        @inbounds for (i, t) in enumerate(t_start:t_end)
            F_t = F_values[t]

            # Expansion regime (F_t)
            X_h[i, 1] = F_t
            X_h[i, 2] = F_t * Y[t, shock_var]
            col = 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = F_t * Y[t - lag, var]
                col += 1
            end

            # Recession regime (1 - F_t)
            X_h[i, k_per_regime + 1] = 1 - F_t
            X_h[i, k_per_regime + 2] = (1 - F_t) * Y[t, shock_var]
            col = k_per_regime + 3
            for lag in 1:lags, var in 1:n
                X_h[i, col] = (1 - F_t) * Y[t - lag, var]
                col += 1
            end
        end

        XtX_inv = robust_inv(X_h' * X_h)
        B_h = XtX_inv * (X_h' * Y_h)
        U_h = Y_h - X_h * B_h

        B_expansion[h + 1] = B_h[1:k_per_regime, :]
        B_recession[h + 1] = B_h[(k_per_regime + 1):end, :]
        residuals_store[h + 1] = U_h

        V_h = compute_block_robust_vcov(X_h, U_h, cov_estimator)

        # Extract regime-specific covariances
        V_exp = zeros(T, k_per_regime * n_response, k_per_regime * n_response)
        V_rec = zeros(T, k_per_regime * n_response, k_per_regime * n_response)
        V_df = zeros(T, k_per_regime * n_response, k_per_regime * n_response)

        for eq in 1:n_response
            eq_start = (eq - 1) * k_total
            exp_idx_full = (eq_start + 1):(eq_start + k_per_regime)
            rec_idx_full = (eq_start + k_per_regime + 1):(eq_start + k_total)
            local_idx = ((eq-1)*k_per_regime + 1):(eq*k_per_regime)

            V_exp[local_idx, local_idx] .= V_h[exp_idx_full, exp_idx_full]
            V_rec[local_idx, local_idx] .= V_h[rec_idx_full, rec_idx_full]
            V_df[local_idx, local_idx] .= V_h[exp_idx_full, exp_idx_full] +
                                           V_h[rec_idx_full, rec_idx_full] -
                                           2 .* V_h[exp_idx_full, rec_idx_full]
        end

        vcov_expansion[h + 1] = V_exp
        vcov_recession[h + 1] = V_rec
        vcov_diff[h + 1] = V_df
    end

    StateLPModel{T}(Matrix{T}(Y), shock_var, response_vars, horizon, lags, state_transition,
                    B_expansion, B_recession, residuals_store, vcov_expansion, vcov_recession,
                    vcov_diff, T_eff, cov_estimator)
end

estimate_state_lp(Y::AbstractMatrix, shock_var::Int, state_var::AbstractVector,
                  horizon::Int; kwargs...) =
    estimate_state_lp(Float64.(Y), shock_var, Float64.(state_var), horizon; kwargs...)

"""
    state_irf(model::StateLPModel{T}; regime::Symbol=:both, conf_level::Real=0.95) -> NamedTuple

Extract state-dependent IRFs.
"""
function state_irf(model::StateLPModel{T}; regime::Symbol=:both,
                   conf_level::Real=0.95) where {T<:AbstractFloat}
    H = model.horizon
    n_response = length(model.response_vars)
    shock_coef_idx = 2
    k_per_regime = size(model.B_expansion[1], 1)

    irf_exp = Matrix{T}(undef, H + 1, n_response)
    irf_rec = Matrix{T}(undef, H + 1, n_response)
    se_exp = Matrix{T}(undef, H + 1, n_response)
    se_rec = Matrix{T}(undef, H + 1, n_response)
    se_diff = Matrix{T}(undef, H + 1, n_response)

    for h in 0:H
        for (j, _) in enumerate(model.response_vars)
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            irf_exp[h + 1, j] = model.B_expansion[h + 1][shock_coef_idx, j]
            irf_rec[h + 1, j] = model.B_recession[h + 1][shock_coef_idx, j]
            se_exp[h + 1, j] = sqrt(model.vcov_expansion[h + 1][var_idx, var_idx])
            se_rec[h + 1, j] = sqrt(model.vcov_recession[h + 1][var_idx, var_idx])
            se_diff[h + 1, j] = sqrt(model.vcov_diff[h + 1][var_idx, var_idx])
        end
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    irf_diff = irf_exp .- irf_rec

    regime == :expansion && return (values=irf_exp, ci_lower=irf_exp .- z .* se_exp,
                                     ci_upper=irf_exp .+ z .* se_exp, se=se_exp,
                                     regime=:expansion, conf_level=T(conf_level))
    regime == :recession && return (values=irf_rec, ci_lower=irf_rec .- z .* se_rec,
                                     ci_upper=irf_rec .+ z .* se_rec, se=se_rec,
                                     regime=:recession, conf_level=T(conf_level))
    regime == :difference && return (values=irf_diff, ci_lower=irf_diff .- z .* se_diff,
                                      ci_upper=irf_diff .+ z .* se_diff, se=se_diff,
                                      regime=:difference, conf_level=T(conf_level))

    (expansion=(values=irf_exp, ci_lower=irf_exp .- z .* se_exp, ci_upper=irf_exp .+ z .* se_exp, se=se_exp),
     recession=(values=irf_rec, ci_lower=irf_rec .- z .* se_rec, ci_upper=irf_rec .+ z .* se_rec, se=se_rec),
     difference=(values=irf_diff, ci_lower=irf_diff .- z .* se_diff, ci_upper=irf_diff .+ z .* se_diff, se=se_diff),
     gamma=model.state.gamma, threshold=model.state.threshold, conf_level=T(conf_level))
end

"""
    test_regime_difference(model::StateLPModel{T}; h::Union{Int,Nothing}=nothing) -> NamedTuple

Test whether IRFs differ across regimes.
"""
function test_regime_difference(model::StateLPModel{T}; h::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}
    H = model.horizon
    n_response = length(model.response_vars)
    shock_coef_idx = 2
    k_per_regime = size(model.B_expansion[1], 1)

    if !isnothing(h)
        @assert 0 <= h <= H
        t_stats = Vector{T}(undef, n_response)
        p_values = Vector{T}(undef, n_response)

        for (j, _) in enumerate(model.response_vars)
            diff = model.B_expansion[h + 1][shock_coef_idx, j] - model.B_recession[h + 1][shock_coef_idx, j]
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            se = sqrt(model.vcov_diff[h + 1][var_idx, var_idx])
            t_stats[j] = diff / se
            p_values[j] = 2 * (1 - cdf(Normal(), abs(t_stats[j])))
        end
        return (horizon=h, t_stats=t_stats, p_values=p_values)
    end

    t_stats = Matrix{T}(undef, H + 1, n_response)
    p_values = Matrix{T}(undef, H + 1, n_response)

    for h_idx in 0:H
        for (j, _) in enumerate(model.response_vars)
            diff = model.B_expansion[h_idx + 1][shock_coef_idx, j] - model.B_recession[h_idx + 1][shock_coef_idx, j]
            var_idx = (j - 1) * k_per_regime + shock_coef_idx
            se = sqrt(model.vcov_diff[h_idx + 1][var_idx, var_idx])
            t_stats[h_idx + 1, j] = diff / se
            p_values[h_idx + 1, j] = 2 * (1 - cdf(Normal(), abs(t_stats[h_idx + 1, j])))
        end
    end

    avg_t = mean(abs.(t_stats))
    (t_stats=t_stats, p_values=p_values, joint_test=(avg_t_stat=avg_t, p_value=2 * (1 - cdf(Normal(), avg_t))))
end
