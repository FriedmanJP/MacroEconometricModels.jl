# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
LP-IV: Instrumental Variables (Stock & Watson 2018).
"""

using LinearAlgebra, Statistics, Distributions

"""
    first_stage_regression(endog::AbstractVector{T}, instruments::AbstractMatrix{T},
                           controls::AbstractMatrix{T}) -> NamedTuple

First-stage regression for 2SLS with F-statistic for instrument relevance.
"""
function first_stage_regression(endog::AbstractVector{T}, instruments::AbstractMatrix{T},
                                 controls::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = length(endog)
    n_inst = size(instruments, 2)

    X = hcat(ones(T, n), instruments, controls)
    k = size(X, 2)

    XtX_inv = robust_inv(X' * X)
    beta = XtX_inv * (X' * endog)
    fitted = X * beta
    residuals = endog - fitted
    sigma2 = sum(residuals.^2) / (n - k)

    # F-statistic for instruments
    inst_coef = beta[2:(n_inst + 1)]
    V_inst = XtX_inv[2:(n_inst + 1), 2:(n_inst + 1)]
    F_stat = n_inst == 1 ? inst_coef[1]^2 / (sigma2 * V_inst[1, 1]) :
                           inst_coef' * inv(V_inst) * inst_coef / (n_inst * sigma2)

    (fitted=fitted, residuals=residuals, F_stat=T(F_stat), coefficients=beta,
     vcov=sigma2 * XtX_inv, sigma2=sigma2, n_instruments=n_inst)
end

"""
    tsls_regression(Y::AbstractMatrix{T}, endog::AbstractVector{T},
                    endog_fitted::AbstractVector{T}, controls::AbstractMatrix{T};
                    cov_estimator::AbstractCovarianceEstimator=NeweyWestEstimator()) -> NamedTuple

Second-stage regression using fitted values from first stage.
"""
function tsls_regression(Y::AbstractMatrix{T}, endog::AbstractVector{T},
                         endog_fitted::AbstractVector{T}, controls::AbstractMatrix{T};
                         cov_estimator::AbstractCovarianceEstimator=NeweyWestEstimator{T}()) where {T<:AbstractFloat}
    n, n_resp = size(Y)

    X_2s = hcat(ones(T, n), endog_fitted, controls)
    k = size(X_2s, 2)

    XtX_inv = robust_inv(X_2s' * X_2s)
    beta = XtX_inv * (X_2s' * Y)

    # Residuals using actual endog for correct SE
    X_actual = hcat(ones(T, n), endog, controls)
    residuals = Y - X_actual * beta

    V = compute_block_robust_vcov(X_2s, residuals, cov_estimator)

    (coefficients=beta, residuals=residuals, vcov=V, n_obs=n, n_regressors=k)
end

"""
    estimate_lp_iv(Y::AbstractMatrix{T}, shock_var::Int, instruments::AbstractMatrix{T},
                   horizon::Int; lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                   cov_type::Symbol=:newey_west, bandwidth::Int=0) -> LPIVModel{T}

Estimate LP with Instrumental Variables (Stock & Watson 2018) using 2SLS.
"""
function estimate_lp_iv(Y::AbstractMatrix{T}, shock_var::Int, instruments::AbstractMatrix{T},
                        horizon::Int; lags::Int=4,
                        response_vars::Vector{Int}=collect(1:size(Y, 2)),
                        cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_inst = size(instruments, 2)

    @assert size(instruments, 1) == T_obs "instruments must have same T as Y"
    @assert n_inst >= 1 "need at least one instrument"
    @assert 1 <= shock_var <= n "shock_var must be in 1:$n"

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)
    n_response = length(response_vars)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals_store = Vector{Matrix{T}}(undef, horizon + 1)
    vcov = Vector{Matrix{T}}(undef, horizon + 1)
    first_stage_F = Vector{T}(undef, horizon + 1)
    first_stage_coef = Vector{Vector{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    for h in 0:horizon
        t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
        T_h = t_end - t_start + 1
        T_eff[h + 1] = T_h

        Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

        # Endogenous and instruments at t
        endog = [Y[t, shock_var] for t in t_start:t_end]
        Z = instruments[t_start:t_end, :]

        # Controls: lagged Y
        n_ctrl = n * lags
        controls = Matrix{T}(undef, T_h, n_ctrl)
        col = 1
        @inbounds for (i, t) in enumerate(t_start:t_end)
            col_local = 1
            for lag in 1:lags, var in 1:n
                controls[i, col_local] = Y[t - lag, var]
                col_local += 1
            end
        end

        # First and second stage
        fs = first_stage_regression(endog, Z, controls)
        first_stage_F[h + 1] = fs.F_stat
        first_stage_coef[h + 1] = fs.coefficients[2:(n_inst + 1)]

        ss = tsls_regression(Y_h, endog, fs.fitted, controls; cov_estimator=cov_estimator)
        B[h + 1] = ss.coefficients
        residuals_store[h + 1] = ss.residuals
        vcov[h + 1] = ss.vcov
    end

    LPIVModel{T}(Matrix{T}(Y), shock_var, response_vars, Matrix{T}(instruments),
                 horizon, lags, B, residuals_store, vcov, first_stage_F,
                 first_stage_coef, T_eff, cov_estimator)
end

estimate_lp_iv(Y::AbstractMatrix, shock_var::Int, instruments::AbstractMatrix,
               horizon::Int; kwargs...) =
    estimate_lp_iv(Float64.(Y), shock_var, Float64.(instruments), horizon; kwargs...)

"""
    weak_instrument_test(model::LPIVModel{T}; threshold::T=T(10.0)) -> NamedTuple

Test for weak instruments using Stock-Yogo rule of thumb (F > 10).
"""
function weak_instrument_test(model::LPIVModel{T}; threshold::T=T(10.0)) where {T<:AbstractFloat}
    F_stats = model.first_stage_F
    weak_horizons = findall(F_stats .< threshold)
    (F_stats=F_stats, weak_horizons=weak_horizons, min_F=minimum(F_stats),
     passes_threshold=isempty(weak_horizons), threshold=threshold)
end

weak_instrument_test(F_stats::Vector{T}; threshold::T=T(10.0)) where {T<:AbstractFloat} =
    (F_stats=F_stats, weak_horizons=findall(F_stats .< threshold),
     min_F=minimum(F_stats), passes_threshold=all(F_stats .>= threshold), threshold=threshold)

"""
    lp_iv_irf(model::LPIVModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract IRF from LP-IV model.
"""
function lp_iv_irf(model::LPIVModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    irf_data = extract_shock_irf(model.B, model.vcov, model.response_vars, 2;
                                  conf_level=conf_level)

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    shock_name = "Instrumented Shock $(model.shock_var)"
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(irf_data.values, irf_data.ci_lower, irf_data.ci_upper,
                         irf_data.se, model.horizon, response_names, shock_name,
                         cov_type_sym, T(conf_level))
end

"""
    sargan_test(model::LPIVModel{T}, h::Int) -> NamedTuple

Sargan-Hansen J-test for overidentification at horizon h.
"""
function sargan_test(model::LPIVModel{T}, h::Int) where {T<:AbstractFloat}
    n_inst = n_instruments(model)
    n_inst <= 1 && return (J_stat=T(NaN), p_value=T(NaN), df=0, valid=false)

    U_h = model.residuals[h + 1]
    n_resp = size(U_h, 2)
    T_h = model.T_eff[h + 1]

    t_start, t_end = compute_horizon_bounds(size(model.Y, 1), h, model.lags)
    Z = model.instruments[t_start:t_end, :]

    J_stats = [begin
        u = @view U_h[:, eq]
        sigma2 = sum(u.^2) / T_h
        Zu = Z' * u
        ZtZ_inv = robust_inv(Z' * Z)
        T_h * (Zu' * ZtZ_inv * Zu) / sigma2
    end for eq in 1:n_resp]

    J_avg = mean(J_stats)
    df = n_inst - 1
    (J_stat=J_avg, p_value=1 - cdf(Chisq(df), J_avg), df=df, valid=true)
end
