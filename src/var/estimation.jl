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
VAR estimation via OLS with StatsAPI interface.
"""

using LinearAlgebra, Statistics, DataFrames, Distributions

# =============================================================================
# Core Estimation
# =============================================================================

"""
    estimate_var(Y::AbstractMatrix{T}, p::Int; check_stability::Bool=true) -> VARModel{T}

Estimate VAR(p) via OLS: Yₜ = c + A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + uₜ.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags
- `check_stability`: If true (default), warns if estimated VAR is non-stationary

# Returns
`VARModel` with estimated coefficients, residuals, covariance matrix, and information criteria.
"""
function estimate_var(Y::AbstractMatrix{T}, p::Int; check_stability::Bool=true) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    validate_var_inputs(T_obs, n, p)

    Y_eff, X = construct_var_matrices(Y, p)
    T_eff, k = size(Y_eff, 1), size(X, 2)

    # OLS: B = (X'X)⁻¹X'Y
    B = robust_inv(X'X) * (X' * Y_eff)
    U = Y_eff - X * B

    # Covariance with dof adjustment
    dof_adj = max(T_eff - k, T_eff)
    T_eff - k <= 0 && @warn "Non-positive dof adjustment, using T_eff"
    Sigma = (U'U) / dof_adj

    # Information criteria (ML estimate)
    Sigma_ml = (U'U) / T_eff
    log_det = logdet_safe(Sigma_ml)
    aic = log_det + 2k / T_eff
    bic = log_det + k * log(T_eff) / T_eff
    hqic = log_det + 2k * log(log(T_eff)) / T_eff

    model = VARModel(Y, p, B, U, Sigma, aic, bic, hqic)

    # Check stationarity via companion matrix eigenvalues
    if check_stability
        F = companion_matrix(B, n, p)
        max_modulus = maximum(abs.(eigvals(F)))
        if max_modulus >= one(T)
            @warn "Estimated VAR is non-stationary (max eigenvalue modulus = $(round(max_modulus, digits=4))). " *
                  "Consider differencing the data or using a VECM specification."
        end
    end

    model
end

@float_fallback estimate_var Y

"""Estimate VAR from DataFrame. Use `vars` to select columns."""
function estimate_var(df::DataFrame, p::Int; vars::Vector{Symbol}=Symbol[], check_stability::Bool=true)
    data = isempty(vars) ? Matrix(df) : Matrix(df[:, vars])
    estimate_var(Float64.(data), p; check_stability=check_stability)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.fit(::Type{VARModel}, Y::AbstractMatrix, p::Int) = estimate_var(Y, p)
StatsAPI.fit(::Type{VARModel}, df::DataFrame, p::Int; vars::Vector{Symbol}=Symbol[]) =
    estimate_var(df, p; vars=vars)

StatsAPI.coef(model::VARModel) = model.B
StatsAPI.residuals(model::VARModel) = model.U
StatsAPI.dof(model::VARModel) = length(model.B)
StatsAPI.dof_residual(model::VARModel) = size(model.U, 1) - size(model.B, 1)
StatsAPI.nobs(model::VARModel) = size(model.Y, 1)
StatsAPI.aic(model::VARModel) = model.aic
StatsAPI.bic(model::VARModel) = model.bic
StatsAPI.islinear(::VARModel) = true

"""Covariance of vectorized coefficients: Σ ⊗ (X'X)⁻¹."""
function StatsAPI.vcov(model::VARModel{T}) where {T}
    _, X = construct_var_matrices(model.Y, model.p)
    kron(model.Sigma, robust_inv(X'X))
end

"""In-sample fitted values."""
StatsAPI.predict(model::VARModel) = @view(model.Y[(model.p+1):end, :]) - model.U

"""Out-of-sample forecasts for `steps` periods."""
function StatsAPI.predict(model::VARModel{T}, steps::Int) where {T}
    steps < 1 && throw(ArgumentError("steps must be positive"))

    n, p, B = nvars(model), model.p, model.B
    forecasts = Matrix{T}(undef, steps, n)
    intercept = @view B[1, :]
    A = extract_ar_coefficients(B, n, p)
    history = copy(model.Y[(end-p+1):end, :])

    @inbounds for h in 1:steps
        y_hat = copy(intercept)
        for lag in 1:p
            y_hat .+= A[lag] * @view(history[end-lag+1, :])
        end
        forecasts[h, :] = y_hat
        history = vcat(@view(history[2:end, :]), y_hat')
    end
    forecasts
end

"""R² for each equation."""
function StatsAPI.r2(model::VARModel{T}) where {T}
    Y_eff = @view model.Y[(model.p+1):end, :]
    [1 - var(@view(model.U[:, i])) / var(@view(Y_eff[:, i])) for i in 1:nvars(model)]
end

"""Gaussian log-likelihood."""
function StatsAPI.loglikelihood(model::VARModel{T}) where {T}
    n, T_eff = nvars(model), effective_nobs(model)
    -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * logdet_safe(model.Sigma) - T(T_eff * n / 2)
end

StatsAPI.stderror(model::VARModel) = sqrt.(diag(vcov(model)))

"""Confidence intervals at given level (default 95%)."""
function StatsAPI.confint(model::VARModel{T}; level::Real=0.95) where {T}
    B_vec, se = vec(model.B), stderror(model)
    crit = T(quantile(TDist(dof_residual(model)), 1 - (1 - level) / 2))
    hcat(B_vec .- crit .* se, B_vec .+ crit .* se)
end

# =============================================================================
# Model Selection
# =============================================================================

"""Select optimal lag order via information criterion (:aic, :bic, :hqic)."""
function select_lag_order(Y::AbstractMatrix{T}, max_p::Int; criterion::Symbol=:bic) where {T<:AbstractFloat}
    max_p < 1 && throw(ArgumentError("max_p must be positive"))
    size(Y, 1) <= max_p + 2 && throw(ArgumentError("Not enough observations"))

    ic = map(1:max_p) do p
        m = estimate_var(Y, p)
        criterion == :aic ? m.aic : criterion == :bic ? m.bic :
        criterion == :hqic ? m.hqic : throw(ArgumentError("Unknown criterion: $criterion"))
    end
    argmin(ic)
end
