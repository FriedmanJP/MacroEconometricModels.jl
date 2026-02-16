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
Smooth LP: B-spline parameterization (Barnichon & Brownlees 2019).
"""

using LinearAlgebra, Statistics, Distributions

"""
    bspline_basis_value(x::T, i::Int, degree::Int, knots::Vector{T}) where T

Evaluate B-spline basis function using Cox-de Boor recursion.
"""
function bspline_basis_value(x::T, i::Int, degree::Int, knots::Vector{T}) where {T<:AbstractFloat}
    if degree == 0
        return (i == length(knots) - 1 ? knots[i] <= x <= knots[i + 1] :
                                          knots[i] <= x < knots[i + 1]) ? one(T) : zero(T)
    end

    n = length(knots)
    denom1 = knots[i + degree] - knots[i]
    w1 = abs(denom1) < eps(T) ? zero(T) : (x - knots[i]) / denom1

    denom2 = knots[i + degree + 1] - knots[i + 1]
    w2 = abs(denom2) < eps(T) ? zero(T) : (knots[i + degree + 1] - x) / denom2

    left = i + degree <= n ? w1 * bspline_basis_value(x, i, degree - 1, knots) : zero(T)
    right = i + 1 + degree <= n ? w2 * bspline_basis_value(x, i + 1, degree - 1, knots) : zero(T)

    left + right
end

"""
    bspline_basis(horizons::AbstractVector{Int}, degree::Int, n_interior_knots::Int;
                  T::Type{<:AbstractFloat}=Float64) -> BSplineBasis{T}

Construct B-spline basis matrix for given horizons.
"""
function bspline_basis(horizons::AbstractVector{Int}, degree::Int, n_interior_knots::Int;
                       T::Type{<:AbstractFloat}=Float64)
    @assert degree >= 0 && n_interior_knots >= 0

    h_min, h_max = T(minimum(horizons)), T(maximum(horizons))
    n_b = n_interior_knots + degree + 1

    # Construct clamped knot vector
    knots = if n_interior_knots > 0
        interior = range(h_min, h_max, length=n_interior_knots + 2)[2:end-1]
        vcat(fill(h_min, degree + 1), collect(T, interior), fill(h_max, degree + 1))
    else
        vcat(fill(h_min, degree + 1), fill(h_max, degree + 1))
    end

    basis_matrix = [bspline_basis_value(T(h), j, degree, knots)
                    for h in horizons, j in 1:n_b]

    BSplineBasis{T}(degree, n_interior_knots, knots, basis_matrix, collect(horizons))
end

"""
    roughness_penalty_matrix(basis::BSplineBasis{T}) -> Matrix{T}

Compute roughness penalty matrix R for B-splines (second derivative penalty).
"""
function roughness_penalty_matrix(basis::BSplineBasis{T}) where {T<:AbstractFloat}
    n_b = n_basis(basis)
    h_min, h_max = T(minimum(basis.horizons)), T(maximum(basis.horizons))
    n_grid = max(100, 10n_b)
    grid = range(h_min, h_max, length=n_grid)
    dx = (h_max - h_min) / (n_grid - 1)

    B = [bspline_basis_value(T(x), j, basis.degree, basis.knots)
         for x in grid, j in 1:n_b]

    # Second derivative via finite differences
    B_dd = [(B[i+1, j] - 2B[i, j] + B[i-1, j]) / dx^2
            for i in 2:(n_grid-1), j in 1:n_b]

    R = dx * (B_dd' * B_dd)
    (R + R') / 2
end

"""
    estimate_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                       degree::Int=3, n_knots::Int=4, lambda::T=T(0.0),
                       lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                       cov_type::Symbol=:newey_west, bandwidth::Int=0) -> SmoothLPModel{T}

Estimate Smooth LP with B-spline parameterization (Barnichon & Brownlees 2019).
"""
function estimate_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                            degree::Int=3, n_knots::Int=4, lambda::T=T(0.0),
                            lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                            cov_type::Symbol=:newey_west, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_response = length(response_vars)
    horizons = collect(0:horizon)

    basis = bspline_basis(horizons, degree, n_knots; T=T)
    n_b = n_basis(basis)
    B_mat = basis.basis_matrix
    R = roughness_penalty_matrix(basis)

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    # Step 1: Standard LP estimates
    lp_model = estimate_lp(Y, shock_var, horizon; lags=lags, response_vars=response_vars,
                           cov_type=cov_type, bandwidth=bandwidth)
    lp_result = lp_irf(lp_model)

    beta_hat = lp_result.values
    beta_se = lp_result.se

    # Step 2: Fit spline with weighted least squares + penalty
    theta = Matrix{T}(undef, n_b, n_response)
    vcov_theta_blocks = Vector{Matrix{T}}(undef, n_response)

    for j in 1:n_response
        w = 1 ./ (beta_se[:, j].^2 .+ eps(T))
        W = Diagonal(w)
        reg_mat = B_mat' * W * B_mat + lambda * R
        theta[:, j] = reg_mat \ (B_mat' * W * beta_hat[:, j])

        reg_inv = inv(reg_mat)
        Sigma_beta = Diagonal(beta_se[:, j].^2)
        vcov_theta_blocks[j] = reg_inv * (B_mat' * W * Sigma_beta * W * B_mat) * reg_inv
    end

    vcov_theta = zeros(T, n_b * n_response, n_b * n_response)
    for j in 1:n_response
        idx = ((j-1)*n_b + 1):(j*n_b)
        vcov_theta[idx, idx] .= vcov_theta_blocks[j]
    end

    irf_values = B_mat * theta
    irf_se = [sqrt(B_mat[h_idx, :]' * vcov_theta_blocks[j] * B_mat[h_idx, :])
              for h_idx in 1:length(horizons), j in 1:n_response]

    residuals = vcat([lp_model.residuals[h+1] for h in horizons]...)
    T_total = sum(T_obs - h - lags for h in horizons)

    SmoothLPModel{T}(Matrix{T}(Y), shock_var, response_vars, horizon, lags, basis,
                     theta, vcov_theta, lambda, irf_values, Matrix{T}(irf_se),
                     residuals, T_total, cov_estimator)
end

estimate_smooth_lp(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) =
    estimate_smooth_lp(Float64.(Y), shock_var, horizon; kwargs...)

"""
    cross_validate_lambda(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                          lambda_grid::Vector{T}=T.(10.0 .^ (-4:0.5:2)),
                          k_folds::Int=5, kwargs...) -> T

Select optimal λ via k-fold cross-validation.
"""
function cross_validate_lambda(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                               lambda_grid::Vector{T}=T.(10.0 .^ (-4:0.5:2)),
                               k_folds::Int=5, kwargs...) where {T<:AbstractFloat}
    T_obs = size(Y, 1)
    lags = get(kwargs, :lags, 4)
    t_start, t_end = compute_horizon_bounds(T_obs, horizon, lags)
    n_usable = t_end - t_start + 1
    fold_size = n_usable ÷ k_folds

    cv_errors = zeros(T, length(lambda_grid))

    for (i, lam) in enumerate(lambda_grid)
        fold_mse = zeros(T, k_folds)
        for k in 1:k_folds
            start_k = t_start + (k - 1) * fold_size
            end_k = k == k_folds ? t_end : start_k + fold_size - 1
            test_idx = collect(start_k:end_k)

            train_mask = trues(T_obs)
            for t in test_idx, h in 0:horizon
                t + h <= T_obs && (train_mask[t + h] = false)
            end
            train_idx = findall(train_mask)

            length(train_idx) < lags + horizon + 10 && continue

            try
                model = estimate_smooth_lp(Y[train_idx, :], shock_var, horizon; lambda=lam, kwargs...)
                fold_mse[k] = mean((model.irf_values .- model.spline_basis.basis_matrix * model.theta).^2)
            catch
                fold_mse[k] = T(Inf)
            end
        end
        cv_errors[i] = mean(fold_mse[fold_mse .< Inf])
    end

    lambda_grid[argmin(cv_errors)]
end

"""
    smooth_lp_irf(model::SmoothLPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract smoothed IRF from SmoothLPModel.
"""
function smooth_lp_irf(model::SmoothLPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = model.irf_values .- z .* model.irf_se
    ci_upper = model.irf_values .+ z .* model.irf_se

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(model.irf_values, ci_lower, ci_upper, model.irf_se, model.horizon,
                         response_names, "Shock $(model.shock_var)", cov_type_sym, T(conf_level))
end

"""
    compare_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                      lambda::T=T(1.0), kwargs...) -> NamedTuple

Compare standard LP and smooth LP.
"""
function compare_smooth_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                           lambda::T=T(1.0), kwargs...) where {T<:AbstractFloat}
    std_irf = lp_irf(estimate_lp(Y, shock_var, horizon; kwargs...))
    sm_irf = smooth_lp_irf(estimate_smooth_lp(Y, shock_var, horizon; lambda=lambda, kwargs...))
    (standard_irf=std_irf, smooth_irf=sm_irf,
     variance_reduction=mean(sm_irf.se.^2) / mean(std_irf.se.^2))
end
