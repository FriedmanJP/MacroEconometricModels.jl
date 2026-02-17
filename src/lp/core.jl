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
Core Local Projection estimation and shared utility functions.

This module provides:
- Shared utility functions for all LP variants
- Core LP estimation (Jordà 2005)
- IRF extraction and cumulative IRF

Note: HAC covariance estimators are defined in covariance_estimators.jl

References:
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections."
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Shared Utility Functions
# =============================================================================

"""
    create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where T

Create covariance estimator from symbol specification.
Eliminates repeated if/else patterns across LP variants.
"""
function create_cov_estimator(cov_type::Symbol, ::Type{T}; bandwidth::Int=0) where {T<:AbstractFloat}
    if cov_type == :newey_west
        NeweyWestEstimator{T}(bandwidth, :bartlett, false)
    elseif cov_type == :white
        WhiteEstimator()
    elseif cov_type == :driscoll_kraay
        DriscollKraayEstimator{T}(bandwidth, :bartlett)
    else
        throw(ArgumentError("cov_type must be :newey_west, :white, or :driscoll_kraay"))
    end
end

"""
    compute_horizon_bounds(T_obs::Int, h::Int, lags::Int) -> (t_start, t_end)

Compute valid observation bounds for horizon h.
"""
function compute_horizon_bounds(T_obs::Int, h::Int, lags::Int)
    t_start = lags + 1
    t_end = T_obs - h
    if t_end < t_start
        throw(ArgumentError("Not enough observations for horizon $h with $lags lags"))
    end
    (t_start, t_end)
end

"""
    build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                          response_vars::Vector{Int}) where T

Build response matrix Y_h at horizon h.
"""
function build_response_matrix(Y::AbstractMatrix{T}, h::Int, t_start::Int, t_end::Int,
                                response_vars::Vector{Int}) where {T<:AbstractFloat}
    T_eff = t_end - t_start + 1
    n_response = length(response_vars)
    Y_h = Matrix{T}(undef, T_eff, n_response)
    @inbounds for (j, var) in enumerate(response_vars)
        for (i, t) in enumerate(t_start:t_end)
            Y_h[i, j] = Y[t + h, var]
        end
    end
    Y_h
end

"""
    build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                           t_start::Int, t_end::Int, lags::Int, start_col::Int) where T

Fill control (lagged Y) columns into regressor matrix X_h.
Returns the next available column index.
"""
function build_control_columns!(X_h::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                 t_start::Int, t_end::Int, lags::Int, start_col::Int) where {T<:AbstractFloat}
    n = size(Y, 2)
    col = start_col
    @inbounds for (i, t) in enumerate(t_start:t_end)
        col_local = start_col
        for lag in 1:lags
            for var in 1:n
                X_h[i, col_local] = Y[t - lag, var]
                col_local += 1
            end
        end
    end
    start_col + n * lags
end

"""
    compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                              cov_estimator::AbstractCovarianceEstimator) where T

Compute block-diagonal robust covariance for multi-equation system.
"""
function compute_block_robust_vcov(X::AbstractMatrix{T}, U::AbstractMatrix{T},
                                    cov_estimator::AbstractCovarianceEstimator) where {T<:AbstractFloat}
    n_eq = size(U, 2)
    k = size(X, 2)
    V = zeros(T, k * n_eq, k * n_eq)
    @inbounds for eq in 1:n_eq
        V_eq = robust_vcov(X, @view(U[:, eq]), cov_estimator)
        idx = ((eq-1)*k + 1):(eq*k)
        V[idx, idx] .= V_eq
    end
    V
end

"""
    _lp_robust_vcov(X_h, U_h, cov_estimator, h)

Horizon-aware robust variance-covariance for LP regressions.

LP residuals at horizon h have MA(h-1) serial correlation (Jordà 2005),
so the Newey-West bandwidth must be at least h+1. When automatic bandwidth
selection (`bandwidth == 0`) yields a smaller value, this function enforces
the floor `max(m̂_NW, h+1)`.

For non-NW estimators (White, Driscoll-Kraay), falls through to the
standard `compute_block_robust_vcov`.
"""
function _lp_robust_vcov(X_h::AbstractMatrix{T}, U_h::AbstractMatrix{T},
                          cov_estimator::NeweyWestEstimator{T}, h::Int) where {T}
    if cov_estimator.bandwidth == 0 && h > 0
        n_eq, k = size(U_h, 2), size(X_h, 2)
        V = zeros(T, k * n_eq, k * n_eq)
        for eq in 1:n_eq
            resid_eq = @view(U_h[:, eq])
            auto_bw = optimal_bandwidth_nw(resid_eq)
            effective_bw = max(auto_bw, h + 1)
            V_eq = newey_west(X_h, resid_eq; bandwidth=effective_bw,
                             kernel=cov_estimator.kernel, prewhiten=cov_estimator.prewhiten)
            idx = ((eq-1)*k + 1):(eq*k)
            V[idx, idx] .= V_eq
        end
        return V
    end
    compute_block_robust_vcov(X_h, U_h, cov_estimator)
end

# Passthrough for White/DriscollKraay
_lp_robust_vcov(X_h::AbstractMatrix{T}, U_h::AbstractMatrix{T},
                cov::AbstractCovarianceEstimator, h::Int) where {T} =
    compute_block_robust_vcov(X_h, U_h, cov)

"""
    extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                      response_vars::Vector{Int}, shock_coef_idx::Int;
                      conf_level::Real=0.95) where T

Generic IRF extraction from coefficient and covariance vectors.
Works for LPModel, LPIVModel, PropensityLPModel.
"""
function extract_shock_irf(B::Vector{Matrix{T}}, vcov::Vector{Matrix{T}},
                           response_vars::Vector{Int}, shock_coef_idx::Int;
                           conf_level::Real=0.95) where {T<:AbstractFloat}
    H = length(B) - 1
    n_response = length(response_vars)
    k = size(B[1], 1)

    values = Matrix{T}(undef, H + 1, n_response)
    se = Matrix{T}(undef, H + 1, n_response)

    @inbounds for h in 0:H
        B_h = B[h + 1]
        V_h = vcov[h + 1]
        for (j, _) in enumerate(response_vars)
            values[h + 1, j] = B_h[shock_coef_idx, j]
            var_idx = (j - 1) * k + shock_coef_idx
            se[h + 1, j] = sqrt(V_h[var_idx, var_idx])
        end
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = values .- z .* se
    ci_upper = values .+ z .* se

    (values=values, se=se, ci_lower=ci_lower, ci_upper=ci_upper)
end

# Note: Kernel functions, bandwidth selection, Newey-West, White, Driscoll-Kraay,
# long_run_variance, and long_run_covariance are now in covariance_estimators.jl

# =============================================================================
# LP Matrix Construction
# =============================================================================

"""
    construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                          response_vars::Vector{Int}=collect(1:size(Y,2))) where T

Construct regressor and response matrices for LP regression at horizon h.

Returns: (Y_h, X_h, valid_idx)
"""
function construct_lp_matrices(Y::AbstractMatrix{T}, shock_var::Int, h::Int, lags::Int;
                                response_vars::Vector{Int}=collect(1:size(Y, 2))) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    t_start, t_end = compute_horizon_bounds(T_obs, h, lags)
    T_eff = t_end - t_start + 1

    # Response matrix
    Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)

    # Regressor matrix: [1, shock_t, y_{t-1}, ..., y_{t-lags}]
    k = 2 + n * lags
    X_h = Matrix{T}(undef, T_eff, k)

    @inbounds for (i, t) in enumerate(t_start:t_end)
        X_h[i, 1] = one(T)
        X_h[i, 2] = Y[t, shock_var]
    end
    build_control_columns!(X_h, Y, t_start, t_end, lags, 3)

    valid_idx = collect(t_start:t_end)
    (Y_h, X_h, valid_idx)
end

# =============================================================================
# Core LP Estimation
# =============================================================================

"""
    estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y,2)),
                cov_type::Symbol=:newey_west, bandwidth::Int=0,
                conf_level::Real=0.95) -> LPModel{T}

Estimate Local Projection impulse response functions (Jordà 2005).

The LP regression for horizon h:
    y_{t+h} = α_h + β_h * shock_t + Γ_h * controls_t + ε_{t+h}
"""
function estimate_lp(Y::AbstractMatrix{T}, shock_var::Int, horizon::Int;
                     lags::Int=4, response_vars::Vector{Int}=collect(1:size(Y, 2)),
                     cov_type::Symbol=:newey_west, bandwidth::Int=0,
                     conf_level::Real=0.95) where {T<:AbstractFloat}
    _validate_data(Y, "Y")
    T_obs, n = size(Y)

    validate_positive(horizon, "horizon")
    @assert 1 <= shock_var <= n "shock_var must be in 1:$n"
    @assert all(1 .<= response_vars .<= n) "response_vars must be in 1:$n"
    @assert lags >= 0 "lags must be non-negative"
    @assert T_obs > lags + horizon + 1 "Not enough observations"

    cov_estimator = create_cov_estimator(cov_type, T; bandwidth=bandwidth)

    B = Vector{Matrix{T}}(undef, horizon + 1)
    residuals = Vector{Matrix{T}}(undef, horizon + 1)
    vcov = Vector{Matrix{T}}(undef, horizon + 1)
    T_eff = Vector{Int}(undef, horizon + 1)

    for h in 0:horizon
        Y_h, X_h, valid_idx = construct_lp_matrices(Y, shock_var, h, lags;
                                                     response_vars=response_vars)
        T_eff[h + 1] = length(valid_idx)

        # OLS: B_h = (X'X)^{-1} X'Y
        XtX_inv = robust_inv(X_h' * X_h)
        B_h = XtX_inv * (X_h' * Y_h)
        U_h = Y_h - X_h * B_h

        B[h + 1] = B_h
        residuals[h + 1] = U_h
        vcov[h + 1] = _lp_robust_vcov(X_h, U_h, cov_estimator, h)
    end

    LPModel(Matrix{T}(Y), shock_var, response_vars, horizon, lags,
            B, residuals, vcov, T_eff, cov_estimator)
end

# Float fallback
estimate_lp(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) =
    estimate_lp(Float64.(Y), shock_var, horizon; kwargs...)

# =============================================================================
# Multiple Shocks
# =============================================================================

"""
    estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                      kwargs...) -> Vector{LPModel{T}}

Estimate LP for multiple shock variables.
"""
function estimate_lp_multi(Y::AbstractMatrix{T}, shock_vars::Vector{Int}, horizon::Int;
                           kwargs...) where {T<:AbstractFloat}
    [estimate_lp(Y, shock, horizon; kwargs...) for shock in shock_vars]
end

# =============================================================================
# LP with Orthogonalized Shocks
# =============================================================================

"""
    estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                         lags::Int=4, cov_type::Symbol=:newey_west, kwargs...) -> Vector{LPModel{T}}

Estimate LP with Cholesky-orthogonalized shocks.
"""
function estimate_lp_cholesky(Y::AbstractMatrix{T}, horizon::Int;
                              lags::Int=4, cov_type::Symbol=:newey_west,
                              kwargs...) where {T<:AbstractFloat}
    T_obs, n = size(Y)

    var_model = estimate_var(Y, lags)
    U = var_model.U
    L = identify_cholesky(var_model)
    eps = (inv(L) * U')'

    Y_eff = Y[(lags+1):end, :]
    @assert size(eps, 1) == size(Y_eff, 1) "Dimension mismatch"

    models = Vector{LPModel{T}}(undef, n)
    for shock in 1:n
        Y_aug = hcat(eps[:, shock], Y_eff)
        models[shock] = estimate_lp(Y_aug, 1, horizon; lags=lags,
                                     response_vars=collect(2:(n+1)),
                                     cov_type=cov_type, kwargs...)
    end
    models
end

# =============================================================================
# Structural LP (Plagborg-Møller & Wolf 2021)
# =============================================================================

"""
    structural_lp(Y::AbstractMatrix{T}, horizon::Int;
                  method=:cholesky, lags=4, var_lags=nothing,
                  cov_type=:newey_west, conf_level=0.95,
                  ci_type=:none, reps=200,
                  check_func=nothing, narrative_check=nothing,
                  max_draws=1000,
                  transition_var=nothing, regime_indicator=nothing) -> StructuralLP{T}

Estimate structural LP impulse responses using VAR-based identification with LP estimation.

Algorithm:
1. Estimate VAR(p) → obtain Σ and residuals
2. Compute rotation matrix Q via chosen identification method
3. Compute structural shocks ε_t = Q'L⁻¹u_t
4. For each shock j, run LP with ε_j as regressor and Y_eff as responses
5. Stack into 3D IRF array: irfs[h, i, j]

# Arguments
- `Y`: T × n data matrix
- `horizon`: Maximum IRF horizon H

# Keyword Arguments
- `method`: Identification method
- `lags`: Number of LP control lags (default: 4)
- `var_lags`: VAR lag order for identification (default: same as `lags`)
- `cov_type`: HAC estimator (:newey_west, :white)
- `conf_level`: Confidence level for CIs (default: 0.95)
- `ci_type`: CI method (:none, :bootstrap)
- `reps`: Number of bootstrap replications
- `check_func`: Sign restriction check function (for :sign/:narrative)
- `narrative_check`: Narrative check function (for :narrative)
- `max_draws`: Maximum draws for sign/narrative identification
- `transition_var`: Transition variable (for :smooth_transition)
- `regime_indicator`: Regime indicator (for :external_volatility)

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# Returns
`StructuralLP{T}` with 3D IRFs, structural shocks, VAR model, and individual LP models.

# References
Plagborg-Møller, M. & Wolf, C. K. (2021). "Local Projections and VARs Estimate the Same
Impulse Responses." *Econometrica*, 89(2), 955–980.
"""
function structural_lp(Y::AbstractMatrix{T}, horizon::Int;
                       method::Symbol=:cholesky, lags::Int=4,
                       var_lags::Union{Nothing,Int}=nothing,
                       cov_type::Symbol=:newey_west, conf_level::Real=0.95,
                       ci_type::Symbol=:none, reps::Int=200,
                       check_func=nothing, narrative_check=nothing,
                       max_draws::Int=1000,
                       transition_var::Union{Nothing,AbstractVector}=nothing,
                       regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    p = isnothing(var_lags) ? lags : var_lags

    validate_positive(horizon, "horizon")
    @assert T_obs > p + horizon + 1 "Not enough observations"

    # Step 1: Estimate VAR for identification
    var_model = estimate_var(Y, p)

    # Step 2: Compute identification matrix Q
    Q = compute_Q(var_model, method, horizon, check_func, narrative_check;
                  max_draws=max_draws, transition_var=transition_var, regime_indicator=regime_indicator)

    # Step 3: Compute structural shocks
    eps = compute_structural_shocks(var_model, Q)

    # Step 4: Run LP for each shock
    Y_eff = Y[(p+1):end, :]
    @assert size(eps, 1) == size(Y_eff, 1) "Dimension mismatch between shocks and effective Y"

    response_vars_aug = collect(2:(n+1))
    lp_models_raw = Vector{Any}(undef, n)
    for shock in 1:n
        Y_aug = hcat(eps[:, shock], Y_eff)
        lp_models_raw[shock] = estimate_lp(Y_aug, 1, horizon; lags=lags,
                                            response_vars=response_vars_aug,
                                            cov_type=cov_type, conf_level=conf_level)
    end
    # Determine element type from first model (may differ from T due to promotions)
    ET = eltype(lp_models_raw[1].Y)
    lp_models = Vector{LPModel{ET}}(lp_models_raw)

    # Step 5: Build 3D IRF array and SE array
    irfs, se_arr = _build_structural_lp_arrays(lp_models, n, horizon, ET)

    # Step 6: Bootstrap CIs if requested
    ci_lower, ci_upper = zeros(ET, horizon, n, n), zeros(ET, horizon, n, n)
    ci_sym = :none
    if ci_type == :bootstrap
        ci_lower, ci_upper = _structural_lp_bootstrap(Matrix{ET}(Y), horizon, n, p, method,
                                                       lags, cov_type, reps, ET(conf_level),
                                                       check_func, narrative_check, max_draws;
                                                       transition_var=transition_var,
                                                       regime_indicator=regime_indicator)
        ci_sym = :bootstrap
    end

    irf_result = ImpulseResponse{ET}(irfs, ci_lower, ci_upper, horizon,
                                      var_model.varnames, var_model.varnames, ci_sym)

    StructuralLP{ET}(irf_result, Matrix{ET}(eps), var_model, Matrix{ET}(Q), method,
                     lags, cov_type, se_arr, lp_models)
end

# Float fallback
structural_lp(Y::AbstractMatrix, horizon::Int; kwargs...) =
    structural_lp(Float64.(Y), horizon; kwargs...)

"""Extract 3D IRF and SE arrays from per-shock LP models."""
function _build_structural_lp_arrays(lp_models::Vector{LPModel{T}}, n::Int,
                                      horizon::Int, ::Type{T}) where {T<:AbstractFloat}
    irfs = zeros(T, horizon, n, n)
    se_arr = zeros(T, horizon, n, n)

    for shock in 1:n
        irf_data = extract_shock_irf(lp_models[shock].B, lp_models[shock].vcov,
                                      lp_models[shock].response_vars, 2)
        # extract_shock_irf returns (H+1) rows (h=0,...,H); IRF array uses h=1,...,H
        for h in 1:horizon
            for resp in 1:n
                irfs[h, resp, shock] = irf_data.values[h+1, resp]
                se_arr[h, resp, shock] = irf_data.se[h+1, resp]
            end
        end
    end
    irfs, se_arr
end

"""Block bootstrap for structural LP confidence intervals."""
function _structural_lp_bootstrap(Y::AbstractMatrix{T}, horizon::Int, n::Int, p::Int,
                                   method::Symbol, lags::Int, cov_type::Symbol,
                                   reps::Int, conf_level::T,
                                   check_func, narrative_check,
                                   max_draws::Int;
                                   transition_var::Union{Nothing,AbstractVector}=nothing,
                                   regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing) where {T<:AbstractFloat}
    T_obs = size(Y, 1)
    sim_irfs = zeros(T, reps, horizon, n, n)
    block_size = max(1, round(Int, T_obs^(1/3)))

    Threads.@threads for r in 1:reps
        # Block bootstrap on Y
        Y_boot = _block_bootstrap(Y, block_size)
        try
            _suppress_warnings() do
                var_m = estimate_var(Y_boot, p; check_stability=false)
                Q_r = compute_Q(var_m, method, horizon, check_func, narrative_check;
                                max_draws=max_draws, transition_var=transition_var, regime_indicator=regime_indicator)
                eps_r = compute_structural_shocks(var_m, Q_r)

                Y_eff_r = Y_boot[(p+1):end, :]
                for shock in 1:n
                    Y_aug = hcat(eps_r[:, shock], Y_eff_r)
                    lp_r = estimate_lp(Y_aug, 1, horizon; lags=lags,
                                        response_vars=collect(2:(n+1)), cov_type=cov_type)
                    irf_data = extract_shock_irf(lp_r.B, lp_r.vcov, lp_r.response_vars, 2)
                    for h in 1:horizon, resp in 1:n
                        sim_irfs[r, h, resp, shock] = irf_data.values[h+1, resp]
                    end
                end
            end
        catch
            # If bootstrap draw fails (e.g., singular matrix), skip
            continue
        end
    end

    alpha = (1 - conf_level) / 2
    ci_lower = zeros(T, horizon, n, n)
    ci_upper = zeros(T, horizon, n, n)
    @inbounds for h in 1:horizon, v in 1:n, s in 1:n
        d = @view sim_irfs[:, h, v, s]
        ci_lower[h, v, s] = quantile(d, alpha)
        ci_upper[h, v, s] = quantile(d, 1 - alpha)
    end
    ci_lower, ci_upper
end

"""Generate a block bootstrap sample from matrix Y."""
function _block_bootstrap(Y::AbstractMatrix{T}, block_size::Int) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    n_blocks = ceil(Int, T_obs / block_size)
    Y_boot = Matrix{T}(undef, n_blocks * block_size, n)

    idx = 1
    for _ in 1:n_blocks
        start = rand(1:max(1, T_obs - block_size + 1))
        len = min(block_size, T_obs - start + 1)
        Y_boot[idx:idx+len-1, :] = Y[start:start+len-1, :]
        idx += len
    end
    Y_boot[1:T_obs, :]
end

# =============================================================================
# Model Comparison
# =============================================================================

"""
    compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where T

Compare VAR-based and LP-based impulse responses.
"""
function compare_var_lp(Y::AbstractMatrix{T}, horizon::Int; lags::Int=4) where {T<:AbstractFloat}
    n = size(Y, 2)

    var_model = estimate_var(Y, lags)
    var_result = irf(var_model, horizon; method=:cholesky)

    lp_models = estimate_lp_cholesky(Y, horizon; lags=lags)
    lp_results = [lp_irf(m) for m in lp_models]

    var_values = var_result.values
    lp_values = zeros(T, horizon, n, n)

    for shock in 1:n
        for (h_idx, h) in enumerate(1:horizon)
            for resp in 1:n
                lp_values[h_idx, resp, shock] = lp_results[shock].values[h + 1, resp]
            end
        end
    end

    (var_irf=var_values, lp_irf=lp_values, difference=var_values - lp_values)
end

# =============================================================================
# StatsAPI predict / residuals
# =============================================================================

"""
    _lp_predict_at_horizon(Y, response_vars, residuals_h, T_eff_h, h)

Reconstruct fitted values at horizon h via Y_h − residuals_h identity.
"""
function _lp_predict_at_horizon(Y::Matrix{T}, response_vars::Vector{Int},
                                 residuals_h::Matrix{T}, T_eff_h::Int, h::Int) where {T}
    T_obs = size(Y, 1)
    t_end = T_obs - h
    t_start = t_end - T_eff_h + 1
    Y_h = build_response_matrix(Y, h, t_start, t_end, response_vars)
    Y_h - residuals_h
end

# --- Per-horizon LP models (LPModel, LPIVModel, StateLPModel, PropensityLPModel) ---

for LP in (:LPModel, :LPIVModel, :StateLPModel, :PropensityLPModel)
    @eval begin
        function StatsAPI.predict(model::$LP{T}, h::Int) where {T}
            _lp_predict_at_horizon(model.Y, model.response_vars,
                                   residuals(model, h), model.T_eff[h + 1], h)
        end

        function StatsAPI.predict(model::$LP{T}) where {T}
            [predict(model, h) for h in 0:model.horizon]
        end
    end
end

# --- LPIVModel: add missing per-horizon residuals ---
StatsAPI.residuals(model::LPIVModel, h::Int) = model.residuals[h + 1]

# --- PropensityLPModel: add missing per-horizon residuals ---
StatsAPI.residuals(model::PropensityLPModel, h::Int) = model.residuals[h + 1]

# --- SmoothLPModel: pooled predict ---

function StatsAPI.predict(model::SmoothLPModel{T}) where {T}
    # Reconstruct pooled Y_h to match the stacked residuals
    Y_pooled = vcat([begin
        T_obs = size(model.Y, 1)
        t_start, t_end = compute_horizon_bounds(T_obs, h, model.lags)
        build_response_matrix(model.Y, h, t_start, t_end, model.response_vars)
    end for h in 0:model.horizon]...)
    Y_pooled - model.residuals
end
