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
Impulse Response Functions for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Frequentist IRF
# =============================================================================

"""
    irf(model, horizon; method=:cholesky, ci_type=:none, reps=200, conf_level=0.95, ...)

Compute IRFs with optional confidence intervals.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# CI types
`:none`, `:bootstrap`, `:theoretical`
"""
function irf(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    ci_type::Symbol=:none, reps::Int=200, conf_level::Real=0.95,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
) where {T<:AbstractFloat}

    n = nvars(model)
    Q = compute_Q(model, method, horizon, check_func, narrative_check;
                  transition_var=transition_var, regime_indicator=regime_indicator)
    point_irf = compute_irf(model, Q, horizon)

    ci_lower, ci_upper = zeros(T, horizon, n, n), zeros(T, horizon, n, n)
    if ci_type != :none
        sim_irfs = _simulate_irfs(model, method, horizon, check_func, narrative_check, ci_type, reps;
                                  transition_var=transition_var, regime_indicator=regime_indicator)
        alpha = (1 - T(conf_level)) / 2
        @inbounds for h in 1:horizon, v in 1:n, s in 1:n
            d = @view sim_irfs[:, h, v, s]
            ci_lower[h, v, s], ci_upper[h, v, s] = quantile(d, alpha), quantile(d, 1 - alpha)
        end
    end

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                       model.varnames, model.varnames, ci_type)
end

"""Simulate IRFs for confidence intervals (bootstrap or asymptotic)."""
function _simulate_irfs(model::VARModel{T}, method::Symbol, horizon::Int,
    check_func, narrative_check, ci_type::Symbol, reps::Int;
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    sim_irfs = zeros(T, reps, horizon, n, n)

    if ci_type == :bootstrap
        U, T_eff = model.U, size(model.U, 1)
        Y_init = model.Y[1:p, :]

        Threads.@threads for r in 1:reps
            _suppress_warnings() do
                U_boot = U[rand(1:T_eff, T_eff), :]
                Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
                m = estimate_var(Y_boot, p; check_stability=false)
                Q = compute_Q(m, method, horizon, check_func, narrative_check;
                              transition_var=transition_var, regime_indicator=regime_indicator)
                sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
            end
        end
    elseif ci_type == :theoretical
        _, X = construct_var_matrices(model.Y, p)
        L_V, L_S = safe_cholesky(robust_inv(X'X)), safe_cholesky(model.Sigma)
        k = ncoefs(model)

        Threads.@threads for r in 1:reps
            _suppress_warnings() do
                B_star = model.B + L_V * randn(T, k, n) * L_S'
                m = VARModel(zeros(T, 0, n), p, B_star, zeros(T, 0, n), model.Sigma, zero(T), zero(T), zero(T))
                Q = compute_Q(m, method, horizon, check_func, narrative_check;
                              transition_var=transition_var, regime_indicator=regime_indicator)
                sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
            end
        end
    end
    sim_irfs
end

"""Simulate VAR data from initial conditions and innovations."""
function _simulate_var(Y_init::AbstractMatrix{T}, B::AbstractMatrix{T},
                       U::AbstractMatrix{T}, T_total::Int) where {T<:AbstractFloat}
    p, n = size(Y_init)
    Y = zeros(T, T_total, n)
    Y[1:p, :] = Y_init

    A = extract_ar_coefficients(B, n, p)
    intercept = @view B[1, :]

    @inbounds for t in (p+1):T_total
        Y[t, :] = intercept
        for i in 1:p
            Y[t, :] .+= A[i] * @view(Y[t-i, :])
        end
        Y[t, :] .+= @view(U[t-p, :])
    end
    Y
end

# =============================================================================
# Bayesian IRF
# =============================================================================

"""
    irf(post::BVARPosterior, horizon; method=:cholesky, quantiles=[0.16, 0.5, 0.84], ...)

Compute Bayesian IRFs from posterior draws with posterior quantiles.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

Uses `process_posterior_samples` and `compute_posterior_quantiles` from bayesian_utils.jl.
"""
function irf(post::BVARPosterior, horizon::Int;
    method::Symbol=:cholesky, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    check_func=nothing, narrative_check=nothing, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    threaded::Bool=false,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
)
    use_data = isempty(data) ? post.data : data
    _validate_narrative_data(method, use_data)

    n = post.n
    ET = eltype(use_data)

    # Process posterior samples using shared utility
    results, samples = process_posterior_samples(post,
        (m, Q, h) -> compute_irf(m, Q, h);
        data=use_data, method=method, horizon=horizon,
        check_func=check_func, narrative_check=narrative_check,
        transition_var=transition_var, regime_indicator=regime_indicator
    )

    # Stack results into single array
    all_irfs = stack_posterior_results(results, (horizon, n, n), ET)

    # Compute quantiles using shared utility (threaded for large arrays)
    q_vec = ET.(quantiles)
    use_threaded = threaded || (samples * horizon * n * n > 100000)
    irf_q, irf_m = compute_posterior_quantiles(all_irfs, q_vec; threaded=use_threaded)

    BayesianImpulseResponse{ET}(irf_q, irf_m, horizon, post.varnames, post.varnames, q_vec)
end

# Deprecated wrapper for old (chain, p, n, horizon) signature
function irf(post::BVARPosterior, p::Int, n::Int, horizon::Int; kwargs...)
    irf(post, horizon; kwargs...)
end

# =============================================================================
# Structural LP IRF Accessor
# =============================================================================

"""
    irf(slp::StructuralLP) -> ImpulseResponse

Extract the impulse response object from a structural LP result.
"""
irf(slp::StructuralLP) = slp.irf

# =============================================================================
# Local Projection IRF
# =============================================================================

"""
    lp_irf(model::LPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract impulse response function with confidence intervals from LP model.
"""
function lp_irf(model::LPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    irf_data = extract_shock_irf(model.B, model.vcov, model.response_vars, 2;
                                  conf_level=conf_level)

    response_names = default_var_names(length(model.response_vars); prefix="Var")
    shock_name = "Shock $(model.shock_var)"
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(irf_data.values, irf_data.ci_lower, irf_data.ci_upper,
                         irf_data.se, model.horizon, response_names, shock_name,
                         cov_type_sym, T(conf_level))
end

"""
    lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) -> LPImpulseResponse

Convenience function: estimate LP and extract IRF in one call.
"""
function lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; conf_level::Real=0.95, kwargs...)
    model = estimate_lp(Y, shock_var, horizon; kwargs...)
    lp_irf(model; conf_level=conf_level)
end

# =============================================================================
# Cumulative IRF
# =============================================================================

"""
    cumulative_irf(irf::LPImpulseResponse{T}) -> LPImpulseResponse{T}

Compute cumulative impulse response: Σₛ₌₀ʰ β_s.
"""
function cumulative_irf(irf::LPImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf.values, dims=1)
    cum_se = sqrt.(cumsum(irf.se.^2, dims=1))

    z = T(quantile(Normal(), 1 - (1 - irf.conf_level) / 2))
    cum_ci_lower = cum_values .- z .* cum_se
    cum_ci_upper = cum_values .+ z .* cum_se

    LPImpulseResponse{T}(cum_values, cum_ci_lower, cum_ci_upper, cum_se, irf.horizon,
                         irf.response_vars, irf.shock_var, irf.cov_type, irf.conf_level)
end

"""
    cumulative_irf(irf_result::ImpulseResponse{T}) -> ImpulseResponse{T}

Compute cumulative impulse response for VAR models: Σₛ₌₀ʰ IRF_s.
"""
function cumulative_irf(irf_result::ImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf_result.values, dims=1)
    cum_lower = cumsum(irf_result.ci_lower, dims=1)
    cum_upper = cumsum(irf_result.ci_upper, dims=1)
    ImpulseResponse{T}(cum_values, cum_lower, cum_upper, irf_result.horizon,
                       irf_result.variables, irf_result.shocks, irf_result.ci_type)
end

"""
    cumulative_irf(irf_result::BayesianImpulseResponse{T}) -> BayesianImpulseResponse{T}

Compute cumulative Bayesian impulse response: Σₛ₌₀ʰ IRF_s.
"""
function cumulative_irf(irf_result::BayesianImpulseResponse{T}) where {T<:AbstractFloat}
    cum_quantiles = cumsum(irf_result.quantiles, dims=1)
    cum_mean = cumsum(irf_result.mean, dims=1)
    BayesianImpulseResponse{T}(cum_quantiles, cum_mean, irf_result.horizon,
                               irf_result.variables, irf_result.shocks, irf_result.quantile_levels)
end
