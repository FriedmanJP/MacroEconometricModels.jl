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
Forecast Error Variance Decomposition for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Frequentist FEVD
# =============================================================================

"""
    fevd(model, horizon; method=:cholesky, ...) -> FEVD

Compute FEVD showing proportion of h-step forecast error variance attributable to each shock.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.
"""
function fevd(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
) where {T<:AbstractFloat}
    irf_result = irf(model, horizon; method, check_func, narrative_check,
                     transition_var=transition_var, regime_indicator=regime_indicator)
    decomp, props = _compute_fevd(irf_result.values, nvars(model), horizon)
    FEVD{T}(decomp, props, model.varnames, model.varnames)
end

"""Compute FEVD from IRF array: decomposition[i,j,h] = cumulative MSE contribution."""
function _compute_fevd(irfs::Array{T,3}, n::Int, horizon::Int) where {T<:AbstractFloat}
    decomp, props = zeros(T, n, n, horizon), zeros(T, n, n, horizon)
    mse = zeros(T, n, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n
            total = zero(T)
            for j in 1:n
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irfs[h, i, j]^2
                total += decomp[i, j, h]
            end
            mse[i, h] = total
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    decomp, props
end

# =============================================================================
# Bayesian FEVD
# =============================================================================

"""
    fevd(post::BVARPosterior, horizon; quantiles=[0.16, 0.5, 0.84], ...) -> BayesianFEVD

Compute Bayesian FEVD from posterior draws with posterior quantiles.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

Uses `process_posterior_samples` and `compute_posterior_quantiles` from bayesian_utils.jl.
"""
# =============================================================================
# Structural LP FEVD — see lp_fevd.jl (Gorodnichenko & Lee 2019)
# =============================================================================

function fevd(post::BVARPosterior, horizon::Int;
    method::Symbol=:cholesky, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    check_func=nothing, narrative_check=nothing, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    threaded::Bool=false, point_estimate::Symbol=:median,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
)
    use_data = isempty(data) ? post.data : data
    _validate_narrative_data(method, use_data)

    n = post.n
    ET = eltype(use_data)

    # Process posterior samples - compute FEVD proportions for each
    results, samples = process_posterior_samples(post,
        (m, Q, h) -> begin
            irf_vals = compute_irf(m, Q, h)
            _, props = _compute_fevd(irf_vals, nvars(m), h)
            props  # Returns (n, n, horizon)
        end;
        data=use_data, method=method, horizon=horizon,
        check_func=check_func, narrative_check=narrative_check,
        transition_var=transition_var, regime_indicator=regime_indicator
    )

    # Stack results: samples are (n, n, horizon), need to rearrange to (horizon, n, n) for output
    all_fevds = zeros(ET, samples, horizon, n, n)
    @inbounds for s in 1:samples
        for h in 1:horizon, v in 1:n, sh in 1:n
            all_fevds[s, h, v, sh] = results[s][v, sh, h]
        end
    end

    # Compute quantiles using shared utility
    q_vec = ET.(quantiles)
    use_threaded = threaded || (samples * horizon * n * n > 100000)
    fevd_q, fevd_m = compute_posterior_quantiles(all_fevds, q_vec; threaded=use_threaded, central=point_estimate)

    BayesianFEVD{ET}(fevd_q, fevd_m, horizon, post.varnames, post.varnames, q_vec)
end

# Deprecated wrapper for old (chain, p, n, horizon) signature
function fevd(post::BVARPosterior, p::Int, n::Int, horizon::Int; kwargs...)
    fevd(post, horizon; kwargs...)
end
