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

# =============================================================================
# table() - Extract data as matrix
# =============================================================================

"""
    table(irf::ImpulseResponse, var, shock; horizons=nothing) -> Matrix

Extract IRF values for a variable-shock pair.
Returns matrix with columns: [Horizon, IRF] or [Horizon, IRF, CI_lo, CI_hi].
"""
function table(irf::ImpulseResponse{T}, var::Int, shock::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    n_vars, n_shocks = length(irf.variables), length(irf.shocks)
    @assert 1 <= var <= n_vars "Variable index out of bounds"
    @assert 1 <= shock <= n_shocks "Shock index out of bounds"

    hs = isnothing(horizons) ? (1:irf.horizon) : horizons
    has_ci = irf.ci_type != :none

    ncols = has_ci ? 4 : 2
    result = Matrix{T}(undef, length(hs), ncols)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        result[i, 2] = irf.values[h, var, shock]
        if has_ci
            result[i, 3] = irf.ci_lower[h, var, shock]
            result[i, 4] = irf.ci_upper[h, var, shock]
        end
    end
    result
end

function table(irf::ImpulseResponse, var::String, shock::String; kwargs...)
    vi, si = _validate_var_shock_indices(var, shock, irf.variables, irf.shocks)
    table(irf, vi, si; kwargs...)
end

"""
    table(irf::BayesianImpulseResponse, var, shock; horizons=nothing) -> Matrix

Extract Bayesian IRF values. Returns [Horizon, Mean, Q1, Q2, ...].
"""
function table(irf::BayesianImpulseResponse{T}, var::Int, shock::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    @assert 1 <= var <= length(irf.variables)
    @assert 1 <= shock <= length(irf.shocks)

    hs = isnothing(horizons) ? (1:irf.horizon) : horizons
    nq = length(irf.quantile_levels)

    result = Matrix{T}(undef, length(hs), 2 + nq)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        result[i, 2] = irf.mean[h, var, shock]
        for q in 1:nq
            result[i, 2 + q] = irf.quantiles[h, var, shock, q]
        end
    end
    result
end

function table(irf::BayesianImpulseResponse, var::String, shock::String; kwargs...)
    vi, si = _validate_var_shock_indices(var, shock, irf.variables, irf.shocks)
    table(irf, vi, si; kwargs...)
end

"""
    table(f::FEVD, var; horizons=nothing) -> Matrix

Extract FEVD proportions for a variable. Returns [Horizon, Shock1, Shock2, ...].
"""
function table(f::FEVD{T}, var::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    n_vars, n_shocks, H = size(f.proportions)
    @assert 1 <= var <= n_vars

    hs = isnothing(horizons) ? (1:H) : horizons

    result = Matrix{T}(undef, length(hs), n_shocks + 1)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        for j in 1:n_shocks
            result[i, j + 1] = f.proportions[var, j, h]
        end
    end
    result
end

"""
    table(f::BayesianFEVD, var; horizons=nothing, stat=:mean) -> Matrix

Extract Bayesian FEVD values. stat can be :mean or quantile index.
"""
function table(f::BayesianFEVD{T}, var::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing,
               stat::Union{Symbol,Int}=:mean) where {T}
    @assert 1 <= var <= length(f.variables)

    hs = isnothing(horizons) ? (1:f.horizon) : horizons
    n_shocks = length(f.shocks)

    result = Matrix{T}(undef, length(hs), n_shocks + 1)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        for j in 1:n_shocks
            result[i, j + 1] = stat == :mean ? f.mean[h, var, j] : f.quantiles[h, var, j, stat]
        end
    end
    result
end

"""
    table(hd::HistoricalDecomposition, var; periods=nothing) -> Matrix

Extract HD contributions for a variable.
Returns [Period, Actual, Shock1, ..., ShockN, Initial].
"""
function table(hd::HistoricalDecomposition{T}, var::Int;
               periods::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    @assert 1 <= var <= length(hd.variables)

    ps = isnothing(periods) ? (1:hd.T_eff) : periods
    n_shocks = length(hd.shock_names)

    result = Matrix{T}(undef, length(ps), n_shocks + 3)
    for (i, t) in enumerate(ps)
        result[i, 1] = t
        result[i, 2] = hd.actual[t, var]
        for j in 1:n_shocks
            result[i, j + 2] = hd.contributions[t, var, j]
        end
        result[i, end] = hd.initial_conditions[t, var]
    end
    result
end

"""
    table(hd::BayesianHistoricalDecomposition, var; periods=nothing, stat=:mean) -> Matrix

Extract Bayesian HD contributions. stat can be :mean or quantile index.
"""
function table(hd::BayesianHistoricalDecomposition{T}, var::Int;
               periods::Union{Nothing,AbstractVector{Int}}=nothing,
               stat::Union{Symbol,Int}=:mean) where {T}
    @assert 1 <= var <= length(hd.variables)

    ps = isnothing(periods) ? (1:hd.T_eff) : periods
    n_shocks = length(hd.shock_names)

    result = Matrix{T}(undef, length(ps), n_shocks + 3)
    for (i, t) in enumerate(ps)
        result[i, 1] = t
        result[i, 2] = hd.actual[t, var]
        for j in 1:n_shocks
            result[i, j + 2] = stat == :mean ? hd.mean[t, var, j] : hd.quantiles[t, var, j, stat]
        end
        result[i, end] = stat == :mean ? hd.initial_mean[t, var] : hd.initial_quantiles[t, var, stat]
    end
    result
end

# --- VolatilityForecast ---

"""
    table(fc::VolatilityForecast) -> Matrix

Extract volatility forecast data.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi, SE].
"""
function table(fc::VolatilityForecast{T}) where {T}
    h = fc.horizon
    result = Matrix{T}(undef, h, 5)
    for i in 1:h
        result[i, 1] = i
        result[i, 2] = fc.forecast[i]
        result[i, 3] = fc.ci_lower[i]
        result[i, 4] = fc.ci_upper[i]
        result[i, 5] = fc.se[i]
    end
    result
end

# --- ARIMAForecast ---

"""
    table(fc::ARIMAForecast) -> Matrix

Extract ARIMA forecast data.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi, SE].
"""
function table(fc::ARIMAForecast{T}) where {T}
    h = fc.horizon
    result = Matrix{T}(undef, h, 5)
    for i in 1:h
        result[i, 1] = i
        result[i, 2] = fc.forecast[i]
        result[i, 3] = fc.ci_lower[i]
        result[i, 4] = fc.ci_upper[i]
        result[i, 5] = fc.se[i]
    end
    result
end

# --- FactorForecast ---

"""
    table(fc::FactorForecast, var_idx::Int; type=:observable) -> Matrix

Extract factor forecast data for a single variable.
`type=:observable` returns observable forecasts, `type=:factor` returns factor forecasts.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi].
"""
function table(fc::FactorForecast{T}, var_idx::Int; type::Symbol=:observable) where {T}
    h = fc.horizon
    if type == :observable
        N = size(fc.observables, 2)
        @assert 1 <= var_idx <= N "Variable index $var_idx out of bounds (1:$N)"
        result = Matrix{T}(undef, h, 4)
        for i in 1:h
            result[i, 1] = i
            result[i, 2] = fc.observables[i, var_idx]
            result[i, 3] = fc.observables_lower[i, var_idx]
            result[i, 4] = fc.observables_upper[i, var_idx]
        end
    else
        r = size(fc.factors, 2)
        @assert 1 <= var_idx <= r "Factor index $var_idx out of bounds (1:$r)"
        result = Matrix{T}(undef, h, 4)
        for i in 1:h
            result[i, 1] = i
            result[i, 2] = fc.factors[i, var_idx]
            result[i, 3] = fc.factors_lower[i, var_idx]
            result[i, 4] = fc.factors_upper[i, var_idx]
        end
    end
    result
end

# --- LPImpulseResponse ---

"""
    table(irf::LPImpulseResponse, var_idx::Int) -> Matrix

Extract LP IRF values for a response variable.
Returns matrix with columns: [Horizon, IRF, SE, CI_lo, CI_hi].
"""
function table(irf::LPImpulseResponse{T}, var_idx::Int) where {T}
    n_resp = length(irf.response_vars)
    @assert 1 <= var_idx <= n_resp "Variable index $var_idx out of bounds (1:$n_resp)"
    H = irf.horizon
    result = Matrix{T}(undef, H + 1, 5)
    for i in 0:H
        result[i + 1, 1] = i
        result[i + 1, 2] = irf.values[i + 1, var_idx]
        result[i + 1, 3] = irf.se[i + 1, var_idx]
        result[i + 1, 4] = irf.ci_lower[i + 1, var_idx]
        result[i + 1, 5] = irf.ci_upper[i + 1, var_idx]
    end
    result
end

function table(irf::LPImpulseResponse, var_name::String)
    idx = findfirst(==(var_name), irf.response_vars)
    isnothing(idx) && throw(ArgumentError("Variable '$var_name' not found in response_vars"))
    table(irf, idx)
end

