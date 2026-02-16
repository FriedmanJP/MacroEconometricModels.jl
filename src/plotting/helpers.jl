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
Internal helper functions for converting result struct fields to JSON data arrays.
"""

# =============================================================================
# Variable Resolution
# =============================================================================

"""Resolve a variable name or index to an integer index."""
function _resolve_var(var::Union{Int,String,Nothing}, names::Vector{String}, default::Int=1)
    var === nothing && return default
    var isa Int && return var
    idx = findfirst(==(var), names)
    idx === nothing && throw(ArgumentError("Variable '$var' not found. Available: $names"))
    idx
end

# =============================================================================
# Series Config JSON
# =============================================================================

"""Build series config JSON array for chart renderers."""
function _series_json(names::Vector{String}, colors::Vector{String};
                      keys::Union{Vector{String},Nothing}=nothing,
                      dash::Union{Vector{String},Nothing}=nothing)
    n = length(names)
    ks = something(keys, ["s$i" for i in 1:n])
    ds = something(dash, fill("", n))
    parts = String[]
    for i in 1:n
        c = colors[mod1(i, length(colors))]
        push!(parts, "{\"name\":$(_json(names[i])),\"color\":$(_json(c)),\"key\":$(_json(ks[i])),\"dash\":$(_json(ds[i]))}")
    end
    "[" * join(parts, ",") * "]"
end

# =============================================================================
# IRF Data
# =============================================================================

"""Convert IRF arrays to JSON data array with x, irf, ci_lower, ci_upper, zero."""
function _irf_data_json(values::AbstractVector, ci_lo::AbstractVector,
                        ci_up::AbstractVector, H::Int)
    rows = Vector{Pair{String,String}}[]
    for h in 1:H
        push!(rows, [
            "x" => _json(h - 1),
            "irf" => _json(values[h]),
            "ci_lo" => _json(ci_lo[h]),
            "ci_hi" => _json(ci_up[h]),
            "zero" => "0"
        ])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# FEVD Data
# =============================================================================

"""Convert FEVD proportions to JSON data array with x + shock columns."""
function _fevd_data_json(proportions::AbstractMatrix, shock_names::Vector{String}, H::Int)
    # proportions: H × n_shocks (for a single variable)
    rows = Vector{Pair{String,String}}[]
    for h in 1:H
        row = Pair{String,String}["x" => _json(h)]
        for (j, name) in enumerate(shock_names)
            push!(row, "s$j" => _json(proportions[h, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# HD Data
# =============================================================================

"""Convert HD contributions to JSON data array with x + shock columns."""
function _hd_data_json(contributions::AbstractMatrix, shock_names::Vector{String}, T_eff::Int)
    # contributions: T_eff × n_shocks (for a single variable)
    rows = Vector{Pair{String,String}}[]
    for t in 1:T_eff
        row = Pair{String,String}["x" => _json(t)]
        for (j, name) in enumerate(shock_names)
            push!(row, "s$j" => _json(contributions[t, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Filter Data
# =============================================================================

"""Convert filter trend/cycle to JSON data array."""
function _filter_data_json(trend::AbstractVector, cyc::AbstractVector;
                           original::Union{AbstractVector,Nothing}=nothing,
                           offset::Int=0)
    n = length(trend)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        row = Pair{String,String}["x" => _json(i + offset)]
        row_push = true
        push!(row, "trend" => _json(trend[i]))
        push!(row, "cycle" => _json(cyc[i]))
        if original !== nothing && i + offset <= length(original)
            push!(row, "orig" => _json(original[i + offset]))
        end
        push!(row, "zero" => "0")
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Forecast Data
# =============================================================================

"""Convert forecast + optional history to JSON data array."""
function _forecast_data_json(fc::AbstractVector, ci_lo::AbstractVector,
                             ci_up::AbstractVector;
                             history::Union{AbstractVector,Nothing}=nothing,
                             n_history::Int=50)
    rows = Vector{Pair{String,String}}[]
    start_idx = 0

    if history !== nothing
        nh = min(n_history, length(history))
        start_idx = -nh
        for i in 1:nh
            t = i - nh  # negative time index for history
            push!(rows, [
                "x" => _json(t),
                "hist" => _json(history[end - nh + i]),
                "fc" => "null",
                "ci_lo" => "null",
                "ci_hi" => "null"
            ])
        end
    end

    for i in 1:length(fc)
        push!(rows, [
            "x" => _json(i),
            "hist" => "null",
            "fc" => _json(fc[i]),
            "ci_lo" => _json(ci_lo[i]),
            "ci_hi" => _json(ci_up[i])
        ])
    end

    # Bridge point: connect history to forecast
    if history !== nothing && !isempty(rows)
        push!(rows[end - length(fc)], "fc" => _json(history[end]))
    end

    _json_array_of_objects(rows)
end

# =============================================================================
# Volatility Data
# =============================================================================

"""Convert returns + conditional variance to JSON data array."""
function _volatility_data_json(y::AbstractVector, cond_var::AbstractVector)
    n = length(y)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, [
            "x" => _json(i),
            "ret" => _json(y[i]),
            "vol" => _json(sqrt(max(cond_var[min(i, length(cond_var))], 0.0))),
            "std_resid" => _json(y[i] / sqrt(max(cond_var[min(i, length(cond_var))], 1e-10)))
        ])
    end
    _json_array_of_objects(rows)
end

"""Convert SV model to JSON with posterior quantile bands."""
function _sv_data_json(y::AbstractVector, vol_mean::AbstractVector,
                       vol_q::AbstractMatrix, q_levels::AbstractVector)
    n = length(y)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        row = Pair{String,String}[
            "x" => _json(i),
            "ret" => _json(y[i]),
            "vol_mean" => _json(sqrt(max(vol_mean[i], 0.0))),
            "std_resid" => _json(y[i] / sqrt(max(vol_mean[i], 1e-10)))
        ]
        for (qi, ql) in enumerate(q_levels)
            push!(row, "q$(qi)" => _json(sqrt(max(vol_q[i, qi], 0.0))))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Time Series Data
# =============================================================================

"""Convert a time series matrix to JSON data array with x + variable columns."""
function _timeseries_data_json(data::AbstractMatrix, varnames::Vector{String};
                               time_index::Union{AbstractVector,Nothing}=nothing)
    T_obs, n_vars = size(data)
    rows = Vector{Pair{String,String}}[]
    for t in 1:T_obs
        row = Pair{String,String}["x" => _json(time_index === nothing ? t : time_index[t])]
        for j in 1:n_vars
            push!(row, "v$j" => _json(data[t, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end
