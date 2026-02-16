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
plot_result methods for FEVD types: FEVD, BayesianFEVD, LPFEVD.
"""

# =============================================================================
# FEVD
# =============================================================================

"""
    plot_result(f::FEVD; var=nothing, ncols=0, title="", save_path=nothing)

Plot forecast error variance decomposition as stacked area charts.

`f.proportions` is n_vars × n_shocks × H.
"""
function plot_result(f::FEVD{T};
                     var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n_vars, n_shocks, H = size(f.proportions)
    shock_names = ["Shock $j" for j in 1:n_shocks]
    var_names = ["Variable $i" for i in 1:n_vars]

    vars_to_plot = var === nothing ? (1:n_vars) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("fevd")
        ptitle = var_names[vi]

        # proportions[vi, :, :] → transpose to H × n_shocks
        props = permutedims(f.proportions[vi, :, :])  # H × n_shocks
        data_json = _fevd_data_json(props, shock_names, H)
        s_json = _series_json(shock_names, _PLOT_COLORS[1:n_shocks];
                              keys=["s$j" for j in 1:n_shocks])

        js = _render_area_js(id, data_json, s_json;
                             xlabel="Horizon", ylabel="Proportion")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "Forecast Error Variance Decomposition"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BayesianFEVD
# =============================================================================

"""
    plot_result(f::BayesianFEVD; var=nothing, stat=:mean, ncols=0, title="", save_path=nothing)

Plot Bayesian FEVD (uses posterior mean by default).

`f.mean` is H × n_vars × n_shocks.
"""
function plot_result(f::BayesianFEVD{T};
                     var::Union{Int,String,Nothing}=nothing,
                     stat::Symbol=:mean, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = f.horizon
    n_vars = length(f.variables)
    n_shocks = length(f.shocks)

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, f.variables)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("bfevd")
        ptitle = f.variables[vi]

        # f.mean is H × n_vars × n_shocks → extract H × n_shocks for variable vi
        props = f.mean[1:H, vi, :]  # H × n_shocks
        # Normalize rows to sum to 1
        row_sums = sum(props, dims=2)
        props = props ./ max.(row_sums, eps(T))

        data_json = _fevd_data_json(props, f.shocks, H)
        s_json = _series_json(f.shocks, _PLOT_COLORS[1:n_shocks];
                              keys=["s$j" for j in 1:n_shocks])

        js = _render_area_js(id, data_json, s_json;
                             xlabel="Horizon", ylabel="Proportion")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "Bayesian FEVD (posterior mean)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LPFEVD
# =============================================================================

"""
    plot_result(f::LPFEVD; var=nothing, bias_corrected=true, ncols=0, title="", save_path=nothing)

Plot LP-FEVD (Gorodnichenko & Lee 2019).

`f.proportions` and `f.bias_corrected` are n_vars × n_shocks × H.
"""
function plot_result(f::LPFEVD{T};
                     var::Union{Int,Nothing}=nothing,
                     bias_corrected::Bool=true, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    source = bias_corrected && f.bias_correction ? f.bias_corrected : f.proportions
    n_vars, n_shocks, H = size(source)
    shock_names = ["Shock $j" for j in 1:n_shocks]
    var_names = ["Variable $i" for i in 1:n_vars]

    vars_to_plot = var === nothing ? (1:n_vars) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("lpfevd")
        ptitle = var_names[vi]

        props = permutedims(source[vi, :, :])  # H × n_shocks
        # Clamp and normalize
        props = max.(props, zero(T))
        row_sums = sum(props, dims=2)
        props = props ./ max.(row_sums, eps(T))

        data_json = _fevd_data_json(props, shock_names, H)
        s_json = _series_json(shock_names, _PLOT_COLORS[1:n_shocks];
                              keys=["s$j" for j in 1:n_shocks])

        js = _render_area_js(id, data_json, s_json;
                             xlabel="Horizon", ylabel="Proportion")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        bc_str = bias_corrected && f.bias_correction ? " (bias-corrected)" : ""
        title = "LP-FEVD$(bc_str)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
