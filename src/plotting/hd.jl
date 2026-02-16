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
plot_result methods for Historical Decomposition types.
"""

# =============================================================================
# HistoricalDecomposition
# =============================================================================

"""
    plot_result(hd::HistoricalDecomposition; var=nothing, ncols=0, title="", save_path=nothing)

Plot historical decomposition: stacked bar of shock contributions + actual line.

- `var`: Variable index or name. `nothing` = all variables.
"""
function plot_result(hd::HistoricalDecomposition{T};
                     var::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    T_eff = hd.T_eff
    n_vars = length(hd.variables)
    n_shocks = length(hd.shock_names)

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, hd.variables)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        # Panel 1: Stacked bar of contributions
        id_bar = _next_plot_id("hd_bar")
        contributions = hd.contributions[:, vi, :]  # T_eff × n_shocks
        data_bar = _hd_data_json(contributions, hd.shock_names, T_eff)
        s_bar = _series_json(hd.shock_names, _PLOT_COLORS[1:n_shocks];
                             keys=["s$j" for j in 1:n_shocks])
        js_bar = _render_bar_js(id_bar, data_bar, s_bar;
                                mode="stacked", xlabel="Period",
                                ylabel="Contribution")
        push!(panels, _PanelSpec(id_bar, "$(hd.variables[vi]) — Shock Contributions", js_bar))

        # Panel 2: Actual vs sum of contributions
        id_line = _next_plot_id("hd_line")
        actual = hd.actual[:, vi]
        total = vec(sum(contributions, dims=2)) .+ hd.initial_conditions[:, vi]

        rows = Vector{Pair{String,String}}[]
        for t in 1:T_eff
            push!(rows, [
                "x" => _json(t),
                "actual" => _json(actual[t]),
                "recon" => _json(total[t])
            ])
        end
        data_line = _json_array_of_objects(rows)
        s_line = _series_json(["Actual", "Reconstructed"],
                              [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                              keys=["actual", "recon"],
                              dash=["", "6,3"])
        js_line = _render_line_js(id_line, data_line, s_line;
                                  xlabel="Period", ylabel="Value")
        push!(panels, _PanelSpec(id_line, "$(hd.variables[vi]) — Actual vs Decomposition", js_line))
    end

    if isempty(title)
        title = "Historical Decomposition ($(hd.method))"
    end
    if ncols <= 0
        ncols = 1
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BayesianHistoricalDecomposition
# =============================================================================

"""
    plot_result(hd::BayesianHistoricalDecomposition; var=nothing, stat=:mean, ncols=0, title="", save_path=nothing)

Plot Bayesian historical decomposition (uses posterior mean by default).
"""
function plot_result(hd::BayesianHistoricalDecomposition{T};
                     var::Union{Int,String,Nothing}=nothing,
                     stat::Symbol=:mean, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    T_eff = hd.T_eff
    n_vars = length(hd.variables)
    n_shocks = length(hd.shock_names)

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, hd.variables)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        # Stacked bar from posterior mean
        id_bar = _next_plot_id("bhd_bar")
        contributions = hd.mean[:, vi, :]  # T_eff × n_shocks
        data_bar = _hd_data_json(contributions, hd.shock_names, T_eff)
        s_bar = _series_json(hd.shock_names, _PLOT_COLORS[1:n_shocks];
                             keys=["s$j" for j in 1:n_shocks])
        js_bar = _render_bar_js(id_bar, data_bar, s_bar;
                                mode="stacked", xlabel="Period",
                                ylabel="Contribution")
        push!(panels, _PanelSpec(id_bar, "$(hd.variables[vi]) — Posterior Mean Contributions", js_bar))

        # Actual vs decomposition
        id_line = _next_plot_id("bhd_line")
        actual = hd.actual[:, vi]
        total = vec(sum(contributions, dims=2)) .+ hd.initial_mean[:, vi]

        rows = Vector{Pair{String,String}}[]
        for t in 1:T_eff
            push!(rows, [
                "x" => _json(t),
                "actual" => _json(actual[t]),
                "recon" => _json(total[t])
            ])
        end
        data_line = _json_array_of_objects(rows)
        s_line = _series_json(["Actual", "Reconstructed"],
                              [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                              keys=["actual", "recon"],
                              dash=["", "6,3"])
        js_line = _render_line_js(id_line, data_line, s_line;
                                  xlabel="Period", ylabel="Value")
        push!(panels, _PanelSpec(id_line, "$(hd.variables[vi]) — Actual vs Decomposition", js_line))
    end

    if isempty(title)
        title = "Bayesian Historical Decomposition ($(hd.method))"
    end
    if ncols <= 0
        ncols = 1
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
