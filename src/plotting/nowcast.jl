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
plot_result methods for nowcasting types: NowcastResult, NowcastNews.
"""

# =============================================================================
# NowcastResult
# =============================================================================

"""
    plot_result(nr::NowcastResult; ncols=0, title="", save_path=nothing)

Plot nowcast result: smoothed data for target variable with nowcast/forecast values.
"""
function plot_result(nr::NowcastResult{T};
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    T_obs, n_vars = size(nr.X_sm)
    n_show = min(n_vars, 6)

    panels = _PanelSpec[]

    # Target variable panel with nowcast/forecast extension
    ti = nr.target_index
    id = _next_plot_id("nc_target")
    ptitle = "Target (col $ti) — Nowcast: $(round(nr.nowcast, digits=3)), Forecast: $(round(nr.forecast, digits=3))"

    rows = Vector{Pair{String,String}}[]
    for t in 1:T_obs
        push!(rows, [
            "x" => _json(t),
            "v1" => _json(nr.X_sm[t, ti]),
            "v2" => _json(t == T_obs ? nr.X_sm[t, ti] : NaN)
        ])
    end
    # Extend with nowcast (T+1) and forecast (T+2) as separate series
    push!(rows, [
        "x" => _json(T_obs + 1),
        "v1" => "null",
        "v2" => _json(nr.nowcast)
    ])
    push!(rows, [
        "x" => _json(T_obs + 2),
        "v1" => "null",
        "v2" => _json(nr.forecast)
    ])
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Smoothed", "Nowcast/Forecast"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]]; keys=["v1", "v2"])
    js = _render_line_js(id, data_json, s_json; xlabel="Period", ylabel="Value")
    push!(panels, _PanelSpec(id, ptitle, js))

    # Additional variable panels
    other_vars = setdiff(1:min(n_vars, n_show + 1), [ti])
    for vi in other_vars[1:min(length(other_vars), n_show - 1)]
        id_v = _next_plot_id("nc_var")

        rows_v = Vector{Pair{String,String}}[]
        for t in 1:T_obs
            push!(rows_v, [
                "x" => _json(t),
                "v1" => _json(nr.X_sm[t, vi])
            ])
        end
        data_v = _json_array_of_objects(rows_v)
        s_v = _series_json(["Smoothed var $vi"], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                           keys=["v1"])
        js_v = _render_line_js(id_v, data_v, s_v; xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id_v, "Variable $vi", js_v))
    end

    if isempty(title)
        title = "Nowcast Result ($(nr.method))"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# NowcastNews
# =============================================================================

"""
    plot_result(nn::NowcastNews; title="", save_path=nothing)

Plot nowcast news decomposition: horizontal bar chart of per-release impact.
"""
function plot_result(nn::NowcastNews{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("news")
    n_releases = length(nn.impact_news)

    # Build bar data (use variable names as x labels)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n_releases
        label = i <= length(nn.variable_names) ? nn.variable_names[i] : "Release $i"
        push!(rows, [
            "x" => _json(label),
            "impact" => _json(nn.impact_news[i])
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["News Impact"], [_PLOT_COLORS[1]]; keys=["impact"])

    js = _render_bar_js(id, data_json, s_json; mode="grouped",
                        xlabel="", ylabel="Impact")

    delta = nn.new_nowcast - nn.old_nowcast
    if isempty(title)
        title = "Nowcast News: $(round(nn.old_nowcast, digits=3)) → $(round(nn.new_nowcast, digits=3)) (Δ = $(round(delta, digits=3)))"
    end

    p = _make_plot([_PanelSpec(id, "Per-Release Impact", js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end
