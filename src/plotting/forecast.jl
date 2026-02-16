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
plot_result methods for forecast types: ARIMAForecast, VolatilityForecast,
VECMForecast, FactorForecast, LPForecast.
"""

# =============================================================================
# ARIMAForecast
# =============================================================================

"""
    plot_result(fc::ARIMAForecast; history=nothing, n_history=50, title="", save_path=nothing)

Plot ARIMA forecast with CI fan. Pass original series as `history` to show context.
"""
function plot_result(fc::ARIMAForecast{T};
                     history::Union{AbstractVector,Nothing}=nothing,
                     n_history::Int=50, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("arima_fc")
    data_json = _forecast_data_json(fc.forecast, fc.ci_lower, fc.ci_upper;
                                     history=history, n_history=n_history)

    series_keys = String[]
    series_names = String[]
    series_colors = String[]
    series_dash = String[]

    if history !== nothing
        push!(series_keys, "hist"); push!(series_names, "History")
        push!(series_colors, _PLOT_COLORS[1]); push!(series_dash, "")
    end
    push!(series_keys, "fc"); push!(series_names, "Forecast")
    push!(series_colors, _PLOT_COLORS[2]); push!(series_dash, "6,3")

    s_json = _series_json(series_names, series_colors; keys=series_keys, dash=series_dash)
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, xlabel="Period", ylabel="Value")

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "ARIMA Forecast (h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot([_PanelSpec(id, title, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VolatilityForecast
# =============================================================================

"""
    plot_result(fc::VolatilityForecast; history=nothing, n_history=50, title="", save_path=nothing)

Plot volatility forecast (conditional variance).
"""
function plot_result(fc::VolatilityForecast{T};
                     history::Union{AbstractVector,Nothing}=nothing,
                     n_history::Int=50, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("vol_fc")
    data_json = _forecast_data_json(fc.forecast, fc.ci_lower, fc.ci_upper;
                                     history=history, n_history=n_history)

    series_keys = String[]
    series_names = String[]
    series_colors = String[]
    series_dash = String[]

    if history !== nothing
        push!(series_keys, "hist"); push!(series_names, "History")
        push!(series_colors, _PLOT_COLORS[1]); push!(series_dash, "")
    end
    push!(series_keys, "fc"); push!(series_names, "Forecast σ²")
    push!(series_colors, _PLOT_COLORS[2]); push!(series_dash, "6,3")

    s_json = _series_json(series_names, series_colors; keys=series_keys, dash=series_dash)
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands, xlabel="Horizon", ylabel="Conditional Variance")

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "Volatility Forecast ($(fc.model_type), h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot([_PanelSpec(id, title, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VECMForecast
# =============================================================================

"""
    plot_result(fc::VECMForecast; var=nothing, ncols=0, title="", save_path=nothing)

Plot VECM forecast in levels with CI bands.
"""
function plot_result(fc::VECMForecast{T};
                     var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.levels)
    vars_to_plot = var === nothing ? (1:n_vars) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("vecm_fc")
        ptitle = "Variable $vi"

        data_json = _forecast_data_json(fc.levels[:, vi], fc.ci_lower[:, vi],
                                         fc.ci_upper[:, vi])

        s_json = _series_json(["Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
        bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel="Level")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "VECM Forecast ($(fc.ci_method))"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorForecast
# =============================================================================

"""
    plot_result(fc::FactorForecast; type=:factor, var=nothing, ncols=0, title="", save_path=nothing)

Plot factor model forecast.

- `type=:factor`: plot factor forecasts
- `type=:observable`: plot observable forecasts
"""
function plot_result(fc::FactorForecast{T};
                     type::Symbol=:factor, var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if type == :factor
        data = fc.factors
        lo = fc.factors_lower
        hi = fc.factors_upper
        label = "Factor"
    else
        data = fc.observables
        lo = fc.observables_lower
        hi = fc.observables_upper
        label = "Observable"
    end

    h, n_cols = size(data)
    vars_to_plot = var === nothing ? (1:min(n_cols, 6)) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("fac_fc")
        ptitle = "$label $vi"

        data_json = _forecast_data_json(data[:, vi], lo[:, vi], hi[:, vi])

        s_json = _series_json(["Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
        has_ci = fc.ci_method != :none
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel=label)
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "Factor Model Forecast ($(fc.ci_method))"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LPForecast
# =============================================================================

"""
    plot_result(fc::LPForecast; var=nothing, ncols=0, title="", save_path=nothing)

Plot LP direct multi-step forecast.
"""
function plot_result(fc::LPForecast{T};
                     var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_resp = size(fc.forecasts)
    vars_to_plot = var === nothing ? (1:n_resp) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("lp_fc")
        ptitle = "Response var $(fc.response_vars[vi])"

        data_json = _forecast_data_json(fc.forecasts[:, vi], fc.ci_lower[:, vi],
                                         fc.ci_upper[:, vi])

        s_json = _series_json(["LP Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
        bands = fc.ci_method != :none ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "LP Forecast (h=$(fc.horizon), $(ci_pct)% CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
