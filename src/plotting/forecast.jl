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
# VARForecast
# =============================================================================

"""
    plot_result(fc::VARForecast; var=nothing, ncols=0, title="", save_path=nothing)

Plot VAR forecast with bootstrap CI bands.
"""
function plot_result(fc::VARForecast{T};
                     var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.forecast)
    vars_to_plot = var === nothing ? (1:n_vars) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("var_fc")
        ptitle = fc.varnames[vi]

        data_json = _forecast_data_json(fc.forecast[:, vi], fc.ci_lower[:, vi],
                                         fc.ci_upper[:, vi])

        s_json = _series_json(["Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
        has_ci = fc.ci_method != :none
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = fc.ci_method == :none ? "VAR Forecast" :
                "VAR Forecast ($(fc.ci_method) CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BVARForecast
# =============================================================================

"""
    plot_result(fc::BVARForecast; var=nothing, ncols=0, title="", save_path=nothing)

Plot Bayesian VAR forecast with posterior credible bands.
"""
function plot_result(fc::BVARForecast{T};
                     var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    h, n_vars = size(fc.forecast)
    vars_to_plot = var === nothing ? (1:n_vars) : [var]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("bvar_fc")
        ptitle = fc.varnames[vi]

        data_json = _forecast_data_json(fc.forecast[:, vi], fc.ci_lower[:, vi],
                                         fc.ci_upper[:, vi])

        pe_label = fc.point_estimate == :median ? "Posterior median" : "Posterior mean"
        s_json = _series_json([pe_label], [_PLOT_COLORS[1]]; keys=["fc"])
        bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel="Forecast")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * fc.conf_level)
        title = "Bayesian VAR Forecast ($(ci_pct)% credible interval)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
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
        ptitle = fc.varnames[vi]

        data_json = _forecast_data_json(fc.levels[:, vi], fc.ci_lower[:, vi],
                                         fc.ci_upper[:, vi])

        s_json = _series_json(["Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
        has_ci = fc.ci_method != :none
        bands = has_ci ?
            "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, xlabel="Horizon", ylabel="Level")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = fc.ci_method == :none ? "VECM Forecast" :
                "VECM Forecast ($(fc.ci_method) CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorForecast
# =============================================================================

"""
    plot_result(fc::FactorForecast; type=:both, var=nothing, ncols=0, title="",
                n_obs=6, save_path=nothing)

Plot factor model forecast.

- `type=:both`: plot factor forecasts and top observable forecasts (default)
- `type=:factor`: plot factor forecasts only
- `type=:observable`: plot observable forecasts only
- `n_obs`: max number of observables to show when `type=:both` (default 6)
"""
function plot_result(fc::FactorForecast{T};
                     type::Symbol=:both, var::Union{Int,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     n_obs::Int=6,
                     save_path::Union{String,Nothing}=nothing) where {T}
    has_ci = fc.ci_method != :none
    bands_str(color) = has_ci ?
        "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(color)\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"

    panels = _PanelSpec[]

    # Factor panels
    if type == :factor || type == :both
        h_f, n_factors = size(fc.factors)
        fvars = var !== nothing && type == :factor ? [var] : (1:n_factors)
        for vi in fvars
            id = _next_plot_id("fac_fc")
            ptitle = "Factor $vi"
            data_json = _forecast_data_json(fc.factors[:, vi], fc.factors_lower[:, vi],
                                             fc.factors_upper[:, vi])
            s_json = _series_json(["Forecast"], [_PLOT_COLORS[1]]; keys=["fc"])
            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands_str(_PLOT_COLORS[1]),
                                 xlabel="Horizon", ylabel="Factor")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    # Observable panels
    if type == :observable || type == :both
        h_o, n_obs_total = size(fc.observables)
        if var !== nothing && type == :observable
            ovars = [var]
        elseif type == :both
            ovars = 1:min(n_obs_total, n_obs)
        else
            ovars = 1:min(n_obs_total, 6)
        end
        for vi in ovars
            id = _next_plot_id("obs_fc")
            ptitle = "Observable $vi"
            data_json = _forecast_data_json(fc.observables[:, vi], fc.observables_lower[:, vi],
                                             fc.observables_upper[:, vi])
            s_json = _series_json(["Forecast"], [_PLOT_COLORS[2]]; keys=["fc"])
            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands_str(_PLOT_COLORS[2]),
                                 xlabel="Horizon", ylabel="Observable")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        ci_part = fc.ci_method == :none ? "" : " ($(fc.ci_method) CI)"
        title = "Factor Model Forecast$ci_part"
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
        ptitle = fc.varnames[fc.response_vars[vi]]

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
