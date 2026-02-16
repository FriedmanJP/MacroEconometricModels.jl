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
plot_result methods for IRF types: ImpulseResponse, BayesianImpulseResponse,
LPImpulseResponse, StructuralLP.
"""

# =============================================================================
# ImpulseResponse
# =============================================================================

"""
    plot_result(r::ImpulseResponse; var=nothing, shock=nothing, ncols=0, title="", save_path=nothing)

Plot frequentist impulse response functions with confidence bands.

- `var`: Select response variable (index or name). `nothing` = all.
- `shock`: Select shock (index or name). `nothing` = all.
- `ncols`: Number of grid columns (0 = auto).
"""
function plot_result(r::ImpulseResponse{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = r.horizon
    n_vars = length(r.variables)
    n_shocks = length(r.shocks)

    # Determine which var/shock combinations to plot
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, r.variables)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, r.shocks)]

    panels = _PanelSpec[]
    for si in shocks_to_plot
        for vi in vars_to_plot
            id = _next_plot_id("irf")
            ptitle = "$(r.variables[vi]) ← $(r.shocks[si])"

            vals = r.values[1:H, vi, si]
            ci_lo = r.ci_lower[1:H, vi, si]
            ci_hi = r.ci_upper[1:H, vi, si]
            data_json = _irf_data_json(vals, ci_lo, ci_hi, H)

            s_json = _series_json(["IRF"], [_PLOT_COLORS[1]]; keys=["irf"])
            bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands, ref_lines_json=refs,
                                 xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        title = r.ci_type == :none ? "Impulse Response Functions" :
                "Impulse Response Functions ($(r.ci_type) CI)"
    end
    if ncols <= 0
        ncols = length(shocks_to_plot)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BayesianImpulseResponse
# =============================================================================

"""
    plot_result(r::BayesianImpulseResponse; var=nothing, shock=nothing, ncols=0, title="", save_path=nothing)

Plot Bayesian IRF with posterior mean and quantile bands.
"""
function plot_result(r::BayesianImpulseResponse{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = r.horizon
    n_vars = length(r.variables)
    n_shocks = length(r.shocks)
    nq = length(r.quantile_levels)

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, r.variables)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, r.shocks)]

    panels = _PanelSpec[]
    for si in shocks_to_plot
        for vi in vars_to_plot
            id = _next_plot_id("birf")
            ptitle = "$(r.variables[vi]) ← $(r.shocks[si])"

            # Use mean as main line; widest quantile pair as band
            vals = r.mean[1:H, vi, si]
            ci_lo = r.quantiles[1:H, vi, si, 1]      # lowest quantile
            ci_hi = r.quantiles[1:H, vi, si, nq]      # highest quantile
            data_json = _irf_data_json(vals, ci_lo, ci_hi, H)

            lo_q = round(Int, 100 * r.quantile_levels[1])
            hi_q = round(Int, 100 * r.quantile_levels[nq])
            s_json = _series_json(["Posterior mean"], [_PLOT_COLORS[1]]; keys=["irf"])
            bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands, ref_lines_json=refs,
                                 xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        lo_q = round(Int, 100 * r.quantile_levels[1])
        hi_q = round(Int, 100 * r.quantile_levels[nq])
        title = "Bayesian IRF ($(lo_q)%–$(hi_q)% posterior band)"
    end
    if ncols <= 0
        ncols = length(shocks_to_plot)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LPImpulseResponse
# =============================================================================

"""
    plot_result(r::LPImpulseResponse; var=nothing, ncols=0, title="", save_path=nothing)

Plot LP impulse responses with robust CI bands.
"""
function plot_result(r::LPImpulseResponse{T};
                     var::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = r.horizon + 1
    n_resp = length(r.response_vars)

    vars_to_plot = var === nothing ? (1:n_resp) : [_resolve_var(var, r.response_vars)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("lpirf")
        ptitle = "$(r.response_vars[vi]) ← $(r.shock_var)"

        vals = r.values[1:H, vi]
        ci_lo = r.ci_lower[1:H, vi]
        ci_hi = r.ci_upper[1:H, vi]
        data_json = _irf_data_json(vals, ci_lo, ci_hi, H)

        s_json = _series_json(["LP-IRF"], [_PLOT_COLORS[1]]; keys=["irf"])
        bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, ref_lines_json=refs,
                             xlabel="Horizon", ylabel="Response")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * r.conf_level)
        title = "LP Impulse Responses ($(r.cov_type), $(ci_pct)% CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# StructuralLP — delegates to ImpulseResponse
# =============================================================================

"""
    plot_result(slp::StructuralLP; kwargs...)

Plot structural LP impulse responses (delegates to ImpulseResponse plot).
"""
function plot_result(slp::StructuralLP{T}; title::String="", kwargs...) where {T}
    if isempty(title)
        title = "Structural LP IRF ($(slp.method))"
    end
    plot_result(slp.irf; title=title, kwargs...)
end
