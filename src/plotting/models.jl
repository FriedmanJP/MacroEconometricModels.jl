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
plot_result methods for model types: ARCH/GARCH/EGARCH/GJR/SV,
FactorModel, DynamicFactorModel, TimeSeriesData, PanelData.
"""

# =============================================================================
# Volatility Model Diagnostics (shared)
# =============================================================================

"""
Generate 3-panel volatility diagnostic Figure:
1. Returns
2. Conditional volatility (σₜ)
3. Standardized residuals (with ±2σ reference lines)
"""
function _plot_volatility_diagnostics(y::AbstractVector, cond_var::AbstractVector,
                                      model_name::String; title::String="",
                                      save_path::Union{String,Nothing}=nothing)
    data_json = _volatility_data_json(y, cond_var)

    # Panel 1: Returns
    id1 = _next_plot_id("vol_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Conditional volatility
    id2 = _next_plot_id("vol_sig")
    s2 = _series_json(["Cond. Volatility (σ)"], [_PLOT_COLORS[2]]; keys=["vol"])
    js2 = _render_line_js(id2, data_json, s2; xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Conditional Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("vol_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period",
                          ylabel="zₜ = εₜ/σₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "$model_name — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ARCHModel
# =============================================================================

"""
    plot_result(m::ARCHModel; title="", save_path=nothing)

Plot ARCH model diagnostics: returns, conditional volatility, standardized residuals.
"""
function plot_result(m::ARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "ARCH($(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GARCHModel
# =============================================================================

"""
    plot_result(m::GARCHModel; title="", save_path=nothing)

Plot GARCH model diagnostics.
"""
function plot_result(m::GARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# EGARCHModel
# =============================================================================

"""
    plot_result(m::EGARCHModel; title="", save_path=nothing)

Plot EGARCH model diagnostics.
"""
function plot_result(m::EGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "EGARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GJRGARCHModel
# =============================================================================

"""
    plot_result(m::GJRGARCHModel; title="", save_path=nothing)

Plot GJR-GARCH model diagnostics.
"""
function plot_result(m::GJRGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GJR-GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# SVModel
# =============================================================================

"""
    plot_result(m::SVModel; title="", save_path=nothing)

Plot SV model diagnostics with posterior quantile bands on volatility.
"""
function plot_result(m::SVModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    data_json = _sv_data_json(m.y, m.volatility_mean, m.volatility_quantiles,
                               m.quantile_levels)
    nq = length(m.quantile_levels)

    # Panel 1: Returns
    id1 = _next_plot_id("sv_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Posterior volatility with CI band
    id2 = _next_plot_id("sv_vol")
    s2 = _series_json(["Posterior mean σ"], [_PLOT_COLORS[2]]; keys=["vol_mean"])
    bands2 = nq >= 2 ?
        "[{\"lo_key\":\"q1\",\"hi_key\":\"q$(nq)\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
    js2 = _render_line_js(id2, data_json, s2;
                          bands_json=bands2, xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Stochastic Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("sv_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period", ylabel="zₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "Stochastic Volatility Model — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorModel
# =============================================================================

"""
    plot_result(fm::FactorModel; title="", save_path=nothing)

Plot factor model: scree plot (eigenvalues) + extracted factor series.
"""
function plot_result(fm::FactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree plot (eigenvalues as bar chart)
    id1 = _next_plot_id("fm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("fm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _PLOT_COLORS[1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors", js2)

    if isempty(title)
        title = "Static Factor Model (r=$(fm.r))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DynamicFactorModel
# =============================================================================

"""
    plot_result(fm::DynamicFactorModel; title="", save_path=nothing)

Plot dynamic factor model: scree + factor series.
"""
function plot_result(fm::DynamicFactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree
    id1 = _next_plot_id("dfm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("dfm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, _PLOT_COLORS[1:n_plot]; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors (VAR($( fm.p)))", js2)

    if isempty(title)
        title = "Dynamic Factor Model (r=$(fm.r), p=$(fm.p))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TimeSeriesData
# =============================================================================

"""
    plot_result(d::TimeSeriesData; vars=nothing, ncols=0, title="", save_path=nothing)

Plot time series data: one panel per variable (capped at 12).

- `vars`: Variable indices or names to plot. `nothing` = all (up to 12).
"""
function plot_result(d::TimeSeriesData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        idxs = collect(1:min(d.n_vars, 12))
    else
        idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    panels = _PanelSpec[]
    for vi in idxs
        id = _next_plot_id("ts")
        ptitle = d.varnames[vi]

        rows = Vector{Pair{String,String}}[]
        for t in 1:d.T_obs
            push!(rows, [
                "x" => _json(d.time_index[t]),
                "v1" => _json(d.data[t, vi])
            ])
        end
        data_json = _json_array_of_objects(rows)
        s_json = _series_json([d.varnames[vi]], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                              keys=["v1"])

        js = _render_line_js(id, data_json, s_json; xlabel="Time", ylabel="")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Time Series Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# PanelData
# =============================================================================

"""
    plot_result(d::PanelData; vars=nothing, ncols=0, title="", save_path=nothing)

Plot panel data: one panel per variable, with lines for each group.
"""
function plot_result(d::PanelData{T};
                     vars::Union{Vector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if vars === nothing
        var_idxs = collect(1:min(d.n_vars, 6))
    else
        var_idxs = [v isa String ? _resolve_var(v, d.varnames) : v for v in vars]
    end

    # Group data by group_id
    unique_groups = sort(unique(d.group_id))
    n_groups_plot = min(length(unique_groups), 10)
    groups_to_plot = unique_groups[1:n_groups_plot]

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("pd")
        ptitle = d.varnames[vi]

        # Build data with one column per group
        # Get unique time points
        unique_times = sort(unique(d.time_id))
        rows = Vector{Pair{String,String}}[]
        for t in unique_times
            row = Pair{String,String}["x" => _json(t)]
            for (gi, gid) in enumerate(groups_to_plot)
                mask = (d.group_id .== gid) .& (d.time_id .== t)
                idx = findfirst(mask)
                val = idx !== nothing ? d.data[idx, vi] : NaN
                push!(row, "g$gi" => _json(val))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)

        group_names = [gi <= length(d.group_names) ? d.group_names[gi] : "Group $gid"
                       for (gi, gid) in enumerate(groups_to_plot)]
        s_json = _series_json(group_names, _PLOT_COLORS[1:n_groups_plot];
                              keys=["g$i" for i in 1:n_groups_plot])

        js = _render_line_js(id, data_json, s_json; xlabel="Time", ylabel="")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = isempty(desc(d)) ? "Panel Data" : desc(d)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
