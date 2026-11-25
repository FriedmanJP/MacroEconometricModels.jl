# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
plot_result methods for IRF types: ImpulseResponse, BayesianImpulseResponse,
LPImpulseResponse, StructuralLP.
"""

# =============================================================================
# ImpulseResponse
# =============================================================================

"""
    plot_result(r::ImpulseResponse; var=nothing, shock=nothing,
                ncols=0, title="", save_path=nothing)

Plot frequentist impulse response functions with confidence bands.

- `var`: Select response variable (index or name). `nothing` = all.
- `shock`: Select shock (index or name). `nothing` = all.
- `ncols`: Number of grid columns (0 = auto).
"""
function plot_result(r::ImpulseResponse{T};
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = r.horizon
    n_vars = length(r.variables)
    n_shocks = length(r.shocks)

    # Determine which var/shock combinations to plot
    vars_to_plot = if var === nothing
        collect(1:n_vars)
    elseif var isa AbstractVector
        Int[_resolve_var(v, r.variables) for v in var]
    else
        [_resolve_var(var, r.variables)]
    end
    shocks_to_plot = if shock === nothing
        collect(1:n_shocks)
    elseif shock isa AbstractVector
        Int[_resolve_var(s, r.shocks) for s in shock]
    else
        [_resolve_var(shock, r.shocks)]
    end

    panels = _PanelSpec[]
    for vi in vars_to_plot
        for si in shocks_to_plot
            id = _next_plot_id("irf")
            ptitle = "$(r.variables[vi]) ← $(r.shocks[si])"

            vals = r.values[1:H, vi, si]
            ci_lo = r.ci_lower[1:H, vi, si]
            ci_hi = r.ci_upper[1:H, vi, si]
            data_json = _irf_data_json(vals, ci_lo, ci_hi, H)

            s_json = _series_json(["IRF"], [_PLOT_COLORS[1]]; keys=["irf"])
            has_ci = r.ci_type != :none
            bands = has_ci ?
                "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\"," *
                "\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands, ref_lines_json=refs, integer_x=true,
                                 xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        title = r.ci_type == :none ? "Impulse Response Functions" :
                "Impulse Response Functions ($(r.ci_type) CI)"
    end
    if ncols <= 0
        ncols = length(shocks_to_plot) == 1 ?
            min(length(vars_to_plot), 3) : length(shocks_to_plot)
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    return p
end

# =============================================================================
# BayesianImpulseResponse
# =============================================================================

"""
    plot_result(r::BayesianImpulseResponse; var=nothing, shock=nothing, stat=:median,
                draws=0, ncols=0, title="", save_path=nothing)

Plot a Bayesian IRF as nested posterior credible fans (PLT-28). **All** quantile bands
in `r.quantile_levels` render — one legend-labelled band per symmetric quantile pair —
rather than only the outermost pair (audit M23). `stat` selects the central line
(`:median` = `r.point_estimate`, `:mean` = mean of `r._draws` when available); `draws>0`
overlays up to 200 subsampled posterior IRF paths from `r._draws` (C7).
"""
function plot_result(r::BayesianImpulseResponse{T};
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     stat::Symbol=:median, draws::Int=0,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    stat in (:median, :mean) ||
        throw(ArgumentError("stat must be :median or :mean, got :$stat"))
    H = r.horizon
    n_vars = length(r.variables)
    n_shocks = length(r.shocks)
    levels = r.quantile_levels
    xs = collect(0:H-1)

    vars_to_plot = if var === nothing
        collect(1:n_vars)
    elseif var isa AbstractVector
        Int[_resolve_var(v, r.variables) for v in var]
    else
        [_resolve_var(var, r.variables)]
    end
    shocks_to_plot = if shock === nothing
        collect(1:n_shocks)
    elseif shock isa AbstractVector
        Int[_resolve_var(s, r.shocks) for s in shock]
    else
        [_resolve_var(shock, r.shocks)]
    end
    central_label = stat === :mean ? "Mean" : "Median"
    has_draws = r._draws !== nothing

    panels = _PanelSpec[]
    note = ""
    for vi in vars_to_plot
        for si in shocks_to_plot
            ptitle = "$(r.variables[vi]) ← $(r.shocks[si])"
            qmat = r.quantiles[1:H, vi, si, :]                 # H×nq
            central = if stat === :mean && has_draws
                vec(sum(@view(r._draws[:, 1:H, vi, si]), dims=1)) ./
                    max(size(r._draws, 1), 1)
            else
                _median_from_quantiles(qmat, levels, r.point_estimate[1:H, vi, si])
            end
            draw_paths = (draws > 0 && has_draws) ? r._draws[:, 1:H, vi, si] : nothing
            panel, n = _bayes_fan_panel("birf", ptitle, xs, qmat, levels,
                                        central, central_label, draw_paths, draws;
                                        xlabel="Horizon", ylabel="Response")
            push!(panels, panel)
            isempty(note) && (note = n)
        end
    end

    if isempty(title)
        lo_q = round(Int, 100 * levels[1])
        hi_q = round(Int, 100 * levels[end])
        title = "Bayesian IRF ($(central_label), $(lo_q)%–$(hi_q)% posterior bands)"
    end
    if ncols <= 0
        ncols = length(shocks_to_plot) == 1 ?
            min(length(vars_to_plot), 3) : length(shocks_to_plot)
    end

    p = _make_plot(panels; title=title, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    return p
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
        bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\"," *
            "\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, ref_lines_json=refs, integer_x=true,
                             xlabel="Horizon", ylabel="Response")
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        ci_pct = round(Int, 100 * r.conf_level)
        title = "LP Impulse Responses ($(r.cov_type), $(ci_pct)% CI)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    return p
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
    return plot_result(slp.irf; title=title, kwargs...)
end

# =============================================================================
# QuantileImpulseResponse
# =============================================================================

"""
    plot_result(qirf_res::QuantileImpulseResponse; var=nothing, shock=nothing,
                max_vars=12, ncols=0, title="", save_path=nothing)

Plot quantile impulse response functions across horizons `h = 0, …, H - 1`
for selected response variables and structural shocks, overlaid with
confidence or credible interval bands when `qirf_res.ci_type != :none`.

# Keyword Arguments
- `var::Union{Int,String,AbstractVector,Nothing} = nothing` — response variable index, name, or vector (`nothing` plots up to `max_vars`)
- `shock::Union{Int,String,AbstractVector,Nothing} = nothing` — shock index, name, or vector (`nothing` plots all shocks)
- `max_vars::Int = 12` — maximum response variables plotted when `var === nothing` (C7 cap note emitted if truncated)
- `ncols::Int = 0` — number of grid columns (`0` = auto)
- `title::String = ""` — custom figure title (`""` = auto-generated)
- `save_path::Union{String,Nothing} = nothing` — optional path to save standalone HTML
"""
function plot_result(qirf_res::QuantileImpulseResponse{T};
                     var::Union{Int,String,AbstractVector,Nothing}=nothing,
                     shock::Union{Int,String,AbstractVector,Nothing}=nothing,
                     max_vars::Int=12,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = qirf_res.horizon
    n_vars = length(qirf_res.variables)
    n_shocks = length(qirf_res.shocks)

    vars_to_plot = if var === nothing
        collect(1:min(n_vars, max(1, max_vars)))
    elseif var isa AbstractVector
        Int[_resolve_var(v, qirf_res.variables) for v in var]
    else
        [_resolve_var(var, qirf_res.variables)]
    end
    shocks_to_plot = if shock === nothing
        collect(1:n_shocks)
    elseif shock isa AbstractVector
        Int[_resolve_var(s, qirf_res.shocks) for s in shock]
    else
        [_resolve_var(shock, qirf_res.shocks)]
    end
    (!isempty(vars_to_plot) && !isempty(shocks_to_plot)) ||
        throw(ArgumentError("Must select at least one variable and one shock to plot"))

    has_ci = qirf_res.ci_type != :none
    tp = qirf_res.tau_profile
    tp_label = isempty(tp) ? "τ=$(_fmt(qirf_res.baseline_tau; digits=2))" :
               (length(unique(round.(Float64.(tp); digits=6))) == 1 ?
                "τ=$(_fmt(tp[1]; digits=2))" :
                "τ∈[$(_fmt(minimum(tp); digits=2)),$(_fmt(maximum(tp); digits=2))]")

    panels = _PanelSpec[]
    for vi in vars_to_plot
        for si in shocks_to_plot
            id = _next_plot_id("qirf")
            ptitle = "$(qirf_res.variables[vi]) ← $(qirf_res.shocks[si])"
            vals = qirf_res.values[1:H, vi, si]
            ci_lo = qirf_res.ci_lower[1:H, vi, si]
            ci_hi = qirf_res.ci_upper[1:H, vi, si]
            data_json = _irf_data_json(vals, ci_lo, ci_hi, H)
            s_json = _series_json(["QIRF ($(tp_label))"], [_PLOT_SERIES[1]]; keys=["irf"])
            bands = has_ci ?
                "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\"," *
                "\"color\":\"$(_PLOT_SERIES[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
            refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
            js = _render_line_js(id, data_json, s_json;
                                 bands_json=bands, ref_lines_json=refs, integer_x=true,
                                 xlabel="Horizon", ylabel="Response")
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        ci_part = has_ci ? ", $(qirf_res.ci_type) $(round(Int, 100 * qirf_res.conf_level))% CI" : ""
        title = "Quantile Impulse Responses ($(tp_label), $(qirf_res.method)$(ci_part))"
    end
    if ncols <= 0
        ncols = length(shocks_to_plot) == 1 ?
            min(length(vars_to_plot), 3) : length(shocks_to_plot)
    end
    note = var === nothing ? _cap_note("response variables", length(vars_to_plot), n_vars, "max_vars") : ""

    p = _make_plot(panels; title=title, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    return p
end
