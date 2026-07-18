# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
                     var::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n_vars, n_shocks, H = size(f.proportions)

    # C3: var accepts Int or variable name, bounds-checked via _resolve_var.
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, f.variables)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("fevd")
        ptitle = f.variables[vi]

        # proportions[vi, :, :] → transpose to H × n_shocks
        props = permutedims(f.proportions[vi, :, :])  # H × n_shocks
        data_json = _fevd_data_json(props, f.shocks, H)
        s_json = _series_json(f.shocks, _colors_for(f.shocks);
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
    plot_result(f::BayesianFEVD; var=nothing, shock=nothing, stat=:mean, ncols=0,
                title="", save_path=nothing)

Plot a Bayesian FEVD as nested posterior credible fans of each shock's contribution
share, one panel per `(variable, shock)` (PLT-28). Unlike the old point-only stacked
area, **all** quantile bands in `f.quantile_levels` render — the posterior uncertainty
that was previously discarded (audit M23). `stat` selects the central line:
- `:mean` (default) — `f.point_estimate` (H × n_vars × n_shocks);
- `:median` — the 0.5 quantile from `f.quantiles` (requires that level in
  `f.quantile_levels`, else an `ArgumentError`).
"""
function plot_result(f::BayesianFEVD{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     stat::Symbol=:mean, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    stat in (:mean, :median) ||
        throw(ArgumentError("stat must be :mean or :median, got :$stat"))
    H = f.horizon
    n_vars = length(f.variables)
    n_shocks = length(f.shocks)
    levels = f.quantile_levels
    xs = collect(1:H)

    qidx = 0
    if stat == :median
        qidx = something(findfirst(x -> isapprox(x, 0.5; atol=1e-8), levels), 0)
        qidx == 0 && throw(ArgumentError(
            "stat=:median requires the 0.5 quantile; available levels: $(levels)"))
    end
    central_label = stat === :median ? "Median" : "Mean"

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, f.variables)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, f.shocks)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        for si in shocks_to_plot
            ptitle = "$(f.variables[vi]) ← $(f.shocks[si])"
            qmat = f.quantiles[1:H, vi, si, :]                 # H×nq contribution share
            central = stat == :median ? f.quantiles[1:H, vi, si, qidx] :
                                        f.point_estimate[1:H, vi, si]
            panel, _ = _bayes_fan_panel("bfevd", ptitle, xs, qmat, levels,
                                        central, central_label, nothing, 0;
                                        xlabel="Horizon", ylabel="Variance share")
            push!(panels, panel)
        end
    end

    if isempty(title)
        stat_word = stat === :median ? "posterior median" : "posterior mean"
        title = "Bayesian FEVD ($stat_word share, credible bands)"
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
                     var::Union{Int,String,Nothing}=nothing,
                     bias_corrected::Bool=true, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    source = bias_corrected && f.bias_correction ? f.bias_corrected : f.proportions
    n_vars, n_shocks, H = size(source)

    # C3: var accepts Int or variable name, bounds-checked via _resolve_var.
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, f.variables)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        id = _next_plot_id("lpfevd")
        ptitle = f.variables[vi]

        props = permutedims(source[vi, :, :])  # H × n_shocks
        # Clamp and normalize
        props = max.(props, zero(T))
        row_sums = sum(props, dims=2)
        props = props ./ max.(row_sums, eps(T))

        data_json = _fevd_data_json(props, f.shocks, H)
        s_json = _series_json(f.shocks, _colors_for(f.shocks);
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
