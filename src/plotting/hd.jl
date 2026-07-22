# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
        s_bar = _series_json(hd.shock_names, _colors_for(hd.shock_names);
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
                                  ref_lines_json="[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]",
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
    plot_result(hd::BayesianHistoricalDecomposition; var=nothing, shock=nothing,
                stat=:mean, ncols=0, title="", save_path=nothing)

Plot a Bayesian historical decomposition as nested posterior credible fans of each
shock's contribution, one panel per `(variable, shock)` (PLT-28). Unlike the old
point-only stacked bar, **all** quantile bands in `hd.quantile_levels` render — the
posterior uncertainty that was previously discarded (audit M23). `stat` selects the
central line:
- `:mean` (default) — `hd.point_estimate`;
- `:median` — the 0.5 quantile from `hd.quantiles` (requires that level in
  `hd.quantile_levels`, else an `ArgumentError`).
"""
function plot_result(hd::BayesianHistoricalDecomposition{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     stat::Symbol=:mean, ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    stat in (:mean, :median) ||
        throw(ArgumentError("stat must be :mean or :median, got :$stat"))
    T_eff = hd.T_eff
    n_vars = length(hd.variables)
    n_shocks = length(hd.shock_names)
    levels = hd.quantile_levels
    xs = collect(1:T_eff)

    qidx = 0
    if stat == :median
        qidx = something(findfirst(x -> isapprox(x, 0.5; atol=1e-8), levels), 0)
        qidx == 0 && throw(ArgumentError(
            "stat=:median requires the 0.5 quantile; available levels: $(levels)"))
    end
    central_label = stat == :median ? "Median" : "Mean"

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, hd.variables)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, hd.shock_names)]

    panels = _PanelSpec[]
    for vi in vars_to_plot
        for si in shocks_to_plot
            ptitle = "$(hd.variables[vi]) ← $(hd.shock_names[si])"
            qmat = hd.quantiles[1:T_eff, vi, si, :]            # T_eff×nq contribution
            central = stat == :median ? hd.quantiles[1:T_eff, vi, si, qidx] :
                                        hd.point_estimate[1:T_eff, vi, si]
            panel, _ = _bayes_fan_panel("bhd", ptitle, xs, qmat, levels,
                                        central, central_label, nothing, 0;
                                        xlabel="Period", ylabel="Contribution")
            push!(panels, panel)
        end
    end

    if isempty(title)
        stat_word = stat === :median ? "posterior median" : "posterior mean"
        title = "Bayesian Historical Decomposition ($(hd.method), $stat_word)"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
