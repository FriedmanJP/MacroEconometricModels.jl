# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# HA-DSGE dynamics (PLT-34, #496)
#
# Plot the *dynamic* heterogeneous-agent outputs that `HASteadyState` cannot show:
#   • distribution IRFs       — how the wealth histogram deviates from SS over time
#   • inequality IRFs         — Gini and wealth-percentile response paths
#   • Krusell-Smith PLM fit    — perceived-law-of-motion R² per aggregate
#   • Den Haan accuracy        — reference vs PLM-only aggregate simulation
#
# All statistics already live in `src/dsge/heterogeneous/analysis.jl`
# (`distribution_irf`, `inequality_irf`) and `krusell_smith.jl` (`den_haan_test`);
# this file is the view-route selection + assembly layer only (plotrule A5). No new
# return type is introduced — the plot dispatches directly on `HADSGESolution`,
# `KrusellSmithSolution`, and `DenHaanAccuracy`, reaching the grid geometry through
# `sol.steady_state.grid`.
# =============================================================================

# -----------------------------------------------------------------------------
# Lane-local converter (A5 — documented `_plot_*` helper directly above the method;
# lives here, NOT in the frozen helpers.jl).
# -----------------------------------------------------------------------------

"""
    _ha_bin_rows(mat, a_grid, max_bins) -> (labels::Vector{String}, binned::Matrix)

Aggregate the ROW dimension (asset nodes) of `mat` (`n_a × H`) into at most
`max_bins` contiguous bins by **summing** every node's value into its bin — no node
is ever dropped, so each bin's column totals equal the exact sum of the node values
in that bin (plotrule: distributions-over-a-grid must conserve mass, never
point-sample; Anti-Pattern #7). Bins use the same edge scheme as the static
`_plot_ha_distribution`. Each bin is labelled by the asset value at its right edge.
Returns the bin labels (ascending asset order) and the `nbins × H` binned matrix.
"""
function _ha_bin_rows(mat::AbstractMatrix, a_grid::AbstractVector, max_bins::Int)
    n_a, H = size(mat)
    nbins = min(n_a, max(1, max_bins))
    edges = round.(Int, range(0, n_a; length=nbins + 1))
    labels = String[]
    binned = Matrix{Float64}(undef, 0, H)
    for b in 1:nbins
        lo = edges[b] + 1
        hi = edges[b + 1]
        hi < lo && continue
        row = Float64[sum(@view mat[lo:hi, h]) for h in 1:H]
        binned = vcat(binned, reshape(row, 1, H))
        push!(labels, _fmt_grid_label(a_grid[hi]))
    end
    labels, binned
end

# =============================================================================
# HADSGESolution — distribution & inequality IRFs (view route)
# =============================================================================

"""
    plot_result(sol::HADSGESolution; view=:distribution_dynamics, horizon=40,
                shock_index=1, shock_size=1.0, max_bins=40, title="", save_path=nothing)

Plot the dynamic response of a linearized heterogeneous-agent DSGE solution to an
aggregate shock. Calls the existing analysis functions (`distribution_irf`,
`inequality_irf`) and assembles the result (plotrule A5).

# Views
- `:distribution_dynamics` (default, alias `:distribution`) — the wealth-histogram
  response marginalized over income states, drawn as an **asset × horizon heatmap**
  of deviations from steady state. Signed deviations ⇒ a diverging color scale
  centred at zero, with a color-scale legend (plotrule Heatmaps). The asset axis is
  bin-aggregated to at most `max_bins` bins so no grid node's mass is dropped.
- `:inequality` — two stacked panels: the Gini coefficient response (top) and the
  `p10…p90` wealth-percentile responses (bottom, a legended multi-line panel).

`method=:ssj` has no distribution basis, so both views re-raise the informative
"use method=:reiter" error that `distribution_irf` raises (plotrule C5). An unknown
`view` throws an `ArgumentError` listing the valid views.

# Keyword arguments
- `horizon::Int=40` — number of periods to track after the shock.
- `shock_index::Int=1` / `shock_size::Real=1.0` — which aggregate shock and its size.
- `max_bins::Int=40` — asset-axis bin cap for the distribution heatmap (C7-surfaced).
"""
function plot_result(sol::HADSGESolution{T};
                     view::Symbol=:distribution_dynamics,
                     horizon::Int=40, shock_index::Int=1, shock_size::Real=1.0,
                     max_bins::Int=40, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :distribution_dynamics || view === :distribution
        p = _plot_ha_dist_dynamics(sol; horizon=horizon, shock_index=shock_index,
                                   shock_size=shock_size, max_bins=max_bins, title=title)
    elseif view === :inequality
        p = _plot_ha_inequality(sol; horizon=horizon, shock_index=shock_index,
                                shock_size=shock_size, title=title)
    else
        throw(ArgumentError("Unknown view: :$view. Use :distribution_dynamics " *
            "(alias :distribution) or :inequality."))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""Asset × horizon heatmap of distribution deviations (marginalized over income)."""
function _plot_ha_dist_dynamics(sol::HADSGESolution{T}; horizon::Int, shock_index::Int,
                                shock_size::Real, max_bins::Int, title::String) where {T}
    # `distribution_irf` itself re-raises for :ssj; guard here too so the view route
    # errors before building anything (plotrule C5, same message).
    sol.method === :ssj && error("distribution IRFs are unavailable for method=:ssj " *
        "(the Ho-Kalman realization has no distribution basis); use method=:reiter.")

    dirf = distribution_irf(sol, horizon; shock_index=shock_index, shock_size=shock_size)
    n_a, n_e, H = size(dirf)
    # Marginalize over income states → (n_a × H) signed deviations.
    marg = dropdims(sum(dirf; dims=2); dims=2)

    a_grid = sol.steady_state.grid.grids[1]
    labels_asc, binned = _ha_bin_rows(marg, a_grid, max_bins)   # (nbins × H)
    nbins = length(labels_asc)
    row_labels = reverse(labels_asc)                            # high asset at top
    col_labels = String[string(h - 1) for h in 1:H]             # horizon 0..H-1

    rows = Vector{Pair{String,String}}[]
    for b in 1:nbins, h in 1:H
        push!(rows, ["x" => _json(col_labels[h]), "y" => _json(labels_asc[b]),
                     "v" => _json(binned[b, h])])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ha_distdyn")
    js = _render_heatmap_js(id, data_json, _json(row_labels), _json(col_labels);
                            xlabel="Horizon", ylabel="Assets", tip_label="Horizon",
                            scale=:diverging, midpoint=0)
    ptitle = _cap_title("Distribution response (deviation from SS)", nbins, n_a)
    p1 = _PanelSpec(id, ptitle, js)

    isempty(title) && (title = "Wealth Distribution Dynamics (shock $(shock_index), " *
        "$(_fmt(float(shock_size); digits=2))σ)")
    _make_plot([p1]; title=title, ncols=1)
end

"""Gini (top panel) + p10…p90 wealth-percentile (bottom panel) response paths."""
function _plot_ha_inequality(sol::HADSGESolution{T}; horizon::Int, shock_index::Int,
                             shock_size::Real, title::String) where {T}
    sol.method === :ssj && error("distribution IRFs are unavailable for method=:ssj " *
        "(the Ho-Kalman realization has no distribution basis); use method=:reiter.")

    ineq = inequality_irf(sol, horizon; shock_index=shock_index, shock_size=shock_size)
    H = length(ineq[:gini])
    xs = collect(0:(H - 1))                                     # IRF horizon starts at 0
    panels = _PanelSpec[]

    # Panel 1 — Gini response (single line, no legend).
    id1 = _next_plot_id("ha_gini")
    rows1 = Vector{Pair{String,String}}[]
    for h in 1:H
        push!(rows1, ["x" => _json(xs[h]), "gini" => _json(ineq[:gini][h])])
    end
    s1 = _series_json(["Gini"], [_PLOT_SERIES[1]]; keys=["gini"])
    js1 = _render_line_js(id1, _json_array_of_objects(rows1), s1;
                          integer_x=true, xlabel="Horizon", ylabel="Gini coefficient")
    push!(panels, _PanelSpec(id1, "Gini Coefficient Response", js1))

    # Panel 2 — p10..p90 percentile responses (legended multi-line).
    pkeys = [:p10, :p25, :p50, :p75, :p90]
    pnames = ["p10", "p25", "p50", "p75", "p90"]
    id2 = _next_plot_id("ha_pctl")
    rows2 = Vector{Pair{String,String}}[]
    for h in 1:H
        row = Pair{String,String}["x" => _json(xs[h])]
        for (j, k) in enumerate(pkeys)
            push!(row, "p$j" => _json(ineq[k][h]))
        end
        push!(rows2, row)
    end
    s2 = _series_json(pnames, _colors_for(pnames); keys=["p$j" for j in 1:length(pkeys)])
    js2 = _render_line_js(id2, _json_array_of_objects(rows2), s2;
                          integer_x=true, xlabel="Horizon", ylabel="Wealth percentile")
    push!(panels, _PanelSpec(id2, "Wealth Percentile Responses", js2))

    isempty(title) && (title = "Inequality Dynamics (shock $(shock_index), " *
        "$(_fmt(float(shock_size); digits=2))σ)")
    _make_plot(panels; title=title, ncols=1)
end

# =============================================================================
# KrusellSmithSolution — perceived-law-of-motion R² fit
# =============================================================================

"""
    plot_result(ks::KrusellSmithSolution; max_aggregates=20, title="", save_path=nothing)

Horizontal bar of the Krusell-Smith perceived-law-of-motion R² (0–1) per aggregate,
with the fitted PLM coefficients annotated in the panel title (plotrule form rule:
R² per *named* aggregate reads as horizontal named-entity bars). At most
`max_aggregates` aggregates are shown; any excess is surfaced in the panel title (C7).
"""
function plot_result(ks::KrusellSmithSolution{T};
                     max_aggregates::Int=20, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    aggs = sort(collect(keys(ks.r_squared)))
    n_total = length(aggs)
    shown = min(n_total, max(1, max_aggregates))
    aggs = aggs[1:shown]

    rows = Vector{Pair{String,String}}[]
    for a in aggs
        push!(rows, ["x" => _json(string(a)), "r2" => _json(ks.r_squared[a])])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ks_plm")
    s_json = _series_json(["R²"], [_PLOT_SERIES[1]]; keys=["r2"])
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                        xlabel="R²", ylabel="Aggregate")

    # Coefficient annotation (rounded, plotrule C9) — visible in the panel title.
    coef_bits = String[]
    for a in aggs
        haskey(ks.plm_coefficients, a) || continue
        cs = join([_fmt(float(c); digits=3) for c in ks.plm_coefficients[a]], ", ")
        push!(coef_bits, "$(a): [$(cs)]")
    end
    ptitle = isempty(coef_bits) ? "PLM R² by aggregate" :
        "PLM R² by aggregate — " * join(coef_bits, "; ")
    ptitle = _cap_title(ptitle, shown, n_total)
    p1 = _PanelSpec(id, ptitle, js)

    isempty(title) && (title = "Krusell-Smith PLM Accuracy")
    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DenHaanAccuracy — reference vs PLM-only aggregate simulation
# =============================================================================

"""
    plot_result(dh::DenHaanAccuracy; max_points=2000, title="", save_path=nothing)

Overlay the Den Haan (2010) accuracy paths: the reference explicit cross-sectional
simulation of the aggregate against the PLM-only path iterated on its own forecasts.
The burn-in (`1:T_burn`) is dropped. The two lines are distinguished by **both**
color and dash (solid reference vs dashed PLM-only; plotrule Color). The maximum and
mean errors are rounded into the panel title (plotrule C9). Paths longer than
`max_points` are strided down (endpoints kept), surfaced in the panel title (C7).
"""
function plot_result(dh::DenHaanAccuracy{T};
                     max_points::Int=2000, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = min(length(dh.ref_path), length(dh.plm_path))
    start = (dh.T_burn > 0 && dh.T_burn < n) ? dh.T_burn + 1 : 1
    idx = collect(start:n)
    total = length(idx)
    stride = max(1, cld(total, max(1, max_points)))
    if stride > 1
        sel = collect(start:stride:n)
        (isempty(sel) || last(sel) != n) && push!(sel, n)
        idx = sel
    end

    rows = Vector{Pair{String,String}}[]
    for t in idx
        push!(rows, ["x" => _json(t), "ref" => _json(dh.ref_path[t]),
                     "plm" => _json(dh.plm_path[t])])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("den_haan")
    s_json = _series_json(["Reference", "PLM-only"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                          keys=["ref", "plm"], dash=["", "6,3"])
    js = _render_line_js(id, data_json, s_json;
                         xlabel="Period", ylabel=string(dh.aggregate))
    ptitle = "Den Haan: max err = $(_fmt(float(dh.dh_max); digits=3)), " *
             "mean = $(_fmt(float(dh.dh_mean); digits=3))"
    ptitle = _cap_title(ptitle, length(idx), total)
    p1 = _PanelSpec(id, ptitle, js)

    isempty(title) && (title = "Den Haan (2010) Accuracy — $(dh.aggregate)")
    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
