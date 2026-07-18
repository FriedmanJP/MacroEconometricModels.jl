# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
PLT-28 — Bayesian nested-fan band upgrades. Routes the Bayesian IRF/FEVD/HD results
(and the new `BayesianDSGESimulation`) through the frozen `_render_fan_js` primitive so
**every** posterior quantile band renders (audit M23 — the old plots drew only the
outermost pair, or, for FEVD/HD, discarded the quantiles entirely). The central line
honours a live `stat=:median/:mean` kwarg (C4); an opt-in `draws=` overlays up to 200
subsampled posterior paths (C7). All quantile-column bookkeeping is done by the shared
`_fan_data_json`/`_fan_bands_json` converters in `helpers.jl` (A6).
"""

# =============================================================================
# Lane-local fan-panel assembly (A5 — a documented `_`-prefixed helper in the lane's
# own file; NOT a helpers.jl converter). Two paths: the default nested fan via
# `_render_fan_js`, and — when `draws>0` and a draw field is present — a line-renderer
# path that overlays the subsampled individual paths (the fan renderer cannot host an
# extra spaghetti series).
# =============================================================================

const _BAYES_DRAW_CAP = 200

# The true posterior-median central line: the 0.5-quantile column of `qmat` (T×nq) when
# that level is present in `levels`, else `fallback` (some result types store only a
# mean point estimate). Keeps `stat=:median` honest even when the stored `point_estimate`
# is the posterior mean (BayesianImpulseResponse defaults to `point_estimate=:mean`).
function _median_from_quantiles(qmat::AbstractMatrix, levels::AbstractVector,
                                fallback::AbstractVector)
    midx = findfirst(l -> isapprox(Float64(l), 0.5; atol=1e-8), levels)
    midx === nothing ? fallback : Float64[Float64(v) for v in @view qmat[:, midx]]
end

# Subsample the rows of an `n_draws × H` path matrix to at most `_BAYES_DRAW_CAP`,
# returning the drawn sub-matrix and the original draw count (0 ⇒ nothing to overlay).
function _subsample_draw_paths(paths::AbstractMatrix, requested::Int)
    nd = size(paths, 1)
    nd == 0 && return (paths[1:0, :], 0)
    keep = min(nd, requested, _BAYES_DRAW_CAP)
    keep <= 0 && return (paths[1:0, :], nd)
    step = max(1, cld(nd, keep))
    idx = collect(1:step:nd)
    length(idx) > keep && (idx = idx[1:keep])
    (paths[idx, :], nd)
end

# data_json rows `{x, q1..qk, med, d1..dm}` for the draws (line-renderer) path: nested
# band bounds (sorted like `_fan_data_json`), the central line, and one key per drawn
# posterior path. `paths_sub` is `m × H` (already subsampled).
function _bayes_draws_data_json(xs::AbstractVector, qmat::AbstractMatrix,
                                levels::AbstractVector, central::AbstractVector,
                                paths_sub::AbstractMatrix)
    Hn, nq = size(qmat)
    perm = sortperm(Float64[Float64(l) for l in levels])
    m = size(paths_sub, 1)
    rows = Vector{Pair{String,String}}[]
    for t in 1:Hn
        row = Pair{String,String}["x" => _json(xs[t])]
        for (ci, j) in enumerate(perm)
            push!(row, "q$ci" => _json(qmat[t, j]))
        end
        push!(row, "med" => _json(central[t]))
        for d in 1:m
            push!(row, "d$d" => _json(paths_sub[d, t]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# Nested-band specs `{lo_key,hi_key,name,alpha,color}` for the LINE renderer (which
# legends on `name`, unlike the fan renderer's `label`). Mirrors `_fan_bands_json`.
function _fan_bands_named_json(levels::AbstractVector; color::String=_PLOT_COLORS[1],
                               alpha_min::Real=0.12, alpha_max::Real=0.35)
    ls = sort(Float64[Float64(l) for l in levels])
    k = length(ls)
    npairs = fld(k, 2)
    parts = String[]
    for i in 1:npairs
        lo = ls[i]; hi = ls[k + 1 - i]
        lo ≈ hi && continue
        frac = npairs == 1 ? 1.0 : (i - 1) / (npairs - 1)
        alpha = alpha_min + frac * (alpha_max - alpha_min)
        label = "$(round(Int, lo * 100))–$(round(Int, hi * 100))%"
        push!(parts, "{\"lo_key\":$(_json("q$i")),\"hi_key\":$(_json("q$(k + 1 - i)"))," *
            "\"name\":$(_json(label)),\"alpha\":$(_json(alpha)),\"color\":$(_json(color))}")
    end
    "[" * join(parts, ",") * "]"
end

# Build one Bayesian fan panel. `qmat` is H×nq, `levels` the nq quantile levels,
# `central` the length-H central line, `draw_paths` an `n_draws × H` matrix or `nothing`.
# Returns `(_PanelSpec, note::String)` — the note carries any draw-cap surfaced in the
# figure (C7). `xs` are the x-axis positions (0-based horizons for IRF, 1-based for HD).
function _bayes_fan_panel(prefix::String, ptitle::String, xs::AbstractVector,
                          qmat::AbstractMatrix, levels::AbstractVector,
                          central::AbstractVector, central_label::String,
                          draw_paths, requested_draws::Int;
                          xlabel::String, ylabel::String)
    id = _next_plot_id(prefix)
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    if requested_draws > 0 && draw_paths !== nothing
        paths_sub, nd = _subsample_draw_paths(draw_paths, requested_draws)
        if size(paths_sub, 1) > 0
            m = size(paths_sub, 1)
            data = _bayes_draws_data_json(xs, qmat, levels, central, paths_sub)
            names = String[central_label]
            colors = String[_PLOT_COLORS[1]]
            keys = String["med"]
            dash = String[""]
            for d in 1:m
                push!(names, d == 1 ? "Posterior draws" : "")
                push!(colors, "#c7c7c7"); push!(keys, "d$d"); push!(dash, "")
            end
            s_json = _series_json(names, colors; keys=keys, dash=dash)
            bands = _fan_bands_named_json(levels)
            js = _render_line_js(id, data, s_json; bands_json=bands, ref_lines_json=refs,
                                 xlabel=xlabel, ylabel=ylabel)
            note = _cap_title(ptitle, m, nd)  # "<title> (m of nd)" when subsampled
            ptitle2 = m < nd ? "$(ptitle) — $(m)/$(nd) draws" : "$(ptitle) — $(m) draws"
            return (_PanelSpec(id, ptitle2, js), _cap_note("draws", m, nd, "draws"))
        end
    end
    fan_data = _fan_data_json(qmat, levels, central; xvals=xs)
    fan_bands = _fan_bands_json(levels)
    js = _render_fan_js(id, fan_data, fan_bands; central_label=central_label,
                        ref_lines_json=refs, xlabel=xlabel, ylabel=ylabel)
    (_PanelSpec(id, ptitle, js), "")
end

# =============================================================================
# BayesianDSGESimulation (NEW dispatch — no prior plot; lives here to keep models.jl
# untouched by this lane). quantiles: T×n_vars×nq; point_estimate: T×n_vars (median);
# all_paths: n_draws×T×n_vars.
# =============================================================================

"""
    plot_result(sim::BayesianDSGESimulation; var=nothing, stat=:median, draws=0,
                max_panels=12, ncols=0, title="", save_path=nothing)

Plot a Bayesian DSGE posterior simulation as nested credible fans, one panel per
variable. All quantile bands in `sim.quantile_levels` render (PLT-28). `stat` selects
the central line (`:median` = `sim.point_estimate`, `:mean` = mean of `sim.all_paths`);
`draws>0` overlays up to 200 subsampled posterior paths from `all_paths`. `var` selects
one variable by `Int` or name (`sim.variables`); otherwise up to `max_panels` variables
are drawn, with any cap surfaced in a figure note (plotrule C7).
"""
function plot_result(sim::BayesianDSGESimulation{T};
                     var::Union{Int,String,Nothing}=nothing,
                     stat::Symbol=:median, draws::Int=0, max_panels::Int=12, ncols::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    stat in (:median, :mean) ||
        throw(ArgumentError("stat must be :median or :mean, got :$stat"))
    Tn = size(sim.point_estimate, 1)
    n_vars = length(sim.variables)
    levels = sim.quantile_levels
    xs = collect(1:Tn)
    central_label = stat === :mean ? "Mean" : "Median"

    if var === nothing
        shown = min(n_vars, max_panels)
        vars_to_plot = collect(1:shown)
        cap_note = _cap_note("variables", shown, n_vars, "max_panels")
    else
        vars_to_plot = [_resolve_var(var, sim.variables)]
        cap_note = ""
    end

    panels = _PanelSpec[]
    draw_note = ""
    for vi in vars_to_plot
        qmat = sim.quantiles[1:Tn, vi, :]                    # T×nq
        central = if stat === :mean
            vec(sum(@view(sim.all_paths[:, 1:Tn, vi]), dims=1)) ./ max(size(sim.all_paths, 1), 1)
        else
            _median_from_quantiles(qmat, levels, sim.point_estimate[1:Tn, vi])
        end
        draw_paths = draws > 0 ? sim.all_paths[:, 1:Tn, vi] : nothing
        panel, n = _bayes_fan_panel("bsim", sim.variables[vi], xs, qmat, levels,
                                    central, central_label, draw_paths, draws;
                                    xlabel="Period", ylabel="Value")
        push!(panels, panel)
        isempty(draw_note) && (draw_note = n)
    end

    note = isempty(cap_note) ? draw_note :
           (isempty(draw_note) ? cap_note : cap_note * " " * draw_note)
    if isempty(title)
        title = "Bayesian DSGE Simulation ($(central_label))"
    end
    p = _make_plot(panels; title=title, ncols=ncols, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end
