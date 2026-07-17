# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for multivariate GARCH models (EV-16, #424):
`view=:correlations` (pairwise conditional-correlation series over time) and
`view=:covariance_heatmap` (the conditional covariance `Hₜ` at a chosen `t`).
"""

# =============================================================================
# MGARCHModel — conditional correlations & covariance heatmap
# =============================================================================

"""
    plot_result(m::MGARCHModel; view=:correlations, at=nothing, title="", save_path=nothing)

Visualize a multivariate GARCH fit.

- `view=:correlations` — every pairwise conditional correlation `ρᵢⱼ,ₜ` over time.
- `view=:covariance_heatmap` — heatmap of the conditional covariance `Hₜ` at time `at`
  (default: the last observation).
"""
function plot_result(m::MGARCHModel{T};
                     view::Symbol=:correlations, at::Union{Int,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :correlations
        p = _mgarch_corr_plot(m, title)
    elseif view === :covariance_heatmap
        p = _mgarch_cov_heatmap(m, at, title)
    else
        throw(ArgumentError("unknown view $view — use :correlations or :covariance_heatmap"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _mgarch_corr_plot(m::MGARCHModel{T}, title::String) where {T}
    id = _next_plot_id("mgarch_corr")
    R = correlations(m)      # n×n×T
    n = m.n
    Tn = size(R, 3)
    pairs = [(i, j) for i in 1:n for j in (i+1):n]
    keys = ["r$(i)_$(j)" for (i, j) in pairs]
    names = ["ρ($i,$j)" for (i, j) in pairs]

    rows = Vector{Pair{String,String}}[]
    for t in 1:Tn
        row = Pair{String,String}["x" => _json(t)]
        for (k, (i, j)) in enumerate(pairs)
            push!(row, keys[k] => _json(R[i, j, t]))
        end
        push!(rows, row)
    end
    data_json = _json_array_of_objects(rows)
    colors = [_PLOT_COLORS[mod1(k, length(_PLOT_COLORS))] for k in 1:length(pairs)]
    s_json = _series_json(names, colors; keys=keys)
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data_json, s_json;
                         ref_lines_json=refs, xlabel="Time", ylabel="Conditional correlation")
    isempty(title) && (title = "Conditional Correlations ($(m.kind))")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

function _mgarch_cov_heatmap(m::MGARCHModel{T}, at::Union{Int,Nothing}, title::String) where {T}
    Tn = size(m.H, 3)
    t = at === nothing ? Tn : at
    (1 <= t <= Tn) || throw(ArgumentError("at=$t out of range 1:$Tn"))
    n = m.n
    labels = ["series $i" for i in 1:n]
    Ht = m.H[:, :, t]
    vmax = maximum(abs, Ht)
    vmax = vmax > 0 ? vmax : one(T)

    rows = Vector{Pair{String,String}}[]
    for i in 1:n, j in 1:n
        push!(rows, [
            "x" => _json(labels[j]),
            "y" => _json(labels[i]),
            "v" => _json(Ht[i, j]),
        ])
    end
    data_json = _json_array_of_objects(rows)
    row_labels_json = _json(labels)
    col_labels_json = _json(labels)
    id = _next_plot_id("mgarch_cov")
    js = _render_heatmap_js(id, data_json, row_labels_json, col_labels_json;
                            xlabel="", ylabel="",
                            color_domain=[-Float64(vmax), Float64(vmax)])
    isempty(title) && (title = "Conditional Covariance Hₜ at t=$t ($(m.kind))")
    _make_plot([_PanelSpec(id, "Covariance Hₜ (t=$t)", js)]; title=title)
end
