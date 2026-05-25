# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for nowcasting types: NowcastResult, NowcastNews.
"""

# =============================================================================
# NowcastResult — view dispatch
# =============================================================================

"""
    plot_result(nr::NowcastResult; view=:default, ncols=0, title="", save_path=nothing, ...)

Plot nowcast result with multiple view options.

# Views
- `:default` — smoothed data for target variable with nowcast/forecast extension, plus DFM factor panels
- `:heatmap` — z-score heatmap of all variables over recent periods
- `:contributions` — stacked bar chart of factor/block contributions to nowcast (DFM only)

# Keyword Arguments
- `view::Symbol=:default` — plot view
- `ncols::Int=0` — number of columns in panel grid (0 = auto)
- `title::String=""` — figure title (auto-generated if empty)
- `save_path::Union{String,Nothing}=nothing` — save HTML to file
- `groups::Union{Vector{Int},Nothing}=nothing` — group assignment per factor (heatmap/contributions)
- `group_names::Union{Vector{String},Nothing}=nothing` — labels for groups
- `variable_names::Union{Vector{String},Nothing}=nothing` — labels for variables (heatmap)
- `n_periods::Int=18` — number of recent periods to show (heatmap)
"""
function plot_result(nr::NowcastResult{T};
                     view::Symbol=:default,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     groups::Union{Vector{Int},Nothing}=nothing,
                     group_names::Union{Vector{String},Nothing}=nothing,
                     variable_names::Union{Vector{String},Nothing}=nothing,
                     n_periods::Int=18) where {T}
    if view == :default
        p = _plot_nowcast_default(nr; ncols=ncols, title=title)
    elseif view == :heatmap
        p = _plot_nowcast_heatmap(nr; title=title, groups=groups,
                                  group_names=group_names,
                                  variable_names=variable_names,
                                  n_periods=n_periods)
    elseif view == :contributions
        p = _plot_nowcast_contributions(nr; title=title, ncols=ncols,
                                        groups=groups, group_names=group_names)
    else
        throw(ArgumentError("Unknown view: $view. Expected :default, :heatmap, or :contributions"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Default View
# =============================================================================

function _plot_nowcast_default(nr::NowcastResult{T}; ncols::Int=0, title::String="") where {T}
    T_obs, n_vars = size(nr.X_sm)
    n_show = min(n_vars, 6)

    panels = _PanelSpec[]

    # Target variable panel with nowcast/forecast extension
    ti = nr.target_index
    id = _next_plot_id("nc_target")
    ptitle = "Target (col $ti) — Nowcast: $(round(nr.nowcast, digits=3)), Forecast: $(round(nr.forecast, digits=3))"

    rows = Vector{Pair{String,String}}[]
    for t in 1:T_obs
        push!(rows, [
            "x" => _json(t),
            "v1" => _json(nr.X_sm[t, ti]),
            "v2" => _json(t == T_obs ? nr.X_sm[t, ti] : NaN)
        ])
    end
    # Extend with nowcast (T+1) and forecast (T+2) as separate series
    push!(rows, [
        "x" => _json(T_obs + 1),
        "v1" => "null",
        "v2" => _json(nr.nowcast)
    ])
    push!(rows, [
        "x" => _json(T_obs + 2),
        "v1" => "null",
        "v2" => _json(nr.forecast)
    ])
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Smoothed", "Nowcast/Forecast"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]]; keys=["v1", "v2"])
    js = _render_line_js(id, data_json, s_json; xlabel="Period", ylabel="Value")
    push!(panels, _PanelSpec(id, ptitle, js))

    # Additional variable panels
    other_vars = setdiff(1:min(n_vars, n_show + 1), [ti])
    for vi in other_vars[1:min(length(other_vars), n_show - 1)]
        id_v = _next_plot_id("nc_var")

        rows_v = Vector{Pair{String,String}}[]
        for t in 1:T_obs
            push!(rows_v, [
                "x" => _json(t),
                "v1" => _json(nr.X_sm[t, vi])
            ])
        end
        data_v = _json_array_of_objects(rows_v)
        s_v = _series_json(["Smoothed var $vi"], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                           keys=["v1"])
        js_v = _render_line_js(id_v, data_v, s_v; xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id_v, "Variable $vi", js_v))
    end

    # DFM factor panels
    if nr.model isa NowcastDFM
        model = nr.model::NowcastDFM{T}
        r = model.r
        n_blocks = size(model.blocks, 2)
        n_factors = r * n_blocks
        n_factor_cols = size(model.F, 2)  # r*p columns
        for fi in 1:min(n_factors, n_factor_cols)
            id_f = _next_plot_id("nc_factor")
            rows_f = Vector{Pair{String,String}}[]
            for t in 1:T_obs
                push!(rows_f, [
                    "x" => _json(t),
                    "v1" => _json(model.F[t, fi])
                ])
            end
            data_f = _json_array_of_objects(rows_f)
            s_f = _series_json(["Factor $fi"], [_PLOT_COLORS[mod1(fi + 2, length(_PLOT_COLORS))]];
                               keys=["v1"])
            js_f = _render_line_js(id_f, data_f, s_f; xlabel="Period", ylabel="")
            push!(panels, _PanelSpec(id_f, "Factor $fi", js_f))
        end
    end

    if isempty(title)
        title = "Nowcast Result ($(nr.method))"
    end

    _make_plot(panels; title=title, ncols=ncols)
end

# =============================================================================
# Heatmap View
# =============================================================================

function _plot_nowcast_heatmap(nr::NowcastResult{T};
                               title::String="",
                               groups::Union{Vector{Int},Nothing}=nothing,
                               group_names::Union{Vector{String},Nothing}=nothing,
                               variable_names::Union{Vector{String},Nothing}=nothing,
                               n_periods::Int=18) where {T}
    data = nr.model.data
    T_obs, n_vars = size(data)

    # Variable labels
    vnames = if variable_names !== nothing
        variable_names
    else
        ["Var $i" for i in 1:n_vars]
    end

    # Compute z-scores per column (using non-NaN values)
    z_data = similar(data)
    for j in 1:n_vars
        col = data[:, j]
        valid = filter(!isnan, col)
        mu = isempty(valid) ? zero(T) : mean(valid)
        sigma = isempty(valid) ? one(T) : std(valid)
        sigma = sigma < T(1e-10) ? one(T) : sigma
        for t in 1:T_obs
            z_data[t, j] = isnan(data[t, j]) ? T(NaN) : (data[t, j] - mu) / sigma
        end
    end

    # Select last n_periods
    t_start = max(1, T_obs - n_periods + 1)
    t_end = T_obs
    col_labels = [string(t) for t in t_start:t_end]

    # Row ordering: by groups if provided, otherwise 1:n_vars
    if groups !== nothing
        row_order = sortperm(groups)
    else
        row_order = collect(1:n_vars)
    end
    row_labels = vnames[row_order]

    # Build heatmap data: {x: col_label, y: row_label, v: z_score or null}
    hm_rows = Vector{Pair{String,String}}[]
    for vi in row_order
        for t in t_start:t_end
            val = z_data[t, vi]
            push!(hm_rows, [
                "x" => _json(string(t)),
                "y" => _json(vnames[vi]),
                "v" => isnan(val) ? "null" : _json(val)
            ])
        end
    end
    data_json = _json_array_of_objects(hm_rows)
    row_labels_json = _json(row_labels)
    col_labels_json = _json(col_labels)

    id = _next_plot_id("nc_heatmap")
    if isempty(title)
        title = "Nowcast Data Availability & Z-Scores ($(nr.method))"
    end

    js = _render_heatmap_js(id, data_json, row_labels_json, col_labels_json;
                            xlabel="Period", ylabel="")

    _make_plot([_PanelSpec(id, "Z-Score Heatmap", js)]; title=title)
end

# =============================================================================
# Contributions View
# =============================================================================

function _plot_nowcast_contributions(nr::NowcastResult{T};
                                     title::String="",
                                     ncols::Int=0,
                                     groups::Union{Vector{Int},Nothing}=nothing,
                                     group_names::Union{Vector{String},Nothing}=nothing) where {T}
    if !(nr.model isa NowcastDFM)
        throw(ArgumentError("Contributions view requires a DFM nowcast model, got $(typeof(nr.model))"))
    end

    model = nr.model::NowcastDFM{T}
    r = model.r
    n_blocks = size(model.blocks, 2)
    n_factors = r * n_blocks
    state_dim = size(model.A, 1)
    ti = nr.target_index

    # Current state from F (pad with zeros if needed)
    n_f_cols = size(model.F, 2)
    current_state = zeros(T, state_dim)
    current_state[1:n_f_cols] = model.F[end, :]

    # Forecast state (one step)
    forecast_state = model.A * current_state

    # Compute per-factor contributions for nowcast and forecast
    # contribution_fi = C[ti, fi] * state[fi] * Wx[ti]
    n_contrib = min(n_factors, state_dim)

    # Group assignment: use blocks if groups not provided
    if groups !== nothing
        group_ids = groups
    else
        # Assign factor fi to block b = div(fi - 1, r) + 1
        group_ids = [div(fi - 1, r) + 1 for fi in 1:n_contrib]
    end
    n_groups = maximum(group_ids)

    if group_names !== nothing
        gnames = group_names
    else
        gnames = ["Block $i" for i in 1:n_groups]
    end

    # Aggregate contributions by group
    nowcast_contribs = zeros(T, n_groups)
    forecast_contribs = zeros(T, n_groups)
    for fi in 1:n_contrib
        g = group_ids[fi]
        nowcast_contribs[g] += model.C[ti, fi] * current_state[fi] * model.Wx[ti]
        forecast_contribs[g] += model.C[ti, fi] * forecast_state[fi] * model.Wx[ti]
    end

    # Mean bar
    mean_val = model.Mx[ti]

    # Build bar data: 3 bars (Mean, Nowcast, Forecast) x n_groups stacked
    bar_labels = ["Mean", "Nowcast", "Forecast"]

    rows = Vector{Pair{String,String}}[]

    # Mean bar: all contribution in group 1
    row_mean = Pair{String,String}["x" => _json("Mean")]
    for g in 1:n_groups
        push!(row_mean, "g$g" => _json(g == 1 ? mean_val : zero(T)))
    end
    push!(rows, row_mean)

    # Nowcast bar
    row_now = Pair{String,String}["x" => _json("Nowcast")]
    for g in 1:n_groups
        push!(row_now, "g$g" => _json(nowcast_contribs[g]))
    end
    push!(rows, row_now)

    # Forecast bar
    row_fc = Pair{String,String}["x" => _json("Forecast")]
    for g in 1:n_groups
        push!(row_fc, "g$g" => _json(forecast_contribs[g]))
    end
    push!(rows, row_fc)

    data_json = _json_array_of_objects(rows)

    names = String[]
    colors = String[]
    keys_arr = String[]
    for g in 1:n_groups
        push!(names, g <= length(gnames) ? gnames[g] : "Block $g")
        push!(colors, _PLOT_COLORS[mod1(g, length(_PLOT_COLORS))])
        push!(keys_arr, "g$g")
    end
    s_json = _series_json(names, colors; keys=keys_arr)

    id = _next_plot_id("nc_contrib")
    if isempty(title)
        title = "Nowcast Contribution Decomposition ($(nr.method))"
    end

    js = _render_bar_js(id, data_json, s_json; mode="stacked",
                        xlabel="", ylabel="Contribution")

    _make_plot([_PanelSpec(id, "Factor Contributions", js)]; title=title, ncols=ncols)
end

# =============================================================================
# NowcastNews
# =============================================================================

"""
    plot_result(nn::NowcastNews; view=:releases, title="", save_path=nothing)

Plot nowcast news decomposition.

# Views
- `:releases` — horizontal bar chart of per-release impact (default)
- `:groups` — stacked bar chart of group impacts
- `:individual` — horizontal bar chart sorted by absolute impact
"""
function plot_result(nn::NowcastNews{T};
                     view::Symbol=:releases,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view == :releases
        p = _plot_news_releases(nn; title=title)
    elseif view == :groups
        p = _plot_news_groups(nn; title=title)
    elseif view == :individual
        p = _plot_news_individual(nn; title=title)
    else
        throw(ArgumentError("Unknown view: $view. Expected :releases, :groups, or :individual"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _news_title(nn::NowcastNews{T}, prefix::String) where {T}
    delta = nn.new_nowcast - nn.old_nowcast
    isempty(prefix) ? "Nowcast News: $(round(nn.old_nowcast, digits=3)) → $(round(nn.new_nowcast, digits=3)) (Δ = $(round(delta, digits=3)))" : prefix
end

function _plot_news_releases(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news")
    n_releases = length(nn.impact_news)

    rows = Vector{Pair{String,String}}[]
    for i in 1:n_releases
        label = i <= length(nn.variable_names) ? nn.variable_names[i] : "Release $i"
        push!(rows, [
            "x" => _json(label),
            "impact" => _json(nn.impact_news[i])
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["News Impact"], [_PLOT_COLORS[1]]; keys=["impact"])

    js = _render_bar_js(id, data_json, s_json; mode="grouped",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "Per-Release Impact", js)];
               title=_news_title(nn, title))
end

function _plot_news_groups(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news_grp")
    n_groups = length(nn.group_impacts)

    bar_labels = String["News"]
    has_other = abs(nn.impact_revision) + abs(nn.impact_reestimation) > T(1e-10)
    if has_other
        push!(bar_labels, "Revision + Re-est.")
    end

    rows = Vector{Pair{String,String}}[]

    row_news = Pair{String,String}["x" => _json("News")]
    for g in 1:n_groups
        push!(row_news, "g$g" => _json(nn.group_impacts[g]))
    end
    push!(rows, row_news)

    if has_other
        row_other = Pair{String,String}["x" => _json("Revision + Re-est.")]
        for g in 1:n_groups
            push!(row_other, "g$g" => _json(g == 1 ? nn.impact_revision + nn.impact_reestimation : zero(T)))
        end
        push!(rows, row_other)
    end

    data_json = _json_array_of_objects(rows)

    names = String[]
    colors = String[]
    keys_arr = String[]
    for g in 1:n_groups
        push!(names, g <= length(nn.group_names) ? nn.group_names[g] : "Group $g")
        push!(colors, _PLOT_COLORS[mod1(g, length(_PLOT_COLORS))])
        push!(keys_arr, "g$g")
    end
    s_json = _series_json(names, colors; keys=keys_arr)

    js = _render_bar_js(id, data_json, s_json; mode="stacked",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "News by Group", js)];
               title=_news_title(nn, title))
end

function _plot_news_individual(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news_ind")
    n_releases = length(nn.impact_news)

    sorted_idx = sortperm(abs.(nn.impact_news), rev=true)

    rows = Vector{Pair{String,String}}[]
    for i in sorted_idx
        label = i <= length(nn.variable_names) ? nn.variable_names[i] : "Release $i"
        push!(rows, [
            "x" => _json(label),
            "impact" => _json(nn.impact_news[i])
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["News Impact"], [_PLOT_COLORS[1]]; keys=["impact"])

    js = _render_bar_js(id, data_json, s_json; mode="grouped",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "Individual Impacts (sorted)", js)];
               title=_news_title(nn, title))
end
