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
- `group_names::Union{Vector{String},Nothing}=nothing` — labels for groups (heatmap row prefixes; contributions legend)
- `variable_names::Union{Vector{String},Nothing}=nothing` — labels for variables (`:default` panels and heatmap)
- `dates::Union{Vector{String},Nothing}=nothing` — calendar/vintage labels for the heatmap columns (length ≥ T_obs)
- `n_periods::Int=18` — number of recent periods to show (heatmap)
"""
function plot_result(nr::NowcastResult{T};
                     view::Symbol=:default,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     groups::Union{Vector{Int},Nothing}=nothing,
                     group_names::Union{Vector{String},Nothing}=nothing,
                     variable_names::Union{Vector{String},Nothing}=nothing,
                     dates::Union{Vector{String},Nothing}=nothing,
                     n_periods::Int=18) where {T}
    if view == :default
        p = _plot_nowcast_default(nr; ncols=ncols, title=title, variable_names=variable_names)
    elseif view == :heatmap
        p = _plot_nowcast_heatmap(nr; title=title, groups=groups,
                                  group_names=group_names,
                                  variable_names=variable_names,
                                  dates=dates,
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

function _plot_nowcast_default(nr::NowcastResult{T}; ncols::Int=0, title::String="",
                               variable_names::Union{Vector{String},Nothing}=nothing) where {T}
    T_obs, n_vars = size(nr.X_sm)
    n_show = min(n_vars, 6)
    # Resolve display labels (#595)
    vlab(i) = (variable_names !== nothing && 1 <= i <= length(variable_names)) ?
              variable_names[i] : "Variable $i"

    panels = _PanelSpec[]

    # Target variable panel with nowcast/forecast extension
    ti = nr.target_index
    id = _next_plot_id("nc_target")
    ptitle = "Target ($(vlab(ti))) — Nowcast: $(round(nr.nowcast, digits=3)), Forecast: $(round(nr.forecast, digits=3))"

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
        s_v = _series_json(["Smoothed $(vlab(vi))"], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                           keys=["v1"])
        js_v = _render_line_js(id_v, data_v, s_v; xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id_v, vlab(vi), js_v))
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
                               dates::Union{Vector{String},Nothing}=nothing,
                               n_periods::Int=18) where {T}
    data = nr.model.data
    T_obs, n_vars = size(data)

    # Variable labels
    vnames = if variable_names !== nothing
        variable_names
    else
        ["Var $i" for i in 1:n_vars]
    end

    # Row label for a variable: group-prefixed when a group assignment is supplied
    # (wires the previously-dead `group_names` kwarg into the heatmap, plotrule C4).
    _row_label = function(vi)
        if groups !== nothing
            g = groups[vi]
            gname = (group_names !== nothing && g <= length(group_names)) ? group_names[g] : "Group $g"
            return "$(gname): $(vnames[vi])"
        else
            return vnames[vi]
        end
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
    # Column labels: calendar dates when supplied (length ≥ T_obs), else integer period.
    use_dates = dates !== nothing && length(dates) >= T_obs
    _col_label = t -> use_dates ? dates[t] : string(t)
    col_labels = [_col_label(t) for t in t_start:t_end]

    # Row ordering: by groups if provided, otherwise 1:n_vars
    if groups !== nothing
        row_order = sortperm(groups)
    else
        row_order = collect(1:n_vars)
    end
    row_labels = String[_row_label(vi) for vi in row_order]

    # Build heatmap data: {x: col_label, y: row_label, v: z_score or null}
    hm_rows = Vector{Pair{String,String}}[]
    for vi in row_order
        for t in t_start:t_end
            val = z_data[t, vi]
            push!(hm_rows, [
                "x" => _json(_col_label(t)),
                "y" => _json(_row_label(vi)),
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

    # Signed z-scores → diverging scale, explicit symmetric ±3 domain (standard
    # z-score cut; plotrule Heatmaps rule 3, PLT-15). tip_label kills the "Period" noun.
    js = _render_heatmap_js(id, data_json, row_labels_json, col_labels_json;
                            xlabel=use_dates ? "Date" : "Period", ylabel="",
                            scale=:diverging, color_domain=[-3.0, 3.0])

    _make_plot([_PanelSpec(id, "Z-Score Heatmap (values are z-scores, clipped ±3)", js)]; title=title)
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

    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                        xlabel="Impact", ylabel="")

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

    # sorted_idx is largest-|impact| first; the y-band runs top→bottom, so the
    # largest impact sits at the top of the horizontal chart.
    js = _render_bar_js(id, data_json, s_json; mode="grouped", orientation="h",
                        xlabel="Impact", ylabel="")

    _make_plot([_PanelSpec(id, "Individual Impacts (sorted)", js)];
               title=_news_title(nn, title))
end

# =============================================================================
# GDFMNowcastResult (GDFM-08 #900)
# =============================================================================

# Resolve a `vars` selector (nothing, single Int/String, or vector/range of Int/String)
# to 1-based column indices through `_resolve_var` (bounds-checked, plotrule C3).
function _gdfm_resolve_vars(vars, names::Vector{String})
    vars === nothing && return collect(1:length(names))
    if vars isa Integer
        return Int[_resolve_var(Int(vars), names)]
    elseif vars isa AbstractString
        return Int[_resolve_var(String(vars), names)]
    elseif vars isa AbstractVector
        isempty(vars) && throw(ArgumentError("vars must not be empty"))
        out = Int[]
        for v in vars
            vv = v isa Integer ? Int(v) : v isa AbstractString ? String(v) :
                throw(ArgumentError("vars entries must be Int or String, got $(typeof(v))"))
            push!(out, _resolve_var(vv, names))
        end
        return out
    else
        throw(ArgumentError("vars must be nothing, Int, String, or a vector/range of Int/String, got $(typeof(vars))"))
    end
end

"""
    plot_result(r::GDFMNowcastResult; vars=1:min(6, length(r.nowcast)), ncols=0,
                title="", save_path=nothing) -> PlotOutput

Plot Wiener–Kolmogorov GDFM ragged-edge nowcast/forecast/backcast paths alongside
the imputed panel series and 95% predictive uncertainty bands (`nowcast ± 1.96 * se`)
for selected series.

# Keyword Arguments
- `vars=1:min(6, length(r.nowcast))`: Indices or names of panel variables to display
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(r::GDFMNowcastResult{T};
                     vars=1:min(6, length(r.nowcast)),
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    T_r, N = size(r.panel_imputed)
    var_idxs = _gdfm_resolve_vars(vars, r.varnames)
    z_val = T(1.96)
    x_nc = T_r + r.horizon

    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id("gdfm_nc")
        nc = r.nowcast[vi]
        se = r.se[vi]
        lo = nc - z_val * se
        hi = nc + z_val * se

        local xs::Vector{Float64}
        local hist_vals::Vector{T}
        local nc_vals::Vector{T}
        local lo_vals::Vector{T}
        local hi_vals::Vector{T}

        if r.horizon > 0
            xs = vcat(Float64.(1:T_r), Float64(x_nc))
            hist_vals = vcat(r.panel_imputed[:, vi], T[T(NaN)])
            nc_vals = fill(T(NaN), T_r + 1)
            lo_vals = fill(T(NaN), T_r + 1)
            hi_vals = fill(T(NaN), T_r + 1)
            nc_vals[T_r] = r.panel_imputed[T_r, vi]
            lo_vals[T_r] = r.panel_imputed[T_r, vi]
            hi_vals[T_r] = r.panel_imputed[T_r, vi]
            nc_vals[end] = nc
            lo_vals[end] = lo
            hi_vals[end] = hi
        elseif T_r == 1 && x_nc == 1
            xs = Float64[0.5, 1.0, 1.5]
            hist_vals = T[T(NaN), r.panel_imputed[1, vi], T(NaN)]
            nc_vals = T[nc, nc, nc]
            lo_vals = T[lo, lo, lo]
            hi_vals = T[hi, hi, hi]
        elseif x_nc < 1
            xs = vcat(Float64(x_nc), Float64.(1:T_r))
            hist_vals = vcat(T[T(NaN)], r.panel_imputed[:, vi])
            nc_vals = fill(T(NaN), T_r + 1)
            lo_vals = fill(T(NaN), T_r + 1)
            hi_vals = fill(T(NaN), T_r + 1)
            nc_vals[1] = nc
            lo_vals[1] = lo
            hi_vals[1] = hi
            nc_vals[2] = r.panel_imputed[1, vi]
            lo_vals[2] = r.panel_imputed[1, vi]
            hi_vals[2] = r.panel_imputed[1, vi]
        else
            xs = Float64.(1:T_r)
            hist_vals = copy(r.panel_imputed[:, vi])
            nc_vals = fill(T(NaN), T_r)
            lo_vals = fill(T(NaN), T_r)
            hi_vals = fill(T(NaN), T_r)
            t_idx = Int(x_nc)
            t_nbr = t_idx > 1 ? t_idx - 1 : t_idx + 1
            nc_vals[t_idx] = nc
            lo_vals[t_idx] = lo
            hi_vals[t_idx] = hi
            nc_vals[t_nbr] = r.panel_imputed[t_nbr, vi]
            lo_vals[t_nbr] = r.panel_imputed[t_nbr, vi]
            hi_vals[t_nbr] = r.panel_imputed[t_nbr, vi]
        end

        data_json = _keyed_rows_json(xs, [
            "hist" => hist_vals,
            "nc" => nc_vals,
            "ci_lo" => lo_vals,
            "ci_hi" => hi_vals,
        ])
        s_json = _series_json(["Imputed panel", "Nowcast"],
                              [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                              keys=["hist", "nc"], dash=["", "6,3"])
        bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"name\":\"95% CI\"," *
                "\"color\":$(_json(_PLOT_SERIES[2])),\"alpha\":$(_PLOT_CI_ALPHA)}]"
        refs = "[{\"value\":$(_json(x_nc)),\"axis\":\"x\"," *
               "\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,3\"}]"
        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands, ref_lines_json=refs,
                             integer_x=(T_r > 1 || r.horizon != 0),
                             xlabel="Period", ylabel=r.varnames[vi])
        status = r.observed_mask[end, vi] ? "" : " [missing at T]"
        ptitle = "$(r.varnames[vi])$(status): $(_fmt(nc; digits=3)) (±$(_fmt(z_val * se; digits=3)))"
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        kind_str = r.horizon == 0 ? "Nowcast (h=0)" :
                   r.horizon > 0 ? "Forecast (h=$(r.horizon))" :
                   "Backcast (h=$(r.horizon))"
        title = "GDFM Ragged-Edge $(kind_str), n_lags=$(r.n_lags)"
    end
    p = _make_plot(panels; title=title, ncols=ncols,
                   note=_cap_note("variables", length(var_idxs), N, "vars"))
    save_path !== nothing && save_plot(p, save_path)
    p
end
