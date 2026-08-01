# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for the statistical-identification SVAR family (PLT-32):

- `MarkovSwitchingSVARResult` — `view=:regimes` (smoothed regime-probability stacked
  area, 0–1), `:B0` (signed structural-impact heatmap, diverging), `:transition`
  (Markov transition-matrix heatmap, nonnegative ⇒ sequential single-hue).
- `GARCHSVARResult` — `view=:variance` (per-shock conditional-variance lines) /
  `:shocks` (structural-shock lines).
- `SmoothTransitionSVARResult` — the transition function `G(s_t)` vs the sorted
  transition variable, with a threshold vline and `γ` annotated.
- `ExternalVolatilitySVARResult` — regime-membership line over observation index.
- `ICASVARResult` — `view=:mixing` (`B0` heatmap) / `:unmixing` (`W` heatmap) /
  `:shocks` (recovered structural-shock lines).
- `NonGaussianMLResult` — `B0` heatmap with a likelihood-ratio annotation.

All rendering reuses the frozen renderers (A1); matrix views go through the shared
heatmap renderer with a color-scale legend and a sign-appropriate scale (PLT-15):
`B0`/`W` are signed ⇒ diverging; the transition matrix is nonnegative ⇒ sequential.
"""

# =============================================================================
# Lane-local converters (A5/A6) — helpers.jl is frozen for Wave 2
# =============================================================================

"""
    _regime_prob_json(P) -> String

Stacked-area `data_json` for smoothed regime probabilities `P` (T × K): one row per
period `[{x, s1, …, sK}, …]` keyed `s1..sK` to match `_series_json`'s default keys.
Non-finite probabilities serialize to `null` (a visible gap; plotrule A7).
"""
function _regime_prob_json(P::AbstractMatrix)
    Tn, K = size(P)
    rows = Vector{Pair{String,String}}[]
    for t in 1:Tn
        row = Pair{String,String}["x" => _json(t)]
        for k in 1:K
            push!(row, "s$k" => _json(P[t, k]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

"""
    _matrix_heatmap_json(M, row_labels, col_labels) -> String

Heatmap-cell `data_json` `[{x, y, v}, …]` for a dense matrix `M`, with `x` the
column label and `y` the row label (matching `_render_heatmap_js`'s band scales).
Non-finite entries serialize to `null` (rendered as the neutral "missing" grey).
"""
function _matrix_heatmap_json(M::AbstractMatrix, row_labels::Vector{String},
                              col_labels::Vector{String})
    nr, nc = size(M)
    (length(row_labels) == nr && length(col_labels) == nc) || throw(ArgumentError(
        "label lengths ($(length(row_labels))×$(length(col_labels))) must match matrix size ($nr×$nc)"))
    rows = Vector{Pair{String,String}}[]
    for i in 1:nr, j in 1:nc
        push!(rows, ["x" => _json(col_labels[j]), "y" => _json(row_labels[i]),
                     "v" => _json(M[i, j])])
    end
    _json_array_of_objects(rows)
end

# One line panel of a T×n series matrix (conditional variances, structural shocks):
# columns become entity-stable series keyed v1..vn over an integer time index.
function _statid_line_panel(id::String, M::AbstractMatrix, names::Vector{String};
                            ylabel::String, xlabel::String="Time")
    data_json = _timeseries_data_json(M, names)
    series_json = _series_json(names, _colors_for(names); keys=["v$j" for j in 1:length(names)])
    _render_line_js(id, data_json, series_json; xlabel=xlabel, ylabel=ylabel)
end

# One heatmap panel for a matrix view (B0/W diverging; transition sequential).
function _statid_heatmap_panel(id::String, M::AbstractMatrix,
                               row_labels::Vector{String}, col_labels::Vector{String};
                               scale::Symbol, xlabel::String, ylabel::String,
                               tip_label::String)
    data_json = _matrix_heatmap_json(M, row_labels, col_labels)
    _render_heatmap_js(id, data_json, _json(row_labels), _json(col_labels);
                       scale=scale, xlabel=xlabel, ylabel=ylabel, tip_label=tip_label)
end

_statid_var_names(n::Int) = String["Var $i" for i in 1:n]
_statid_shock_names(n::Int) = String["Shock $j" for j in 1:n]

# =============================================================================
# MarkovSwitchingSVARResult
# =============================================================================

"""
    plot_result(r::MarkovSwitchingSVARResult; view=:regimes, title="", save_path=nothing)

Markov-switching SVAR diagnostics. Views (plotrule C5):

- `:regimes` (default) — the `K` smoothed regime probabilities as a stacked area
  over time, bounded to [0, 1], one legend entry per regime.
- `:B0` — the structural impact matrix `B₀` (signed) as a diverging heatmap with a
  color-scale legend.
- `:transition` — the `K×K` Markov transition matrix (nonnegative ∈ [0, 1]) as a
  sequential single-hue heatmap (**not** diverging — no meaningful midpoint).

Unknown `view` throws an `ArgumentError` naming the valid views.
"""
function plot_result(r::MarkovSwitchingSVARResult{T};
                     view::Symbol=:regimes, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B0, 1)
    K = r.n_regimes
    if view === :regimes
        id = _next_plot_id("ms_regime")
        names = String["Regime $k" for k in 1:K]
        data_json = _regime_prob_json(r.regime_probs)
        series_json = _series_json(names, _colors_for(names))
        js = _render_area_js(id, data_json, series_json; tip_label="t",
                             xlabel="Time", ylabel="Regime probability")
        ptitle = "Smoothed Regime Probabilities ($K regimes)"
        panel = _PanelSpec(id, ptitle, js)
        ftitle = isempty(title) ? "Markov-Switching SVAR — regime probabilities" : title
    elseif view === :B0
        id = _next_plot_id("ms_b0")
        js = _statid_heatmap_panel(id, r.B0, _statid_var_names(n), _statid_shock_names(n);
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Structural Impact Matrix (B₀)", js)
        ftitle = isempty(title) ? "Markov-Switching SVAR — impact matrix" : title
    elseif view === :transition
        id = _next_plot_id("ms_trans")
        rl = String["From R$i" for i in 1:K]
        cl = String["To R$j" for j in 1:K]
        js = _statid_heatmap_panel(id, r.transition_matrix, rl, cl;
                                   scale=:sequential, xlabel="To regime",
                                   ylabel="From regime", tip_label="")
        panel = _PanelSpec(id, "Transition Matrix (P)", js)
        ftitle = isempty(title) ? "Markov-Switching SVAR — transition matrix" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :regimes, :B0, :transition"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# GARCHSVARResult
# =============================================================================

"""
    plot_result(r::GARCHSVARResult; view=:variance, title="", save_path=nothing)

GARCH-SVAR diagnostics. `view=:variance` (default) draws the per-shock conditional
variances over time as overlaid lines; `view=:shocks` draws the recovered structural
shocks. Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::GARCHSVARResult{T};
                     view::Symbol=:variance, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B0, 1)
    names = _statid_shock_names(n)
    if view === :variance
        id = _next_plot_id("garch_var")
        js = _statid_line_panel(id, r.cond_var, names; ylabel="Conditional variance")
        panel = _PanelSpec(id, "Conditional Variances", js)
        ftitle = isempty(title) ? "GARCH-SVAR — conditional variances" : title
    elseif view === :shocks
        id = _next_plot_id("garch_shk")
        js = _statid_line_panel(id, r.shocks, names; ylabel="Structural shock")
        panel = _PanelSpec(id, "Structural Shocks", js)
        ftitle = isempty(title) ? "GARCH-SVAR — structural shocks" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :variance, :shocks"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SmoothTransitionSVARResult
# =============================================================================

"""
    plot_result(r::SmoothTransitionSVARResult; title="", save_path=nothing)

Draw the smooth-transition function `G(s_t)` against the sorted transition variable
`s_t` as a line, with a vertical reference line at the estimated `threshold` and the
transition speed `γ` (rounded via `_fmt`) in the panel title (plotrule C9). `G` runs
0→1 across the threshold.
"""
function plot_result(r::SmoothTransitionSVARResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("star_g")
    perm = sortperm(r.transition_var)
    sv = r.transition_var[perm]
    gv = r.G_values[perm]
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(sv)
        push!(rows, ["x" => _json(sv[i]), "g" => _json(gv[i])])
    end
    data_json = _json_array_of_objects(rows)
    series_json = _series_json(["G(s_t)"], [_PLOT_SERIES[1]]; keys=["g"])
    refs = "[{\"value\":$(_json(r.threshold)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}]"
    js = _render_line_js(id, data_json, series_json; ref_lines_json=refs,
                         xlabel="Transition variable", ylabel="G(s_t)")
    ptitle = "Transition Function (γ = $(_fmt(r.gamma; digits=3)), threshold = $(_fmt(r.threshold; digits=3)))"
    panel = _PanelSpec(id, ptitle, js)
    ftitle = isempty(title) ? "Smooth-Transition SVAR — transition function" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ExternalVolatilitySVARResult
# =============================================================================

"""
    plot_result(r::ExternalVolatilitySVARResult; title="", save_path=nothing)

Regime-membership plot: the assigned regime index over the observation index, read
from `regime_indices` (`regime_indices[k]` lists the observations in regime `k`).
Integer axes on both dimensions.
"""
function plot_result(r::ExternalVolatilitySVARResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("extvol_reg")
    K = length(r.regime_indices)
    T_total = 0
    for idxs in r.regime_indices
        isempty(idxs) || (T_total = max(T_total, maximum(idxs)))
    end
    membership = fill(NaN, max(T_total, 1))
    for (k, idxs) in enumerate(r.regime_indices)
        for i in idxs
            (1 <= i <= length(membership)) && (membership[i] = k)
        end
    end
    rows = Vector{Pair{String,String}}[]
    for t in 1:length(membership)
        push!(rows, ["x" => _json(t), "reg" => _json(membership[t])])
    end
    data_json = _json_array_of_objects(rows)
    series_json = _series_json(["Regime"], [_PLOT_SERIES[1]]; keys=["reg"])
    js = _render_line_js(id, data_json, series_json; integer_x=true,
                         xlabel="Observation", ylabel="Regime")
    panel = _PanelSpec(id, "Regime Membership ($K regimes)", js)
    ftitle = isempty(title) ? "External-Volatility SVAR — regime membership" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ICASVARResult
# =============================================================================

"""
    plot_result(r::ICASVARResult; view=:mixing, title="", save_path=nothing)

ICA-SVAR diagnostics. Views (plotrule C5):

- `:mixing` (default) — the structural impact / mixing matrix `B₀` (signed) as a
  diverging heatmap with a color-scale legend.
- `:unmixing` — the unmixing matrix `W` as a diverging heatmap.
- `:shocks` — the recovered independent structural shocks as overlaid lines.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::ICASVARResult{T};
                     view::Symbol=:mixing, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B0, 1)
    if view === :mixing
        id = _next_plot_id("ica_b0")
        js = _statid_heatmap_panel(id, r.B0, _statid_var_names(n), _statid_shock_names(n);
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Mixing Matrix (B₀)", js)
        ftitle = isempty(title) ? "ICA-SVAR — mixing matrix ($(r.method))" : title
    elseif view === :unmixing
        id = _next_plot_id("ica_w")
        js = _statid_heatmap_panel(id, r.W, _statid_shock_names(n), _statid_var_names(n);
                                   scale=:diverging, xlabel="Variable", ylabel="Shock",
                                   tip_label="")
        panel = _PanelSpec(id, "Unmixing Matrix (W)", js)
        ftitle = isempty(title) ? "ICA-SVAR — unmixing matrix ($(r.method))" : title
    elseif view === :shocks
        id = _next_plot_id("ica_shk")
        js = _statid_line_panel(id, r.shocks, _statid_shock_names(n); ylabel="Structural shock")
        panel = _PanelSpec(id, "Recovered Structural Shocks", js)
        ftitle = isempty(title) ? "ICA-SVAR — structural shocks ($(r.method))" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :mixing, :unmixing, :shocks"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# NonGaussianMLResult
# =============================================================================

"""
    plot_result(r::NonGaussianMLResult; title="", save_path=nothing)

Draw the maximum-likelihood non-Gaussian SVAR impact matrix `B₀` (signed) as a
diverging heatmap with a color-scale legend. The panel title carries the
likelihood-ratio statistic `LR = 2·(loglik − loglik_gaussian)` (against the Gaussian
model), rounded via `_fmt` (plotrule C9).
"""
function plot_result(r::NonGaussianMLResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B0, 1)
    id = _next_plot_id("ml_b0")
    js = _statid_heatmap_panel(id, r.B0, _statid_var_names(n), _statid_shock_names(n);
                               scale=:diverging, xlabel="Shock", ylabel="Variable",
                               tip_label="")
    lr = max(2 * (r.loglik - r.loglik_gaussian), zero(T))
    ptitle = "Impact Matrix (B₀) — non-Gaussian MLE ($(r.distribution)); LR vs Gaussian = $(_fmt(lr; digits=3))"
    panel = _PanelSpec(id, ptitle, js)
    ftitle = isempty(title) ? "Non-Gaussian ML SVAR — impact matrix" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
