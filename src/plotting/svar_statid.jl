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
- `NonGaussianGMMResult` — `view=:mixing` (`B0` heatmap; same panel helper as ICA).
- `LewisTVVResult` — `view=:mixing` (`B0` heatmap; same panel helper as ICA/GMM).
- `SVSVARResult` — `view=:B` (impact heatmap) / `:volatility` (smoothed
  log-volatility lines).
- `ProxySVARResult` — `view=:B0` (impact heatmap) / `:impact` (identified columns).
- `MaxShareResult` — `view=:Q` (rotation heatmap) / `:eigvals` (criterion eigenvalues).
- `SVARModel` — `view=:B0` (impact ``A^{-1}B`` heatmap).
- `SVECResult` — `view=:B0` (permanent/transitory impact heatmap).

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
_statid_shock_names(r::AbstractNonGaussianSVAR) = r.shock_names
_statid_var_names(r::AbstractNonGaussianSVAR) =
    hasfield(typeof(r), :varnames) ? getfield(r, :varnames) : _statid_var_names(size(r.B0, 1))

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
        js = _statid_heatmap_panel(id, r.B0, _statid_var_names(r), _statid_shock_names(r);
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
    names = _statid_shock_names(r)
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
        js = _statid_heatmap_panel(id, r.B0, _statid_var_names(r), _statid_shock_names(r);
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Mixing Matrix (B₀)", js)
        ftitle = isempty(title) ? "ICA-SVAR — mixing matrix ($(r.method))" : title
    elseif view === :unmixing
        id = _next_plot_id("ica_w")
        js = _statid_heatmap_panel(id, r.W, _statid_shock_names(r), _statid_var_names(r);
                                   scale=:diverging, xlabel="Variable", ylabel="Shock",
                                   tip_label="")
        panel = _PanelSpec(id, "Unmixing Matrix (W)", js)
        ftitle = isempty(title) ? "ICA-SVAR — unmixing matrix ($(r.method))" : title
    elseif view === :shocks
        id = _next_plot_id("ica_shk")
        js = _statid_line_panel(id, r.shocks, _statid_shock_names(r); ylabel="Structural shock")
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
    js = _statid_heatmap_panel(id, r.B0, _statid_var_names(r), _statid_shock_names(r);
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

# =============================================================================
# NonGaussianGMMResult
# =============================================================================

"""
    plot_result(r::NonGaussianGMMResult; view=:mixing, title="", save_path=nothing)

Moment-based GMM SVAR diagnostics. `view=:mixing` (default) draws the structural
impact / mixing matrix `B₀` as a diverging heatmap, reusing the ICA mixing-matrix
panel helper. Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::NonGaussianGMMResult{T};
                     view::Symbol=:mixing, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view === :mixing || throw(ArgumentError("Unknown view :$view. Valid views: :mixing"))
    id = _next_plot_id("gmm_b0")
    js = _statid_heatmap_panel(id, r.B0, r.varnames, r.shock_names;
                               scale=:diverging, xlabel="Shock", ylabel="Variable",
                               tip_label="")
    panel = _PanelSpec(id, "Mixing Matrix (B₀)", js)
    ftitle = isempty(title) ?
             "Moment-based GMM SVAR — mixing matrix ($(r.moments), $(r.weighting))" :
             title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LewisTVVResult
# =============================================================================

"""
    plot_result(r::LewisTVVResult; view=:mixing, title="", save_path=nothing)

Lewis TVV-ID diagnostics. `view=:mixing` (default) draws the structural impact /
mixing matrix `B₀` as a diverging heatmap, reusing the ICA mixing-matrix panel
helper. Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::LewisTVVResult{T};
                     view::Symbol=:mixing, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    view === :mixing || throw(ArgumentError("Unknown view :$view. Valid views: :mixing"))
    id = _next_plot_id("lewis_b0")
    js = _statid_heatmap_panel(id, r.B0, r.varnames, r.shock_names;
                               scale=:diverging, xlabel="Shock", ylabel="Variable",
                               tip_label="")
    panel = _PanelSpec(id, "Mixing Matrix (B₀)", js)
    ftitle = isempty(title) ?
             "Lewis TVV-ID — mixing matrix ($(r.weighting))" :
             title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SVSVARResult
# =============================================================================

"""
    plot_result(r::SVSVARResult; view=:B, title="", save_path=nothing)

SV-SVAR diagnostics. `view=:B` (default) draws the structural impact matrix as a
diverging heatmap; `view=:volatility` draws the smoothed log-volatilities
(`H_smooth`) as overlaid per-shock lines. Unknown `view` throws an
`ArgumentError`.
"""
function plot_result(r::SVSVARResult{T};
                     view::Symbol=:B, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B, 1)
    if view === :B
        id = _next_plot_id("svsvar_b")
        js = _statid_heatmap_panel(id, r.B, r.varnames, r.shock_names;
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Impact Matrix (B)", js)
        ftitle = isempty(title) ? "SV-SVAR — impact matrix" : title
    elseif view === :volatility
        id = _next_plot_id("svsvar_h")
        js = _statid_line_panel(id, r.H_smooth, r.shock_names;
                                ylabel="Smoothed log-volatility")
        panel = _PanelSpec(id, "Smoothed Log-Volatilities", js)
        ftitle = isempty(title) ? "SV-SVAR — smoothed log-volatilities" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :B, :volatility"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ProxySVARResult
# =============================================================================

"""
    plot_result(r::ProxySVARResult; view=:B0, title="", save_path=nothing)

Proxy-SVAR diagnostics. Views:

- `:B0` (default) — structural impact matrix as a diverging heatmap.
- `:impact` — identified columns of `B₀` as a diverging heatmap.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::ProxySVARResult{T};
                     view::Symbol=:B0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :B0
        id = _next_plot_id("proxy_b0")
        js = _statid_heatmap_panel(id, r.B0, r.varnames, r.shock_names;
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Impact Matrix (B₀)", js)
        ftitle = isempty(title) ? "Proxy SVAR — impact matrix" : title
    elseif view === :impact
        idx = [i for (i, s) in enumerate(r.shock_names) if occursin("Proxy", s)]
        isempty(idx) && (idx = collect(1:r.k))
        cols = r.B0[:, idx]
        id = _next_plot_id("proxy_imp")
        js = _statid_heatmap_panel(id, cols, r.varnames, r.shock_names[idx];
                                   scale=:diverging, xlabel="Identified shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Identified Impact Columns", js)
        ftitle = isempty(title) ? "Proxy SVAR — identified impacts" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :B0, :impact"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# MaxShareResult
# =============================================================================

"""
    plot_result(r::MaxShareResult; view=:Q, title="", save_path=nothing)

Max-share diagnostics. Views:

- `:Q` (default) — rotation matrix as a diverging heatmap.
- `:eigvals` — criterion eigenvalues as a one-column heatmap.

Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::MaxShareResult{T};
                     view::Symbol=:Q, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :Q
        id = _next_plot_id("maxshare_q")
        js = _statid_heatmap_panel(id, r.Q, r.varnames, r.shock_names;
                                   scale=:diverging, xlabel="Shock", ylabel="Variable",
                                   tip_label="")
        panel = _PanelSpec(id, "Rotation (Q)", js)
        ftitle = isempty(title) ? "Max-share — rotation" : title
    elseif view === :eigvals
        id = _next_plot_id("maxshare_eig")
        n = length(r.eigvals)
        labels = ["λ$j" for j in 1:n]
        M = reshape(r.eigvals, :, 1)
        js = _statid_heatmap_panel(id, M, labels, ["eigenvalue"];
                                   scale=:sequential, xlabel="", ylabel="Eigenvalue",
                                   tip_label="λ")
        panel = _PanelSpec(id, "Criterion eigenvalues", js)
        ftitle = isempty(title) ? "Max-share — eigenvalues" : title
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :Q, :eigvals"))
    end
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SVARModel (AB-model ML)
# =============================================================================

"""
    plot_result(r::SVARModel; view=:B0, title="", save_path=nothing)

AB-model SVAR. `view=:B0` (default) plots the impact ``A^{-1} B`` as a diverging
heatmap. Unknown `view` throws an `ArgumentError`.
"""
function plot_result(r::SVARModel{T};
                     view::Symbol=:B0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.A, 1)
    view === :B0 || throw(ArgumentError("Unknown view :$view. Valid views: :B0"))
    B0 = r.A \ r.B
    id = _next_plot_id("svar_b0")
    js = _statid_heatmap_panel(id, B0, r.varnames, r.shock_names;
                               scale=:diverging, xlabel="Shock", ylabel="Variable",
                               tip_label="")
    panel = _PanelSpec(id, "Impact Matrix (B₀)", js)
    ftitle = isempty(title) ? "AB-model SVAR — impact matrix" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SVECResult
# =============================================================================

"""
    plot_result(r::SVECResult; view=:B0, title="", save_path=nothing)

Structural VECM. `view=:B0` (default) plots the contemporaneous impact with
permanent (`P*`) then transitory (`T*`) shock columns. Unknown `view` throws
an `ArgumentError`.
"""
function plot_result(r::SVECResult{T};
                     view::Symbol=:B0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n = size(r.B0, 1)
    view === :B0 || throw(ArgumentError("Unknown view :$view. Valid views: :B0"))
    id = _next_plot_id("svec_b0")
    js = _statid_heatmap_panel(id, r.B0, r.vecm.varnames, r.shock_names;
                               scale=:diverging, xlabel="Shock", ylabel="Variable",
                               tip_label="")
    panel = _PanelSpec(id, "Impact Matrix (B₀)", js)
    ftitle = isempty(title) ? "SVEC — impact matrix" : title
    p = _make_plot([panel]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# NGID / SSVARMA Helpers (plotrule A5/A6 — # comments only, no docstrings)
# =============================================================================

# Resolve optional shock selector (`nothing` => up to `cap` shocks, else single resolved index)
function _ngid_select_shocks(shock::Union{Int,String,Nothing},
                             shock_names::Vector{String}; cap::Int=12)
    n = length(shock_names)
    if shock !== nothing
        idx = _resolve_var(shock, shock_names)
        return [idx], n, false
    end
    shown = min(n, cap)
    return collect(1:shown), n, n > cap
end

# Safe PDF evaluation for SVARMAModel marginal families (display parameterization)
function _svarma_shock_pdf(dist::Symbol, x::Float64, dp::Vector{T}) where {T<:AbstractFloat}
    dist === :nonparametric && return NaN
    isempty(dp) && return NaN
    lp = try
        if dist === :student_t && length(dp) >= 1
            _student_t_logpdf(x, Float64(dp[1]))
        elseif dist === :mixture_normal && length(dp) >= 3
            _mixture_normal_logpdf(x, Float64(dp[1]), Float64(dp[2]), Float64(dp[3]))
        elseif dist === :mixture_normal_asym && length(dp) >= 5
            _mixture_normal_logpdf(x, Float64(dp[1]), Float64(dp[2]), Float64(dp[3]),
                                   Float64(dp[4]), Float64(dp[5]))
        elseif (dist === :pearson_iv || dist === :pml) && length(dp) >= 2
            _pearson_iv_logpdf(x, Float64(dp[1]), Float64(dp[2]))
        elseif dist === :skew_normal && length(dp) >= 1
            _lms_skew_normal_logpdf(x, Float64(dp[1]))
        elseif dist === :skewed_t && length(dp) >= 2
            _skewed_t_logpdf(x, Float64(dp[1]), Float64(dp[2]))
        elseif (dist === :ged || dist === :laplace_generalized) && length(dp) >= 1
            _ged_logpdf(x, Float64(dp[1]))
        else
            NaN
        end
    catch
        NaN
    end
    isfinite(lp) ? exp(lp) : NaN
end

# Safe PDF evaluation for LMSSVARResult marginal families (constrained lam vector)
function _lms_shock_pdf(dist::Symbol, x::Float64, dp::Vector{T}) where {T<:AbstractFloat}
    isempty(dp) && return NaN
    lp = try
        Float64(_lms_logpdf_scalar(dist, x, Float64.(dp)))
    catch
        NaN
    end
    isfinite(lp) ? exp(lp) : NaN
end

# Shared Normal Q-Q scatter panel builder across structural shocks
function _ngid_qq_panels(shocks::AbstractMatrix, shock_names::Vector{String},
                         sel::Vector{Int}; dist_labels::Union{Nothing,Vector{String}}=nothing,
                         id_prefix::String="ngid_qq")
    panels = _PanelSpec[]
    for j in sel
        col = @view shocks[:, j]
        has_nonfinite = any(!isfinite, col)
        fin = sort(Float64[Float64(v) for v in col if isfinite(v)])
        nq = length(fin)
        grp = shock_names[j]
        color = _PLOT_SERIES[mod1(j, length(_PLOT_SERIES))]
        rows = Vector{Pair{String,String}}[]
        if nq >= 1
            m = sum(fin) / nq
            sd = nq >= 2 ? sqrt(sum(abs2, fin .- m) / (nq - 1)) : 1.0
            for i in 1:nq
                theo = _norm_quantile((i - 0.5) / nq)
                samp = sd > 0 ? (fin[i] - m) / sd : 0.0
                push!(rows, ["x" => _json(theo), "y" => _json(samp), "group" => _json(grp)])
            end
        end
        if has_nonfinite || nq == 0
            push!(rows, ["x" => "null", "y" => "null", "group" => _json(grp)])
        end
        lo = nq >= 1 ? _norm_quantile(0.5 / nq) : -3.0
        hi = -lo
        overlay = "[{\"x1\":$(_json(lo)),\"y1\":$(_json(lo))," *
                  "\"x2\":$(_json(hi)),\"y2\":$(_json(hi))," *
                  "\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        groups_json = "[{\"name\":$(_json(grp)),\"color\":$(_json(color))}]"
        id = _next_plot_id(id_prefix)
        js = _render_scatter_js(id, _json_array_of_objects(rows), groups_json;
                                line_overlays_json=overlay,
                                xlabel="Theoretical N(0,1) Quantiles",
                                ylabel="Sample Quantiles")
        ptitle = dist_labels === nothing ? grp : "$(grp) ($(dist_labels[j]))"
        push!(panels, _PanelSpec(id, ptitle, js))
    end
    panels
end

# Shared KDE vs Parametric vs N(0,1) density overlay panels
function _ngid_density_line_panels(shocks::AbstractMatrix, shock_names::Vector{String},
                                   distributions::Vector{Symbol}, dist_params::Vector{<:AbstractVector},
                                   sel::Vector{Int}, pdf_eval_fn;
                                   id_prefix::String="ngid_dens")
    panels = _PanelSpec[]
    for j in sel
        col = Float64[Float64(v) for v in @view(shocks[:, j])]
        fin = Float64[v for v in col if isfinite(v)]
        has_nonfinite = length(fin) < length(col)
        id = _next_plot_id(id_prefix)
        dist = distributions[j]
        dp = dist_params[j]
        if length(fin) < 2
            data_json = "[{\"x\":0,\"kde\":null,\"norm\":null}]"
            s_json = _series_json(["Empirical KDE", "N(0,1)"],
                                  [_PLOT_SERIES[1], _PLOT_ALERT];
                                  keys=["kde", "norm"], dash=["", "3,3"])
            js = _render_line_js(id, data_json, s_json;
                                 xlabel=shock_names[j], ylabel="Density")
            push!(panels, _PanelSpec(id, "$(shock_names[j]) — $(dist)", js))
            continue
        end
        bw = std(fin) > 1e-8 ? 0 : 0.5
        xs, kde_vals = _kde_line(fin; bw=bw)
        norm_vals = Float64[pdf(Normal(), xg) for xg in xs]
        fit_vals = Float64[pdf_eval_fn(dist, xg, dp) for xg in xs]
        has_fit = any(isfinite, fit_vals)
        rows = Vector{Pair{String,String}}[]
        for g in eachindex(xs)
            row = Pair{String,String}[
                "x" => _json(xs[g]),
                "kde" => _json(kde_vals[g]),
                "norm" => _json(norm_vals[g]),
            ]
            has_fit && push!(row, "fit" => _json(fit_vals[g]))
            push!(rows, row)
        end
        if has_nonfinite
            # Preserve explicit null trace if sample contained non-finite shocks
            row_na = Pair{String,String}[
                "x" => _json(xs[end] + (xs[end] - xs[1]) * 0.01),
                "kde" => "null",
                "norm" => "null",
            ]
            has_fit && push!(row_na, "fit" => "null")
            push!(rows, row_na)
        end
        s_json = if has_fit
            _series_json(["Empirical KDE", "Fitted ($(dist))", "N(0,1)"],
                         [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_ALERT];
                         keys=["kde", "fit", "norm"], dash=["", "6,3", "3,3"])
        else
            _series_json(["Empirical KDE", "N(0,1)"],
                         [_PLOT_SERIES[1], _PLOT_ALERT];
                         keys=["kde", "norm"], dash=["", "3,3"])
        end
        js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                             xlabel="Standardized shock", ylabel="Density")
        ptitle = has_fit ? "$(shock_names[j]) ($(dist))" : "$(shock_names[j]) (nonparametric)"
        push!(panels, _PanelSpec(id, ptitle, js))
    end
    panels
end

# Extract (τ, δ) coordinates for GMR (2020) Fig. 6 stability diagram across arbitrary (n, q)
function _svarma_regime_coords(m::SVARMAModel{T}) where {T}
    n = size(m.C, 1)
    m.q == 0 && return (0.0, 0.0, " (q = 0: pure SVAR)")
    if any(!isfinite, m.Theta) || any(!isfinite, m.ma_eig_re) || any(!isfinite, m.ma_eig_im)
        return (NaN, NaN, "")
    end
    if n == 2 && m.q == 1
        A_ma = -@view(m.Theta[:, :, 1])
        return (Float64(tr(A_ma)), Float64(det(A_ma)), "")
    end
    eigs = ComplexF64.(m.ma_eig_re, m.ma_eig_im)
    isempty(eigs) && return (0.0, 0.0, "")
    if length(eigs) == 1
        return (real(eigs[1]), 0.0, " (n=$(n), q=$(m.q): univariate root)")
    end
    ord = sortperm(eigs; by=abs, rev=true)
    e1 = eigs[ord[1]]
    if abs(imag(e1)) > 1e-8
        return (2.0 * real(e1), abs2(e1), " (n=$(n), q=$(m.q): leading conjugate pair)")
    end
    e2 = eigs[ord[2]]
    if abs(imag(e2)) > 1e-8 && abs(e2) >= abs(e1) - 1e-8
        return (2.0 * real(e2), abs2(e2), " (n=$(n), q=$(m.q): leading conjugate pair)")
    end
    # Pick second real eigenvalue if available; otherwise pair e1 with second entry
    real_idx = findfirst(k -> k != ord[1] && abs(imag(eigs[k])) <= 1e-8, ord)
    e2_use = real_idx !== nothing ? eigs[ord[real_idx]] : e2
    return (real(e1 + e2_use), real(e1 * e2_use), " (n=$(n), q=$(m.q): leading eigenvalue pair)")
end

function _plot_svarma_regime_map(m::SVARMAModel{T}; title::String="", ncols::Int=0) where {T}
    tau, delta, note_dim = _svarma_regime_coords(m)
    t_lim = max(3.2, isfinite(tau) ? abs(tau) + 0.8 : 3.2)

    # Stability boundaries in (τ, δ) = (tr(-Θ₁), det(-Θ₁)):
    # 1 - τ + δ = 0 (δ = τ - 1), 1 + τ + δ = 0 (δ = -τ - 1), δ = 1 on |τ| ≤ 2
    lines = [
        "{\"x1\":$(_json(-t_lim)),\"y1\":$(_json(-t_lim - 1.0)),\"x2\":$(_json(t_lim)),\"y2\":$(_json(t_lim - 1.0)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}",
        "{\"x1\":$(_json(-t_lim)),\"y1\":$(_json(t_lim - 1.0)),\"x2\":$(_json(t_lim)),\"y2\":$(_json(-t_lim - 1.0)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}",
        "{\"x1\":-2,\"y1\":1,\"x2\":2,\"y2\":1,\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"\"}",
    ]
    line_overlays = "[" * join(lines, ",") * "]"

    # Discriminant parabola τ² = 4δ (δ = τ²/4): separates real vs complex roots
    ts = range(-t_lim, t_lim; length=81)
    ppts = ["{\"x\":$(_json(tv)),\"y\":$(_json(tv^2 / 4.0))}" for tv in ts]
    curve_overlays = "[{\"points\":[" * join(ppts, ",") * "],\"color\":\"#666\",\"dash\":\"5,3\",\"band\":false}]"

    # Reference axes at 0 plus y = d_lo / d_hi anchors via zero lines
    refs = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#bbb\",\"dash\":\"2,2\"}," *
           "{\"value\":0,\"axis\":\"y\",\"color\":\"#bbb\",\"dash\":\"2,2\"}]"

    grp = "Region $(m.gmr_region) ($(m.regime))"
    pt_color = m.regime === :fundamental ? _PLOT_SERIES[1] :
               m.regime === :unit_root ? _PLOT_ALERT : _PLOT_SERIES[4]
    # Keep bounds well-formed even when point is NaN
    data_rows = Vector{Pair{String,String}}[
        ["x" => _json(tau), "y" => _json(delta), "group" => _json(grp)]
    ]
    groups_json = "[{\"name\":$(_json(grp)),\"color\":$(_json(pt_color))}]"
    id = _next_plot_id("svarma_gmr")
    js = _render_scatter_js(id, _json_array_of_objects(data_rows), groups_json;
                            ref_lines_json=refs,
                            line_overlays_json=line_overlays,
                            curve_overlays_json=curve_overlays,
                            xlabel="Trace τ = tr(-Θ₁)",
                            ylabel="Determinant δ = det(-Θ₁)")
    tau_s = isfinite(tau) ? _fmt(tau; digits=3) : "n/a"
    del_s = isfinite(delta) ? _fmt(delta; digits=3) : "n/a"
    ptitle = "GMR Fig. 6 Regime Map — Region $(m.gmr_region) (τ=$(tau_s), δ=$(del_s))$(note_dim)"
    ft = isempty(title) ? "SVARMA($(m.p), $(m.q)) — Fundamentalness Regime Map" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
end

function _plot_svarma_shock_densities(m::SVARMAModel{T};
                                      shock::Union{Int,String,Nothing}=nothing,
                                      title::String="", ncols::Int=0) where {T}
    sel, n, capped = _ngid_select_shocks(shock, m.shock_names)
    panels = _ngid_density_line_panels(m.shocks, m.shock_names, m.distributions,
                                       m.dist_params, sel, _svarma_shock_pdf;
                                       id_prefix="svarma_dens")
    note = _cap_note("shocks", length(sel), n, "shock")
    ft = isempty(title) ? "SVARMA($(m.p), $(m.q)) — Shock Densities vs N(0,1)" : title
    _make_plot(panels; title=ft, ncols=ncols, note=note)
end

function _plot_svarma_roots(m::SVARMAModel{T}; title::String="", ncols::Int=0) where {T}
    n = size(m.C, 1)
    ar_eigs = ComplexF64[]
    if m.p > 0
        if any(!isfinite, m.Phi)
            ar_eigs = ComplexF64[complex(NaN, NaN)]
        else
            np = n * m.p
            F_ar = zeros(Float64, np, np)
            for k in 1:m.p
                F_ar[1:n, ((k - 1) * n + 1):(k * n)] .= Float64.(@view(m.Phi[:, :, k]))
            end
            if m.p > 1
                for i in 1:((m.p - 1) * n)
                    F_ar[n + i, i] = 1.0
                end
            end
            ar_eigs = ComplexF64.(eigvals(F_ar))
        end
    end
    ma_eigs = ComplexF64.(Float64.(m.ma_eig_re), Float64.(m.ma_eig_im))

    rows = Vector{Pair{String,String}}[]
    used = Set{String}()
    for z in ar_eigs
        mag = abs(z)
        grp = (isfinite(mag) && mag < 1.0 - 1e-8) ? "AR stable (|λ|<1)" : "AR unstable (|λ|≥1)"
        push!(used, grp)
        push!(rows, ["x" => _json(real(z)), "y" => _json(imag(z)), "group" => _json(grp)])
    end
    for z in ma_eigs
        mag = abs(z)
        grp = (isfinite(mag) && mag < 1.0 - 1e-8) ? "MA fundamental (|λ|<1)" :
              "MA non-fundamental (|λ|≥1)"
        push!(used, grp)
        push!(rows, ["x" => _json(real(z)), "y" => _json(imag(z)), "group" => _json(grp)])
    end

    order = [
        "AR stable (|λ|<1)" => _PLOT_SERIES[1],
        "AR unstable (|λ|≥1)" => _PLOT_ALERT,
        "MA fundamental (|λ|<1)" => _PLOT_SERIES[2],
        "MA non-fundamental (|λ|≥1)" => _PLOT_SERIES[4],
    ]
    gparts = String[]
    for (nm, col) in order
        nm in used || continue
        push!(gparts, "{\"name\":$(_json(nm)),\"color\":$(_json(col))}")
    end
    isempty(gparts) && push!(gparts, "{\"name\":\"Roots\",\"color\":$(_json(_PLOT_SERIES[1]))}")
    groups_json = "[" * join(gparts, ",") * "]"
    shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    refs = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#ccc\",\"dash\":\"2,2\"}," *
           "{\"value\":0,\"axis\":\"y\",\"color\":\"#ccc\",\"dash\":\"2,2\"}]"
    id = _next_plot_id("svarma_roots")
    js = _render_scatter_js(id, _json_array_of_objects(rows), groups_json;
                            ref_lines_json=refs, ref_shapes_json=shapes,
                            xlabel="Real", ylabel="Imaginary")
    ptitle = "AR & MA Companion Eigenvalues — $(m.regime) (nonfund=$(m.nonfundamental_order))"
    ft = isempty(title) ? "SVARMA($(m.p), $(m.q)) — Companion Roots" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
end

# =============================================================================
# SVARMAModel
# =============================================================================

"""
    plot_result(m::SVARMAModel; view=:regime_map, shock=nothing, ncols=0,
                title="", save_path=nothing)

Visualize a Strong Structural VARMA(p, q) model (`SVARMAModel`). `view` selects:

- `:regime_map` (default) — Gouriéroux, Monfort & Renne (2020, Fig. 6) 5-region
  stability diagram in the ``(\\tau, \\delta) = (\\operatorname{tr}(-\\Theta_1), \\det(-\\Theta_1))``
  plane, drawing the parabola ``\\tau^2 = 4\\delta`` (real vs. complex roots), unit-root
  boundary lines ``\\delta = \\tau - 1``, ``\\delta = -\\tau - 1``, and ``\\delta = 1``
  (for ``|\\tau| \\le 2``), together with the estimated point. For ``n \\ne 2`` or
  ``q > 1``, projects onto the leading companion-eigenvalue pair; for ``q = 0`` places
  the point at ``(0, 0)``.
- `:shock_densities` — GMR (2020, Fig. 4) empirical KDE of standardized structural
  shocks overlaid against the fitted parametric family density (when estimated via
  `:ml`/`:pml`) and a standard normal ``N(0, 1)`` benchmark. `shock` selects one shock
  by index or name (`nothing` plots up to 12 shocks).
- `:roots` — AR and MA companion-matrix eigenvalues on the complex plane with unit
  circle reference.

Passing `shock` to `:regime_map` or `:roots`, or an unknown `view`, throws
`ArgumentError`.
"""
function plot_result(m::SVARMAModel{T};
                     view::Symbol=:regime_map,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    p = if view === :regime_map
        shock === nothing || throw(ArgumentError("shock applies only to view=:shock_densities"))
        _plot_svarma_regime_map(m; title=title, ncols=ncols)
    elseif view === :shock_densities
        _plot_svarma_shock_densities(m; shock=shock, title=title, ncols=ncols)
    elseif view === :roots
        shock === nothing || throw(ArgumentError("shock applies only to view=:shock_densities"))
        _plot_svarma_roots(m; title=title, ncols=ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :regime_map, :shock_densities, :roots"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LMSSVARResult
# =============================================================================

function _plot_lms_qq(r::LMSSVARResult{T};
                      shock::Union{Int,String,Nothing}=nothing,
                      title::String="", ncols::Int=0) where {T}
    sel, n, _ = _ngid_select_shocks(shock, r.shock_names)
    dlabels = String[string(d) for d in r.distributions]
    panels = _ngid_qq_panels(r.shocks, r.shock_names, sel;
                             dist_labels=dlabels, id_prefix="lms_qq")
    note = _cap_note("shocks", length(sel), n, "shock")
    ft = isempty(title) ? "LMS Non-Gaussian SVAR — Structural Shock Q–Q" : title
    _make_plot(panels; title=ft, ncols=ncols, note=note)
end

function _plot_lms_densities(r::LMSSVARResult{T};
                             shock::Union{Int,String,Nothing}=nothing,
                             title::String="", ncols::Int=0) where {T}
    sel, n, _ = _ngid_select_shocks(shock, r.shock_names)
    panels = _ngid_density_line_panels(r.shocks, r.shock_names, r.distributions,
                                       r.dist_params, sel, _lms_shock_pdf;
                                       id_prefix="lms_dens")
    note = _cap_note("shocks", length(sel), n, "shock")
    ft = isempty(title) ? "LMS Non-Gaussian SVAR — Standardized Shock Densities" : title
    _make_plot(panels; title=ft, ncols=ncols, note=note)
end

"""
    plot_result(r::LMSSVARResult; view=:qq, shock=nothing, ncols=0,
                title="", save_path=nothing)

Visualize standardized structural shocks from Lanne–Meitz–Saikkonen (2017)
non-Gaussian SVAR estimation. `view` selects:

- `:qq` (default) — normal Q–Q scatter plots of standardized structural shocks with a
  45° reference line across shocks.
- `:densities` — empirical KDE compared with the fitted marginal distribution density
  and a standard normal ``N(0, 1)`` reference curve.

`shock` selects one shock by `Int` index or `String` name (`nothing` plots up to 12
shocks). Unknown `view` throws `ArgumentError`.
"""
function plot_result(r::LMSSVARResult{T};
                     view::Symbol=:qq,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    p = if view === :qq
        _plot_lms_qq(r; shock=shock, title=title, ncols=ncols)
    elseif view === :densities
        _plot_lms_densities(r; shock=shock, title=title, ncols=ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :qq, :densities"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LanneLuotoGMMResult
# =============================================================================

function _plot_llgmm_qq(r::LanneLuotoGMMResult{T};
                        shock::Union{Int,String,Nothing}=nothing,
                        title::String="", ncols::Int=0) where {T}
    sel, n, _ = _ngid_select_shocks(shock, r.shock_names)
    panels = _ngid_qq_panels(r.shocks, r.shock_names, sel; id_prefix="llgmm_qq")
    note = _cap_note("shocks", length(sel), n, "shock")
    ft = isempty(title) ? "Lanne–Luoto GMM SVAR — Structural Shock Q–Q" : title
    _make_plot(panels; title=ft, ncols=ncols, note=note)
end

function _plot_llgmm_densities(r::LanneLuotoGMMResult{T};
                               shock::Union{Int,String,Nothing}=nothing,
                               title::String="", ncols::Int=0) where {T}
    sel, n, _ = _ngid_select_shocks(shock, r.shock_names)
    panels = _PanelSpec[]
    for j in sel
        col = Float64[Float64(v) for v in @view(r.shocks[:, j])]
        fin = Float64[v for v in col if isfinite(v)]
        id = _next_plot_id("llgmm_dens")
        if isempty(fin)
            bins_json = "[{\"x0\":-0.5,\"x1\":0.5,\"y\":null}]"
            dens_json = "[]"
        else
            bins_json = _histogram_bins(fin; density=true)
            bw = length(fin) >= 2 && std(fin) > 1e-8 ? 0 : 0.5
            dens_json = length(fin) >= 2 ? _kde_line_json(fin; bw=bw) : "[]"
        end
        s_json = _series_json(["Histogram", "Empirical KDE"],
                              [_PLOT_SERIES[1], _PLOT_SERIES[2]]; keys=["bar", "d"])
        js = _render_histogram_js(id, bins_json, s_json; density_json=dens_json,
                                  xlabel=r.shock_names[j], ylabel="Density")
        push!(panels, _PanelSpec(id, r.shock_names[j], js))
    end
    note = _cap_note("shocks", length(sel), n, "shock")
    ft = isempty(title) ? "Lanne–Luoto GMM SVAR — Shock Densities ($(r.weighting))" : title
    _make_plot(panels; title=ft, ncols=ncols, note=note)
end

"""
    plot_result(r::LanneLuotoGMMResult; view=:qq, shock=nothing, ncols=0,
                title="", save_path=nothing)

Visualize structural shocks from Lanne–Luoto (2021) non-Gaussian GMM SVAR estimation.
`view` selects:

- `:qq` (default) — normal Q–Q scatter plots of structural shocks with a 45° reference
  line across shocks.
- `:densities` — density-scaled shock histograms with empirical KDE overlays.

`shock` selects one shock by `Int` index or `String` name (`nothing` plots up to 12
shocks). Unknown `view` throws `ArgumentError`.
"""
function plot_result(r::LanneLuotoGMMResult{T};
                     view::Symbol=:qq,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    p = if view === :qq
        _plot_llgmm_qq(r; shock=shock, title=title, ncols=ncols)
    elseif view === :densities
        _plot_llgmm_densities(r; shock=shock, title=title, ncols=ncols)
    else
        throw(ArgumentError("Unknown view :$view. Valid views: :qq, :densities"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# NonGaussianDiagnostics
# =============================================================================

"""
    plot_result(d::NonGaussianDiagnostics; level=5, ncols=0, title="",
                save_path=nothing)

Visualize LMS (2017, §5.2) non-Gaussian SVAR residual and identification-strength
diagnostics as a three-panel figure:

1. Jarque–Bera test statistics (per shock and joint multivariate) against their
   ``\\chi^2`` critical value at `level`% significance.
2. Per-shock serial-correlation and independence p-values (Ljung–Box on levels,
   Ljung–Box on squares, and BDS(2)) against `α = level/100`.
3. Non-Gaussianity (Holm-adjusted JB) and system-wide independence p-values
   (Multivariate JB, Portmanteau `Q(lags)`, and cross-moment independence) against `α`.
"""
function plot_result(d::NonGaussianDiagnostics{T};
                     level::Int=5,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    0 < level < 100 || throw(ArgumentError("level must be in (0, 100), got $level"))
    alpha = level / 100.0
    n = length(d.shock_names)
    cap = 12
    shown = min(n, cap)
    snames = d.shock_names[1:shown]

    # Panel 1: Jarque-Bera statistics vs level% chi-square critical value
    jb_cats = vcat(snames, ["Multivariate JB"])
    jb_vals = vcat(Float64[Float64(d.jb_stats[i]) for i in 1:shown],
                   [Float64(d.multivariate_jb_stat)])
    cv_uni = quantile(Chisq(2), 1.0 - alpha)
    cv_mul = quantile(Chisq(max(2 * n, 1)), 1.0 - alpha)
    jb_cvs = vcat(fill(cv_uni, shown), [cv_mul])
    rows1 = Vector{Pair{String,String}}[]
    for i in eachindex(jb_cats)
        push!(rows1, ["x" => _json(jb_cats[i]),
                      "stat" => _json(jb_vals[i]),
                      "cv" => _json(jb_cvs[i])])
    end
    s1 = _series_json(["JB Statistic", "$(level)% χ² CV"],
                      [_PLOT_SERIES[1], _PLOT_ALERT]; keys=["stat", "cv"])
    id1 = _next_plot_id("ngdiag_jb")
    js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                         mode="grouped", orientation="h",
                         xlabel="Jarque–Bera statistic", ylabel="")
    id_note = d.weak_id_warning ? "WARNING: ≥2 Gaussian shocks" : "Non-Gaussianity supported"
    p1 = _PanelSpec(id1, "Jarque–Bera Non-Gaussianity ($(id_note))", js1)

    # Panel 2: Per-shock Ljung-Box (levels, squares) & BDS(2) p-values vs alpha
    rows2 = Vector{Pair{String,String}}[]
    for i in 1:shown
        push!(rows2, [
            "x" => _json(snames[i]),
            "lbl" => _json(Float64(d.lb_levels_pvals[i])),
            "lbs" => _json(Float64(d.lb_squares_pvals[i])),
            "bds" => _json(Float64(d.bds_pvals[i])),
            "alp" => _json(alpha),
        ])
    end
    s2 = _series_json(["LB Levels", "LB Squares", "BDS(2)", "α = $(_fmt(alpha; digits=2))"],
                      [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[3], _PLOT_ALERT];
                      keys=["lbl", "lbs", "bds", "alp"])
    id2 = _next_plot_id("ngdiag_dep")
    js2 = _render_bar_js(id2, _json_array_of_objects(rows2), s2;
                         mode="grouped", orientation="h",
                         xlabel="p-value", ylabel="")
    p2 = _PanelSpec(id2, "Serial & BDS Independence (lags=$(d.lags))", js2)

    # Panel 3: Holm JB + system-wide test p-values vs alpha
    sys_cats = vcat(["JB (Holm): $(nm)" for nm in snames],
                    ["Multivariate JB", "Portmanteau Q($(d.lags))", "Cross-Moment Indep."])
    sys_pvals = vcat(Float64[Float64(d.jb_pvals_holm[i]) for i in 1:shown],
                     [Float64(d.multivariate_jb_pval),
                      Float64(d.portmanteau_pval),
                      Float64(d.cross_moment_pval)])
    rows3 = Vector{Pair{String,String}}[]
    for i in eachindex(sys_cats)
        push!(rows3, ["x" => _json(sys_cats[i]),
                      "pval" => _json(sys_pvals[i]),
                      "alp" => _json(alpha)])
    end
    s3 = _series_json(["p-value", "α = $(_fmt(alpha; digits=2))"],
                      [_PLOT_SERIES[1], _PLOT_ALERT]; keys=["pval", "alp"])
    id3 = _next_plot_id("ngdiag_sys")
    js3 = _render_bar_js(id3, _json_array_of_objects(rows3), s3;
                         mode="grouped", orientation="h",
                         xlabel="p-value", ylabel="")
    p3 = _PanelSpec(id3, "JB (Holm) & System Diagnostics", js3)

    note = _cap_note("shocks", shown, n, "ncols")
    ft = isempty(title) ?
         "Non-Gaussian SVAR Diagnostics (n=$(d.nvars), T=$(d.nobs), lags=$(d.lags))" : title
    res = _make_plot([p1, p2, p3]; title=ft, ncols=ncols, note=note)
    save_path !== nothing && save_plot(res, save_path)
    res
end

# =============================================================================
# SVARHypothesisTest
# =============================================================================

"""
    plot_result(r::SVARHypothesisTest; level=5, ncols=0, title="",
                save_path=nothing)

Visualize a structural SVAR hypothesis test (`SVARHypothesisTest`) across three panels:

1. Asymptotic ``\\chi^2(\\text{df})`` reference density with vertical markers at the
   `level`% critical value and the observed test statistic.
2. Observed test statistic compared against 1%, 5%, and 10% ``\\chi^2(\\text{df})``
   critical values.
3. Asymptotic (and bootstrap, when `n_boot > 0`) p-values against `α = level/100`.
"""
function plot_result(r::SVARHypothesisTest{T};
                     level::Int=5,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    0 < level < 100 || throw(ArgumentError("level must be in (0, 100), got $level"))
    alpha = level / 100.0
    df_eff = max(r.df, 1)
    d_chi = Chisq(df_eff)
    cv_lvl = quantile(d_chi, 1.0 - alpha)
    stat_f = Float64(r.statistic)

    # Panel 1: Chi-square reference PDF with observed stat & critical value markers
    x_hi = max(quantile(d_chi, 0.99), cv_lvl * 1.25,
               isfinite(stat_f) ? stat_f * 1.15 : 1.0, 4.0)
    x_lo = df_eff == 1 ? 0.05 : 0.0
    xs = collect(range(x_lo, x_hi; length=140))
    rows1 = Vector{Pair{String,String}}[]
    for xv in xs
        yv = pdf(d_chi, xv)
        push!(rows1, ["x" => _json(xv), "chi2" => _json(isfinite(yv) ? yv : NaN)])
    end
    refs1 = String["{\"value\":$(_json(cv_lvl)),\"axis\":\"x\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}"]
    isfinite(stat_f) && push!(refs1,
        "{\"value\":$(_json(stat_f)),\"axis\":\"x\",\"color\":$(_json(_PLOT_SERIES[1])),\"dash\":\"\"}")
    s1 = _series_json(["χ²($(df_eff)) PDF"], [_PLOT_SERIES[2]]; keys=["chi2"])
    id1 = _next_plot_id("svarhyp_pdf")
    js1 = _render_line_js(id1, _json_array_of_objects(rows1), s1;
                          ref_lines_json="[" * join(refs1, ",") * "]",
                          xlabel="Test statistic", ylabel="Density")
    stat_s = isfinite(stat_f) ? _fmt(stat_f; digits=3) : "n/a"
    p1 = _PanelSpec(id1, "χ²($(df_eff)) Reference (stat = $(stat_s), $(level)% CV = $(_fmt(cv_lvl; digits=3)))", js1)

    # Panel 2: Statistic vs 1%/5%/10% critical values
    cats2 = ["Observed Stat", "10% CV", "5% CV", "1% CV"]
    vals2 = [stat_f, quantile(d_chi, 0.90), quantile(d_chi, 0.95), quantile(d_chi, 0.99)]
    rows2 = Vector{Pair{String,String}}[]
    for i in eachindex(cats2)
        push!(rows2, ["x" => _json(cats2[i]), "v" => _json(vals2[i])])
    end
    s2 = _series_json(["Value"], [_PLOT_SERIES[1]]; keys=["v"])
    id2 = _next_plot_id("svarhyp_cv")
    js2 = _render_bar_js(id2, _json_array_of_objects(rows2), s2;
                         mode="grouped", orientation="h",
                         xlabel="Statistic / critical value", ylabel="")
    p2 = _PanelSpec(id2, "Statistic vs χ²($(df_eff)) Critical Values", js2)

    # Panel 3: Asymptotic & Bootstrap p-values vs alpha
    p_cats = String["Asymptotic p-value"]
    p_vals = Float64[Float64(r.pvalue)]
    if r.bootstrap_pvalue !== nothing
        push!(p_cats, "Bootstrap ($(r.bootstrap_scheme), B=$(r.n_boot))")
        push!(p_vals, Float64(r.bootstrap_pvalue))
    end
    rows3 = Vector{Pair{String,String}}[]
    for i in eachindex(p_cats)
        push!(rows3, ["x" => _json(p_cats[i]), "pval" => _json(p_vals[i]), "alp" => _json(alpha)])
    end
    s3 = _series_json(["p-value", "α = $(_fmt(alpha; digits=2))"],
                      [_PLOT_SERIES[1], _PLOT_ALERT]; keys=["pval", "alp"])
    id3 = _next_plot_id("svarhyp_pv")
    js3 = _render_bar_js(id3, _json_array_of_objects(rows3), s3;
                         mode="grouped", orientation="h",
                         xlabel="p-value", ylabel="")
    eff_p = r.bootstrap_pvalue !== nothing ? Float64(r.bootstrap_pvalue) : Float64(r.pvalue)
    rej = isfinite(eff_p) ? (eff_p < alpha ? "reject H₀" : "fail to reject H₀") : "n/a"
    p3 = _PanelSpec(id3, "P-Values vs α = $(_fmt(alpha; digits=2)) ($(rej))", js3)

    tlab = r.test_type === :lr ? "LR" :
           r.test_type === :lm ? "LM" :
           r.test_type === :gmm_lr ? "GMM-LR" : "Wald"
    ft = isempty(title) ? "SVAR Hypothesis Test ($(tlab)) — $(r.description)" : title
    res = _make_plot([p1, p2, p3]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(res, save_path)
    res
end

# =============================================================================
# GMMMomentSelectionResult
# =============================================================================

"""
    plot_result(r::GMMMomentSelectionResult; max_candidates=15, ncols=0,
                title="", save_path=nothing)

Visualize Andrews (1999) MSC and Hall, Inoue, Jana & Shin (2007) RMSC co-kurtosis
moment selection across evaluated candidate specifications:

1. Stage-1 Andrews MSC scores across candidate moment sets.
2. Stage-2 Hall et al. RMSC scores across candidate moment sets (★ marks selected set).
3. Hansen `J`-test p-values across candidates against the 5% significance threshold.

When `candidates` exceeds `max_candidates` (default 15), displays the top candidates
(ranked by RMSC, always retaining the chosen specification) with a figure note.
"""
function plot_result(r::GMMMomentSelectionResult{T};
                     max_candidates::Int=15,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    max_candidates >= 1 || throw(ArgumentError("max_candidates must be ≥ 1, got $max_candidates"))
    cands = r.candidates
    if isempty(cands)
        id = _next_plot_id("gmmms_sum")
        rows = Vector{Pair{String,String}}[
            ["x" => _json("MSC score"), "v" => _json(Float64(r.msc_score))],
            ["x" => _json("RMSC score"), "v" => _json(Float64(r.rmsc_score))],
            ["x" => _json("Hansen J-stat"), "v" => _json(Float64(r.J_stat))],
        ]
        s_json = _series_json(["Score"], [_PLOT_SERIES[1]]; keys=["v"])
        js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                            mode="grouped", orientation="h",
                            xlabel="Value", ylabel="")
        ptitle = "Selected: $(length(r.asym_pairs)) asym, $(length(r.sym_pairs)) sym (J p=$(_fmt(r.J_pvalue; digits=3)))"
        ft = isempty(title) ? "GMM Co-Kurtosis Moment Selection ($(r.method))" : title
        res = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
        save_path !== nothing && save_plot(res, save_path)
        return res
    end

    total = length(cands)
    sel_idx = findfirst(c -> c.asym_pairs == r.asym_pairs && c.sym_pairs == r.sym_pairs, cands)
    ord = sortperm(eachindex(cands); by=i -> (
        !(i == sel_idx),
        isfinite(cands[i].rmsc) ? 0 : 1,
        isfinite(cands[i].rmsc) ? Float64(cands[i].rmsc) : Inf,
        isfinite(cands[i].msc) ? Float64(cands[i].msc) : Inf,
    ))
    shown = min(total, max_candidates)
    pick = ord[1:shown]

    labels = String[]
    msc_vals = Float64[]
    rmsc_vals = Float64[]
    jp_vals = Float64[]
    for idx in pick
        c = cands[idx]
        star = (idx == sel_idx) ? "★ " : ""
        push!(labels, "$(star)#$(idx) (a=$(length(c.asym_pairs)),s=$(length(c.sym_pairs)),q=$(c.n_moments))")
        push!(msc_vals, Float64(c.msc))
        push!(rmsc_vals, Float64(c.rmsc))
        push!(jp_vals, Float64(c.J_pvalue))
    end

    rows1 = Vector{Pair{String,String}}[
        ["x" => _json(labels[i]), "msc" => _json(msc_vals[i])] for i in eachindex(labels)
    ]
    s1 = _series_json(["MSC"], [_PLOT_SERIES[1]]; keys=["msc"])
    id1 = _next_plot_id("gmmms_msc")
    js1 = _render_bar_js(id1, _json_array_of_objects(rows1), s1;
                         mode="grouped", orientation="h",
                         xlabel="Andrews MSC", ylabel="")
    msc_s = isfinite(r.msc_score) ? _fmt(r.msc_score; digits=3) : "n/a"
    p1 = _PanelSpec(id1, "Stage-1 Andrews MSC (best = $(msc_s))", js1)

    rows2 = Vector{Pair{String,String}}[
        ["x" => _json(labels[i]), "rmsc" => _json(rmsc_vals[i])] for i in eachindex(labels)
    ]
    s2 = _series_json(["RMSC"], [_PLOT_SERIES[2]]; keys=["rmsc"])
    id2 = _next_plot_id("gmmms_rmsc")
    js2 = _render_bar_js(id2, _json_array_of_objects(rows2), s2;
                         mode="grouped", orientation="h",
                         xlabel="Hall et al. RMSC", ylabel="")
    rmsc_s = isfinite(r.rmsc_score) ? _fmt(r.rmsc_score; digits=3) : "n/a"
    p2 = _PanelSpec(id2, "Stage-2 Hall et al. RMSC (best = $(rmsc_s))", js2)

    rows3 = Vector{Pair{String,String}}[
        ["x" => _json(labels[i]), "jp" => _json(jp_vals[i]), "alp" => _json(0.05)]
        for i in eachindex(labels)
    ]
    s3 = _series_json(["J p-value", "α = 0.05"], [_PLOT_SERIES[3], _PLOT_ALERT]; keys=["jp", "alp"])
    id3 = _next_plot_id("gmmms_jp")
    js3 = _render_bar_js(id3, _json_array_of_objects(rows3), s3;
                         mode="grouped", orientation="h",
                         xlabel="Hansen J p-value", ylabel="")
    jp_s = isfinite(r.J_pvalue) ? _fmt(r.J_pvalue; digits=3) : "n/a"
    p3 = _PanelSpec(id3, "Overidentification J p-values (selected p = $(jp_s))", js3)

    note = _cap_note("candidates", shown, total, "max_candidates")
    ft = isempty(title) ?
         "GMM Co-Kurtosis Moment Selection ($(r.method), q=$(r.n_moments), df=$(r.df))" : title
    res = _make_plot([p1, p2, p3]; title=ft, ncols=ncols, note=note)
    save_path !== nothing && save_plot(res, save_path)
    res
end

# =============================================================================
# FundamentalnessTestResult
# =============================================================================

"""
    plot_result(r::FundamentalnessTestResult; level=5, ncols=0, title="",
                save_path=nothing)

Visualize the Chen, Choi & Escanciano (2017) higher-moment fundamentalness test:

1. Per-lag higher-moment portmanteau statistics across lags ``1\\dots m_{\\max}``.
2. Per-lag multiplier-bootstrap p-values across lags ``1\\dots m_{\\max}`` with a
   horizontal reference line at `α = level/100` (default 5%).
3. Moment-category (`:skew`, `:kurt`) and omnibus bootstrap p-values compared with `α`.
"""
function plot_result(r::FundamentalnessTestResult{T};
                     level::Int=5,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    0 < level < 100 || throw(ArgumentError("level must be in (0, 100), got $level"))
    alpha = level / 100.0
    lags = collect(1:r.max_lag)

    # Panel 1: Per-lag statistics
    rows1 = Vector{Pair{String,String}}[
        ["x" => _json(lags[i]),
         "y" => _json(i <= length(r.lag_statistics) ? Float64(r.lag_statistics[i]) : NaN)]
        for i in eachindex(lags)
    ]
    id1 = _next_plot_id("fund_stat")
    js1 = _render_vbar_js(id1, _json_array_of_objects(rows1);
                          bar_color=_PLOT_SERIES[1], integer_x=true,
                          xlabel="Lag k", ylabel="Statistic")
    p1 = _PanelSpec(id1, "Per-Lag Higher-Moment Statistics", js1)

    # Panel 2: Per-lag bootstrap p-values + alpha threshold line
    rows2 = Vector{Pair{String,String}}[
        ["x" => _json(lags[i]),
         "y" => _json(i <= length(r.lag_pvalues) ? Float64(r.lag_pvalues[i]) : NaN)]
        for i in eachindex(lags)
    ]
    refs2 = "[{\"value\":$(_json(alpha)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}]"
    id2 = _next_plot_id("fund_pval")
    js2 = _render_vbar_js(id2, _json_array_of_objects(rows2);
                          bar_color=_PLOT_SERIES[2], ref_lines_json=refs2,
                          integer_x=true, xlabel="Lag k", ylabel="Bootstrap p-value")
    p2 = _PanelSpec(id2, "Per-Lag P-Values (α = $(_fmt(alpha; digits=2)))", js2)

    # Panel 3: Moment-category & omnibus p-values vs alpha
    m_cats = String["Omnibus"]
    m_pvals = Float64[Float64(r.pvalue)]
    for m in r.moments
        if haskey(r.moment_pvalues, m)
            push!(m_cats, "Moment :$(m)")
            push!(m_pvals, Float64(r.moment_pvalues[m]))
        end
    end
    rows3 = Vector{Pair{String,String}}[
        ["x" => _json(m_cats[i]), "pval" => _json(m_pvals[i]), "alp" => _json(alpha)]
        for i in eachindex(m_cats)
    ]
    s3 = _series_json(["p-value", "α = $(_fmt(alpha; digits=2))"],
                      [_PLOT_SERIES[1], _PLOT_ALERT]; keys=["pval", "alp"])
    id3 = _next_plot_id("fund_mom")
    js3 = _render_bar_js(id3, _json_array_of_objects(rows3), s3;
                         mode="grouped", orientation="h",
                         xlabel="p-value", ylabel="")
    pv_s = isfinite(r.pvalue) ? _fmt(r.pvalue; digits=3) : "n/a"
    p3 = _PanelSpec(id3, "Moment Categories & Omnibus (p = $(pv_s))", js3)

    stat_s = isfinite(r.statistic) ? _fmt(r.statistic; digits=3) : "n/a"
    ft = isempty(title) ?
         "CCE Fundamentalness Test ($(uppercase(string(r.model_type))), stat=$(stat_s), p=$(pv_s))" :
         title
    res = _make_plot([p1, p2, p3]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(res, save_path)
    res
end

# =============================================================================
# SufficientInformationTestResult
# =============================================================================

"""
    plot_result(r::SufficientInformationTestResult; level=5, ncols=0, title="",
                save_path=nothing)

Visualize the Forni & Gambetti (2014) informational sufficiency test across factor
counts ``r \\in 1\\dots r_{\\max}``:

1. System ``F``-test and HAC Wald ``\\chi^2`` p-values across `r_grid`, with horizontal
   reference lines at `α = level/100` (default 5%) and the overall Holm-adjusted
   `pvalue`.
2. System ``F``-statistics across `r_grid`.
3. Newey–West HAC Wald ``\\chi^2`` statistics across `r_grid`.
"""
function plot_result(r::SufficientInformationTestResult{T};
                     level::Int=5,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    0 < level < 100 || throw(ArgumentError("level must be in (0, 100), got $level"))
    alpha = level / 100.0
    rg = r.r_grid

    # Panel 1: F & Wald p-values across r_grid with alpha and overall Holm pvalue refs
    rows1 = Vector{Pair{String,String}}[
        ["x" => _json(rg[i]),
         "fp" => _json(i <= length(r.f_pvalues) ? Float64(r.f_pvalues[i]) : NaN),
         "wp" => _json(i <= length(r.wald_pvalues) ? Float64(r.wald_pvalues[i]) : NaN)]
        for i in eachindex(rg)
    ]
    s1 = _series_json(["F-test p-value", "HAC Wald p-value"],
                      [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                      keys=["fp", "wp"], dash=["", "6,3"])
    refs1 = String["{\"value\":$(_json(alpha)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"5,4\"}"]
    isfinite(r.pvalue) && push!(refs1,
        "{\"value\":$(_json(Float64(r.pvalue))),\"axis\":\"y\",\"color\":$(_json(_PLOT_SERIES[4])),\"dash\":\"2,2\"}")
    id1 = _next_plot_id("suff_pval")
    js1 = _render_line_js(id1, _json_array_of_objects(rows1), s1;
                          ref_lines_json="[" * join(refs1, ",") * "]",
                          integer_x=true, xlabel="Factor count r", ylabel="p-value")
    pv_f = isfinite(r.pvalue) ? _fmt(r.pvalue; digits=3) : "n/a"
    pv_w = isfinite(r.wald_pvalue) ? _fmt(r.wald_pvalue; digits=3) : "n/a"
    p1 = _PanelSpec(id1, "P-Values vs r (Holm F p=$(pv_f), Wald p=$(pv_w))", js1)

    # Panel 2: F-statistics across r_grid
    rows2 = Vector{Pair{String,String}}[
        ["x" => _json(rg[i]),
         "y" => _json(i <= length(r.f_statistics) ? Float64(r.f_statistics[i]) : NaN)]
        for i in eachindex(rg)
    ]
    id2 = _next_plot_id("suff_fstat")
    js2 = _render_vbar_js(id2, _json_array_of_objects(rows2);
                          bar_color=_PLOT_SERIES[1], integer_x=true,
                          xlabel="Factor count r", ylabel="F-statistic")
    p2 = _PanelSpec(id2, "System F-Statistics", js2)

    # Panel 3: HAC Wald statistics across r_grid
    rows3 = Vector{Pair{String,String}}[
        ["x" => _json(rg[i]),
         "y" => _json(i <= length(r.wald_statistics) ? Float64(r.wald_statistics[i]) : NaN)]
        for i in eachindex(rg)
    ]
    id3 = _next_plot_id("suff_wstat")
    js3 = _render_vbar_js(id3, _json_array_of_objects(rows3);
                          bar_color=_PLOT_SERIES[2], integer_x=true,
                          xlabel="Factor count r", ylabel="HAC Wald χ²")
    p3 = _PanelSpec(id3, "HAC Wald χ² Statistics", js3)

    ft = isempty(title) ?
         "Forni–Gambetti Informational Sufficiency Test (p_f=$(r.p_f), Holm F p=$(pv_f), Wald p=$(pv_w))" :
         title
    res = _make_plot([p1, p2, p3]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(res, save_path)
    res
end


