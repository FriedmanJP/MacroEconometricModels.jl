# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
PLT-37 — GMM / SMM moment-fit plot. `GMMModel`/`SMMModel` store only the sample-moment
discrepancy `g_bar` at the solution (≈ 0 at a good fit), NOT an empirical-vs-model
pair, so this draws the discrepancy per moment as a vertical bar against a zero line
with a J-test annotation (C6 — the docstring claims only what the struct backs). The
GARCH news-impact half of PLT-37 lives in `models.jl` (`view=:news_impact`).
"""

# =============================================================================
# AbstractGMMModel (GMMModel / SMMModel) — moment discrepancy bar
# =============================================================================

"""
    plot_result(m::AbstractGMMModel; title="", save_path=nothing)

Bar chart of the sample-moment discrepancy `g_bar` (one bar per moment condition,
indexed 1…`n_moments`) against a zero reference line — a well-fitting GMM/SMM estimate
sits near zero at every moment. The J-test of over-identifying restrictions
(`J = …, df = n_moments − n_params, p = …`) is annotated in the panel title via `_fmt`
/ `_format_pvalue` (C9). No empirical-vs-model overlay is drawn: the struct stores only
the discrepancy, not a paired series (C6).
"""
function plot_result(m::AbstractGMMModel;
                     title::String="", save_path::Union{String,Nothing}=nothing)
    g = m.g_bar
    id = _next_plot_id("gmm_g")
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(g)
        push!(rows, ["x" => _json(i), "y" => _json(g[i])])
    end
    data = _json_array_of_objects(rows)
    js = _render_vbar_js(id, data; bar_color=_PLOT_COLORS[1], integer_x=true,
                         tip_label="Moment", xlabel="Moment condition",
                         ylabel="Sample moment ḡ")

    df = m.n_moments - m.n_params
    pstr = isnan(m.J_pvalue) ? "n/a (identity W)" : _format_pvalue(m.J_pvalue)
    jnote = "J = $(_fmt(m.J_stat))  (df = $df, p = $pstr)"
    model_str = m isa SMMModel ? "SMM" : "GMM"
    ptitle = "$model_str moment fit — $jnote"

    isempty(title) && (title = "$model_str Moment Discrepancy")
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end
