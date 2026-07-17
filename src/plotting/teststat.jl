# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Hypothesis-test plotting dispatches (D3.js, zero external deps).
# EV-30 (#438): the signature explosive-bubble chart — the backward sup-ADF
# (BSADF) sequence against its 95% critical-value sequence, with the stamped
# bubble episodes shaded. This is the standard PSY (2015) central-bank monitor.
# =============================================================================

"""
    plot_result(r::BubbleResult; title="", save_path=nothing)

Draw the sup-ADF bubble monitor: the BSADF sequence and its 95% critical-value
sequence as two lines (right-tailed — exuberance where BSADF pierces the CV),
with the date-stamped bubble [`episodes`](@ref BubbleResult) shaded. The x-axis
is the level index of `y` (`r2_index`), so shaded regions line up with the
user's calendar.
"""
function plot_result(r::BubbleResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("bubble")

    # Vertical shading extent for episodes (span the plotted y-range).
    yall = vcat(collect(r.bsadf), collect(r.cv_seq))
    yall = filter(isfinite, yall)
    ylo = isempty(yall) ? -1.0 : minimum(yall)
    yhi = isempty(yall) ? 1.0 : maximum(yall)
    ypad = (yhi - ylo) * 0.10 + eps(Float64)
    shade_lo = ylo - ypad
    shade_hi = yhi + ypad

    in_episode(idx) = any(ep -> ep[1] <= idx <= ep[2], r.episodes)

    rows = Vector{Pair{String,String}}[]
    for k in eachindex(r.r2_index)
        idx = r.r2_index[k]
        shaded = in_episode(idx)
        push!(rows, [
            "x" => _json(idx),
            "bsadf" => _json(r.bsadf[k]),
            "cv95" => _json(r.cv_seq[k]),
            "ep_lo" => (shaded ? _json(shade_lo) : "null"),
            "ep_hi" => (shaded ? _json(shade_hi) : "null"),
        ])
    end
    data_json = _json_array_of_objects(rows)

    seq_name = r.kind == :sadf ? "Recursive ADF" : "BSADF"
    s_json = _series_json([seq_name, "95% Critical Value"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["bsadf", "cv95"], dash=["", "6,3"])
    bands = "[{\"lo_key\":\"ep_lo\",\"hi_key\":\"ep_hi\",\"color\":\"$(_PLOT_COLORS[4])\",\"alpha\":0.18}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands,
                         xlabel="Observation index", ylabel="sup-ADF statistic")
    kind_name = r.kind == :sadf ? "SADF" : "GSADF"
    panels = [_PanelSpec(id, "$(kind_name) Bubble Monitor", js)]

    if isempty(title)
        n_ep = length(r.episodes)
        title = string(kind_name, " Explosive-Behaviour Monitor (",
                       n_ep, n_ep == 1 ? " episode" : " episodes", ", n=", r.nobs, ")")
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
