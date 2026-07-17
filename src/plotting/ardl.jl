# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for the nonlinear ARDL (NARDL, EV-09): cumulative dynamic
multipliers m⁺_h / m⁻_h with their bootstrap bands and the asymmetry (difference)
curve, one panel per asymmetric regressor.
"""

# =============================================================================
# NARDLMultipliers — one panel per asymmetric regressor
# =============================================================================

function _plot_nardl_multiplier_panel(mm::NARDLMultipliers{T}, i::Int) where {T}
    id = _next_plot_id("nardl_mult")
    has_ci = mm.nreps > 0
    rows = Vector{Pair{String,String}}[]
    for (h, hz) in enumerate(mm.horizons)
        row = Pair{String,String}[
            "x"    => _json(hz),
            "pos"  => _json(mm.m_pos[i, h]),
            "neg"  => _json(mm.m_neg[i, h]),
            "diff" => _json(mm.m_diff[i, h]),
        ]
        if has_ci
            push!(row, "plo" => _json(mm.m_pos_lo[i, h]))
            push!(row, "phi" => _json(mm.m_pos_hi[i, h]))
            push!(row, "nlo" => _json(mm.m_neg_lo[i, h]))
            push!(row, "nhi" => _json(mm.m_neg_hi[i, h]))
        end
        push!(rows, row)
    end
    data = _json_array_of_objects(rows)

    series = _series_json(["m⁺ (positive)", "m⁻ (negative)", "asymmetry m⁺−m⁻"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2], _PLOT_COLORS[3]];
                          keys=["pos", "neg", "diff"], dash=["", "", "6,3"])
    bands = has_ci ?
        "[{\"lo_key\":\"plo\",\"hi_key\":\"phi\",\"color\":\"$(_PLOT_COLORS[1])\"}," *
        "{\"lo_key\":\"nlo\",\"hi_key\":\"nhi\",\"color\":\"$(_PLOT_COLORS[2])\"}]" :
        "[]"
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    js = _render_line_js(id, data, series; bands_json=bands, ref_lines_json=refs,
                         xlabel="Horizon h", ylabel="Cumulative response of y")
    _PanelSpec(id, "Dynamic multipliers: $(mm.reg_names[i])", js)
end

"""
    plot_result(mm::NARDLMultipliers; view=:multipliers, title="", save_path=nothing)

Plot the NARDL cumulative dynamic multipliers. One panel per asymmetric regressor
shows `m⁺_h` and `m⁻_h` (with their bootstrap bands, if present) and the asymmetry
curve `m⁺_h − m⁻_h`, converging to θ⁺ / θ⁻ as `h` grows.
"""
function plot_result(mm::NARDLMultipliers{T}; view::Symbol=:multipliers,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view == :multipliers ||
        throw(ArgumentError("view must be :multipliers for NARDLMultipliers; got :$view"))
    panels = [_plot_nardl_multiplier_panel(mm, i) for i in eachindex(mm.reg_names)]
    isempty(title) && (title = "NARDL Cumulative Dynamic Multipliers")
    ncols = length(panels) == 1 ? 1 : 2
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::NARDLModel; view=:multipliers, H=24, bootstrap=false, title="",
                save_path=nothing, kwargs...)

Convenience: compute the cumulative dynamic multipliers of a fitted
[`NARDLModel`](@ref) out to horizon `H` and plot them. Defaults to `bootstrap=false`
for a fast preview; pass `bootstrap=true` (and an `rng`) for percentile bands.
"""
function plot_result(m::NARDLModel{T}; view::Symbol=:multipliers, H::Int=24,
                     bootstrap::Bool=false, title::String="",
                     save_path::Union{String,Nothing}=nothing, kwargs...) where {T}
    view == :multipliers ||
        throw(ArgumentError("view must be :multipliers for NARDLModel; got :$view"))
    mm = dynamic_multipliers(m, H; bootstrap=bootstrap, kwargs...)
    plot_result(mm; view=:multipliers, title=title, save_path=save_path)
end
